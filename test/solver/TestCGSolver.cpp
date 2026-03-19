// Tests the conjugate gradient solver for Poisson problems
// by checking the relative error from the exact solution
// Usage:
//      TestCGSolver [size [scaling_type , preconditioner]]
//      ./TestCGSolver 6 j --info 5

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <typeinfo>

#include "Utility/Inform.h"
#include "Utility/IpplTimings.h"

#include "PoissonSolvers/PoissonCG.h"

#ifdef ENABLE_GINKGO
#include <ginkgo/ginkgo.hpp>
#endif



#ifdef ENABLE_GINKGO
// Custom Ginkgo Operator that wraps IPPL's matrix-free Laplacian
template <typename FieldType>
class IpplLaplaceOp : public gko::EnableLinOp<IpplLaplaceOp<FieldType>>,
                      public gko::EnableCreateMethod<IpplLaplaceOp<FieldType>> {
public:
    // 1. Required by Ginkgo's polymorphic architecture
    IpplLaplaceOp(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<IpplLaplaceOp<FieldType>>(exec) {}

    // 2. Our actual physics constructor
    IpplLaplaceOp(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
                  FieldType& templated_field)
        : gko::EnableLinOp<IpplLaplaceOp<FieldType>>(exec, size)
    {
        // Use shared_ptr so the class remains copy-assignable (required by Ginkgo!)
        temp_in_ = std::make_shared<FieldType>(templated_field.get_mesh(), templated_field.getLayout());
        temp_out_ = std::make_shared<FieldType>(templated_field.get_mesh(), templated_field.getLayout());
        
        // Copy boundary conditions so the physics remain correct
        temp_in_->setFieldBC(templated_field.getFieldBC());
        temp_out_->setFieldBC(templated_field.getFieldBC());
    }

protected:
    // This is the basic function: out = A * in
    void apply_impl(const gko::LinOp* in, gko::LinOp* out) const override {
        auto dense_in = gko::as<gko::matrix::Dense<double>>(in);
        auto dense_out = gko::as<gko::matrix::Dense<double>>(out);

        auto view_in = temp_in_->getView();
        auto ext1 = view_in.extent(1); 
        auto ext2 = view_in.extent(2);
        const double* gko_in_data = dense_in->get_const_values();
        
        // 1. Move Ginkgo's current vector state into the IPPL Field
        Kokkos::parallel_for("CopyIn", temp_in_->getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                size_t flat_idx = i * ext1 * ext2 + j * ext2 + k;
                view_in(i, j, k) = gko_in_data[flat_idx];
            });
        Kokkos::fence(); // Wait for copy

        // 2. APPLY IPPL'S NATIVE MATRIX-FREE LAPLACIAN!
        *temp_out_ = -laplace(*temp_in_);
        Kokkos::fence(); // Wait for physics

        // 3. Move the physics result back into Ginkgo's vector
        auto view_out = temp_out_->getView();
        double* gko_out_data = dense_out->get_values();
        
        Kokkos::parallel_for("CopyOut", temp_out_->getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                size_t flat_idx = i * ext1 * ext2 + j * ext2 + k;
                gko_out_data[flat_idx] = view_out(i, j, k);
            });
        Kokkos::fence();
    }

    // The advanced function: out = alpha * A * in + beta * out
    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* in,
                    const gko::LinOp* beta, gko::LinOp* out) const override {
        auto dense_out = gko::as<gko::matrix::Dense<double>>(out);
        auto dense_alpha = gko::as<gko::matrix::Dense<double>>(alpha);
        auto dense_beta = gko::as<gko::matrix::Dense<double>>(beta);

        // 1. Create a temporary Ginkgo vector to hold A * in
        auto temp_result = gko::matrix::Dense<double>::create(
            this->get_executor(), dense_out->get_size());

        // 2. Call our basic apply_impl to compute (A * in) -> temp_result
        this->apply_impl(in, temp_result.get());

        // 3. Scale the existing out vector: out = beta * out
        dense_out->scale(dense_beta);

        // 4. Add the new result: out = out + alpha * temp_result
        dense_out->add_scaled(dense_alpha, temp_result.get());
    }

private:
    mutable std::shared_ptr<FieldType> temp_in_;
    mutable std::shared_ptr<FieldType> temp_out_;
};
#endif



int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, 3>;
        using Centering_t          = Mesh_t::DefaultCentering;

        int pt = 4, ptY = 4;
        bool isWeak = false;
        // Preconditioner Setup Start
        int gauss_seidel_inner_iterations;
        int gauss_seidel_outer_iterations;
        int newton_level;
        int chebyshev_degree;
        int richardson_iterations;
        int communication;
        double ssor_omega;
        std::string solver              = "not preconditioned";
        std::string preconditioner_type = "";
        // Preconditioner Setup End
        Inform info("Config");
        if (argc >= 2) {
            // First argument is the problem size (log2)
            double N = strtol(argv[1], NULL, 10);
            info << "Got " << N << " as size parameter" << endl;
            pt = ptY = 1 << (int)N;
            if (argc >= 3) {
                if (argv[2][0] == 'w') {
                    // If weak scaling is specified, increase the problem size
                    // along the Y axis such that each rank has the same workload
                    // (the simplest enlargement method)
                    ptY = 1 << (5 + (int)N);
                    pt  = 32;
                    info << "Performing weak scaling" << endl;
                    isWeak = true;
                } else {
                    if (argv[2][0] == 'j') {
                        solver              = "preconditioned";
                        preconditioner_type = "jacobi";
                    }
                    if (argv[2][0] == 'n') {
                        solver              = "preconditioned";
                        preconditioner_type = "newton";
                        newton_level        = std::atoi(argv[3]);
                    }
                    if (argv[2][0] == 'c') {
                        solver              = "preconditioned";
                        preconditioner_type = "chebyshev";
                        chebyshev_degree    = std::atoi(argv[3]);
                    }
                    if (argv[2][0] == 'g') {
                        solver                        = "preconditioned";
                        preconditioner_type           = "gauss-seidel";
                        gauss_seidel_inner_iterations = std::atoi(argv[3]);
                        gauss_seidel_outer_iterations = std::atoi(argv[4]);
                        communication                 = std::atoi(argv[5]);
                    }
                    if (argv[2][0] == 's') {
                        solver                        = "preconditioned";
                        preconditioner_type           = "ssor";
                        gauss_seidel_inner_iterations = std::atoi(argv[3]);
                        gauss_seidel_outer_iterations = std::atoi(argv[4]);
                        ssor_omega                    = std::stod(argv[5]);
                    }
                    if (argv[2][0] == 'r') {
                        solver                = "preconditioned";
                        preconditioner_type   = "richardson";
                        richardson_iterations = std::atoi(argv[3]);
                        communication         = std::atoi(argv[4]);
                    }
                }
                if (argc >= 4) {
                    if (argv[3][0] == 'j') {
                        solver              = "preconditioned";
                        preconditioner_type = "jacobi";
                    }
                    if (argv[3][0] == 'n') {
                        solver              = "preconditioned";
                        preconditioner_type = "newton";
                        newton_level        = std::atoi(argv[4]);
                    }
                    if (argv[3][0] == 'c') {
                        solver              = "preconditioned";
                        preconditioner_type = "chebyshev";
                        chebyshev_degree    = std::atoi(argv[4]);
                    }
                    if (argv[3][0] == 'g') {
                        solver                        = "preconditioned";
                        preconditioner_type           = "gauss-seidel";
                        gauss_seidel_inner_iterations = std::atoi(argv[4]);
                        gauss_seidel_outer_iterations = std::atoi(argv[5]);
                        communication                 = std::atoi(argv[6]);
                    }
                    if (argv[3][0] == 's') {
                        solver                        = "preconditioned";
                        preconditioner_type           = "ssor";
                        gauss_seidel_inner_iterations = std::atoi(argv[4]);
                        gauss_seidel_outer_iterations = std::atoi(argv[5]);
                        ssor_omega                    = std::stod(argv[6]);
                    }
                    if (argv[3][0] == 'r') {
                        solver                = "preconditioned";
                        preconditioner_type   = "richardson";
                        richardson_iterations = std::atoi(argv[4]);
                        communication         = std::atoi(argv[5]);
                    }
                }
            }
        }
        info << "Solver is " << solver << endl;
        if (solver == "preconditioned") {
            info << "Preconditioner is " << preconditioner_type << endl;
        }

        ippl::Index I(pt), Iy(ptY);
        ippl::NDIndex<dim> owned(I, Iy, I);

        std::array<bool, dim> isParallel;  // Specifies SERIAL, PARALLEL dims
        for (unsigned int d = 0; d < dim; d++) {
            isParallel[d] = true;
        }

        // all parallel layout, standard domain, normal axis order
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        // Unit box
        double dx                        = 2.0 / double(pt);
        double dy                        = 2.0 / double(ptY);
        ippl::Vector<double, dim> hx     = {dx, dy, dx};
        ippl::Vector<double, dim> origin = -1;
        Mesh_t mesh(owned, hx, origin);

        double pi = Kokkos::numbers::pi_v<double>;

        typedef ippl::Field<double, dim, Mesh_t, Centering_t> field_type;
        field_type rhs(mesh, layout), lhs(mesh, layout), solution(mesh, layout);

        typedef ippl::BConds<field_type, dim> bc_type;

        bc_type bcField;

        for (unsigned int i = 0; i < 6; ++i) {
            bcField[i] = std::make_shared<ippl::PeriodicFace<field_type>>(i);
        }

        lhs.setFieldBC(bcField);

        typename field_type::view_type &viewRHS = rhs.getView(), viewSol = solution.getView();

        const ippl::NDIndex<dim>& lDom = layout.getLocalNDIndex();

        using Kokkos::pow, Kokkos::sin, Kokkos::cos;

        int shift1     = solution.getNghost();
        auto policySol = solution.getFieldRangePolicy();
        Kokkos::parallel_for(
            "Assign solution", policySol, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const size_t ig = i + lDom[0].first() - shift1;
                const size_t jg = j + lDom[1].first() - shift1;
                const size_t kg = k + lDom[2].first() - shift1;
                double x        = (ig + 0.5) * hx[0];
                double y        = (jg + 0.5) * hx[1];
                double z        = (kg + 0.5) * hx[2];

                viewSol(i, j, k) = sin(sin(pi * x)) * sin(sin(pi * y)) * sin(sin(pi * z));
            });

        const int shift2 = rhs.getNghost();
        auto policyRHS   = rhs.getFieldRangePolicy();
        Kokkos::parallel_for(
            "Assign rhs", policyRHS, KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const size_t ig = i + lDom[0].first() - shift2;
                const size_t jg = j + lDom[1].first() - shift2;
                const size_t kg = k + lDom[2].first() - shift2;
                double x        = (ig + 0.5) * hx[0];
                double y        = (jg + 0.5) * hx[1];
                double z        = (kg + 0.5) * hx[2];

                // https://gitlab.psi.ch/OPAL/Libraries/ippl-solvers/-/blob/5-fftperiodicpoissonsolver/test/TestFFTPeriodicPoissonSolver.cpp#L91
                viewRHS(i, j, k) =
                    pow(pi, 2)
                    * (cos(sin(pi * z)) * sin(pi * z) * sin(sin(pi * x)) * sin(sin(pi * y))
                       + (cos(sin(pi * y)) * sin(pi * y) * sin(sin(pi * x))
                          + (cos(sin(pi * x)) * sin(pi * x)
                             + (pow(cos(pi * x), 2) + pow(cos(pi * y), 2) + pow(cos(pi * z), 2))
                                   * sin(sin(pi * x)))
                                * sin(sin(pi * y)))
                             * sin(sin(pi * z)));
            });

        ippl::PoissonCG<field_type> lapsolver;

        ippl::ParameterList params;
        params.add("max_iterations", 500);
        params.add("solver", solver);
        // Preconditioner Setup
        params.add("preconditioner_type", preconditioner_type);
        params.add("gauss_seidel_inner_iterations", gauss_seidel_inner_iterations);
        params.add("gauss_seidel_outer_iterations", gauss_seidel_outer_iterations);
        params.add("newton_level", newton_level);
        params.add("chebyshev_degree", chebyshev_degree);
        params.add("richardson_iterations", richardson_iterations);
        params.add("communication", communication);
        params.add("ssor_omega", ssor_omega);

        lapsolver.mergeParameters(params);

        lapsolver.setRhs(rhs);
        lapsolver.setLhs(lhs);

        lhs = 0;
        info << "Solver is set up" << endl;

#ifdef ENABLE_GINKGO
        // Test: Just intercept the data and wrap it in Ginkgo views.
        info << "--- GINKGO INTERCEPTION START ---" << endl;

        auto gko_exec = gko::ReferenceExecutor::create();
        auto gko_viewLHS = lhs.getView();
        auto gko_viewRHS = rhs.getView();

        // Calculate total 1D size of the 3D grid
        size_t total_size = gko_viewLHS.extent(0) * gko_viewLHS.extent(1) * gko_viewLHS.extent(2);

        // Wrap the raw Kokkos pointers into Ginkgo Dense Matrices
        // Avoid copy by constructing the view inline (as a temporary).
        auto gko_x = gko::matrix::Dense<double>::create(
            gko_exec, gko::dim<2>{total_size, 1}, 
            gko::array<double>::view(gko_exec, total_size, gko_viewLHS.data()), 
            1); // 1 is the stride
            
        auto gko_b = gko::matrix::Dense<double>::create(
            gko_exec, gko::dim<2>{total_size, 1}, 
            gko::array<double>::view(gko_exec, total_size, gko_viewRHS.data()), 
            1); // 1 is the stride

        info << "Successfully wrapped Kokkos Views in Ginkgo Dense vectors!" << endl;
        
        // Let's be paranoid: verify that the Ginkgo views are indeed pointing to the same memory as the Kokkos views, and that mutating one mutates the other. This is the crux of zero-copy interoperability.

        // Proof 1: Check the raw hardware memory addresses
        const double* kokkos_ptr = gko_viewRHS.data();
        const double* ginkgo_ptr = gko_b->get_const_values();
        
        info << "Kokkos memory address: " << kokkos_ptr << endl;
        info << "Ginkgo memory address: " << ginkgo_ptr << endl;
        
        if (kokkos_ptr == ginkgo_ptr) {
            info << "SUCCESS: Addresses match! Ginkgo is using Kokkos memory directly." << endl;
        } else {
            info << "WARNING: Addresses differ. A copy occurred!" << endl;
        }

        // Proof 2: Mutate the data in Kokkos and see if Ginkgo instantly sees it
        double original_val = gko_viewRHS(0, 0, 0);
        gko_viewRHS(0, 0, 0) = 9999.99; // Mutate via Kokkos
        
        double ginkgo_val = gko_b->at(0, 0); // Read back via Ginkgo
        
        info << "Mutated value in Kokkos: 9999.99" << endl;
        info << "Value read from Ginkgo:  " << ginkgo_val << endl;
        
        gko_viewRHS(0, 0, 0) = original_val; // Restore the physics!
        // ==========================================

        // Prove Ginkgo can read the data that Kokkos generated!
        auto gko_b_norm = gko::matrix::Dense<double>::create(gko_exec, gko::dim<2>{1, 1});
        gko_b->compute_norm2(gko_b_norm);
        
        info << "Ginkgo computed RHS Norm2: " << gko_b_norm->at(0, 0) << endl;
        info << "--- GINKGO INTERCEPTION END ---" << endl;

        info << "Setting up Ginkgo CG Solver..." << endl;
        
        // Instantiate our custom IPPL-wrapper operator
        auto A_matrix = gko::share(IpplLaplaceOp<field_type>::create(gko_exec, gko::dim<2>{total_size, total_size}, lhs));

        // Build Ginkgo's CG Solver
        auto gko_cg_factory = gko::solver::Cg<double>::build()
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(500u).on(gko_exec),
                gko::stop::ResidualNorm<double>::build().with_reduction_factor(1e-9).on(gko_exec)
            )
            .on(gko_exec);
            
        auto gko_solver = gko_cg_factory->generate(A_matrix);

        // 3. GINKGO SOLVES THE SYSTEM!
        info << "Ginkgo is solving A * x = b ..." << endl;
        gko_solver->apply(gko_b, gko_x);
        
        info << "Ginkgo solver finished!" << endl;
        info << "--- GINKGO INTERCEPTION END ---" << endl;
#endif

        // Debug: Compute the norm with IPPL, to check it's the same as Ginkgo's (it should be, since they point to the same data)
        double ippl_rhs_norm = norm(rhs);
        info << "IPPL computed RHS Norm2:   " << ippl_rhs_norm << endl;

        // IPPL solve: (we try to get rid of it!)
        // lapsolver.solve(); 
        
        info << "Solver is done" << endl;

        const char* name = isWeak ? "Convergence (weak)" : "Convergence";
        Inform m(name);

        field_type error(mesh, layout);
        // Solver solution - analytical solution
        error           = lhs - solution;
        double relError = norm(error) / norm(solution);

        // Laplace(solver solution) - rhs
        error          = -laplace(lhs) - rhs;
        double residue = norm(error) / norm(rhs);

        int size    = isWeak ? pt * pt * ptY : pt;
        int itCount = lapsolver.getIterationCount();
        m << size << "," << std::setprecision(16) << relError << "," << residue << "," << itCount
          << endl;
        IpplTimings::print();
        // IpplTimings::print("timings" + std::to_string(pt) + ".dat");
    }
    ippl::finalize();

    return 0;
}