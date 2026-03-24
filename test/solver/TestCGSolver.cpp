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

// ============================================================================
// STEP 1: THE GINKGO OPERATOR (MATRIX-FREE PHYSICS)
// ============================================================================
#ifdef ENABLE_GINKGO

#include <ginkgo/ginkgo.hpp>


template <typename FieldType>
class IpplLaplaceOp : public gko::EnableLinOp<IpplLaplaceOp<FieldType>>,
                      public gko::EnableCreateMethod<IpplLaplaceOp<FieldType>> {
public:
    IpplLaplaceOp(std::shared_ptr<const gko::Executor> exec)
        : gko::EnableLinOp<IpplLaplaceOp<FieldType>>(exec) {}

    IpplLaplaceOp(std::shared_ptr<const gko::Executor> exec, gko::dim<2> size,
                  FieldType& templated_field)
        : gko::EnableLinOp<IpplLaplaceOp<FieldType>>(exec, size)
    {
        temp_in_ = std::make_shared<FieldType>(templated_field.get_mesh(), templated_field.getLayout());
        temp_out_ = std::make_shared<FieldType>(templated_field.get_mesh(), templated_field.getLayout());
        temp_in_->setFieldBC(templated_field.getFieldBC());
        temp_out_->setFieldBC(templated_field.getFieldBC());
    }

protected:
    void apply_impl(const gko::LinOp* in, gko::LinOp* out) const override {
        auto dense_in = gko::as<gko::matrix::Dense<double>>(in);
        auto dense_out = gko::as<gko::matrix::Dense<double>>(out);

        auto view_in = temp_in_->getView();
        auto ext1 = view_in.extent(1); 
        auto ext2 = view_in.extent(2);
        const double* gko_in_data = dense_in->get_const_values();
        
        Kokkos::parallel_for("CopyIn", temp_in_->getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                size_t flat_idx = i * ext1 * ext2 + j * ext2 + k;
                view_in(i, j, k) = gko_in_data[flat_idx];
            });
        Kokkos::fence();

        *temp_out_ = -laplace(*temp_in_);
        Kokkos::fence();

        auto view_out = temp_out_->getView();
        double* gko_out_data = dense_out->get_values();
        
        Kokkos::parallel_for("CopyOut", temp_out_->getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                size_t flat_idx = i * ext1 * ext2 + j * ext2 + k;
                gko_out_data[flat_idx] = view_out(i, j, k);
            });
        Kokkos::fence();
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* in,
                    const gko::LinOp* beta, gko::LinOp* out) const override {
        auto dense_out = gko::as<gko::matrix::Dense<double>>(out);
        auto dense_alpha = gko::as<gko::matrix::Dense<double>>(alpha);
        auto dense_beta = gko::as<gko::matrix::Dense<double>>(beta);

        auto temp_result = gko::matrix::Dense<double>::create(this->get_executor(), dense_out->get_size());
        this->apply_impl(in, temp_result.get());
        dense_out->scale(dense_beta);
        dense_out->add_scaled(dense_alpha, temp_result.get());
    }

private:
    mutable std::shared_ptr<FieldType> temp_in_;
    mutable std::shared_ptr<FieldType> temp_out_;
};

// ============================================================================
// STEP 2: THE TRUE CSR GINKGO SOLVER CLASS (GPU + MEMORY BRIDGE)
// ============================================================================
template <typename FieldType>
class PoissonGinkgo {
public:
    using value_type = double;

    PoissonGinkgo() {
        params_m.add("solver_type", "cg");
        params_m.add("tolerance", 1e-9);
        params_m.add("max_iterations", 500);
    }

    void mergeParameters(ippl::ParameterList& params) { params_m.merge(params); }
    void setRhs(FieldType& rhs) { rhs_mp = &rhs; }
    void setLhs(FieldType& lhs) { lhs_mp = &lhs; }
    
    void setGridParams(int nx, int ny, int nz, double dx, double dy, double dz) {
        nx_ = nx; ny_ = ny; nz_ = nz;
        dx_ = dx; dy_ = dy; dz_ = dz;
    }
    
    int getIterationCount() const { return itCount_; }

    void setup() {
        // Safe Executor Bootup
        omp_exec_ = gko::OmpExecutor::create();
        try {
            exec_ = gko::CudaExecutor::create(0, omp_exec_);
        } catch (...) {
            std::cout << "CUDA failed, falling back to OMP." << std::endl;
            exec_ = omp_exec_;
        }

        std::string solver_type = params_m.template get<std::string>("solver_type");
        double tolerance = params_m.template get<double>("tolerance");
        int max_iters = params_m.template get<int>("max_iterations");

        int N = nx_ * ny_ * nz_;
        
        // Allocate Device memory natively in Ginkgo
        gko_b_ = gko::matrix::Dense<value_type>::create(exec_, gko::dim<2>{(size_t)N, 1});
        gko_x_ = gko::matrix::Dense<value_type>::create(exec_, gko::dim<2>{(size_t)N, 1});

        // 1. ASSEMBLE CSR MATRIX
        gko::matrix_data<value_type, int> md{gko::dim<2>{(size_t)N, (size_t)N}};
        double inv_dx2 = 1.0 / (dx_ * dx_);
        double inv_dy2 = 1.0 / (dy_ * dy_);
        double inv_dz2 = 1.0 / (dz_ * dz_);
        double diag = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2);

        for (int i = 0; i < nx_; ++i) {
            for (int j = 0; j < ny_; ++j) {
                for (int k = 0; k < nz_; ++k) {
                    int row = i * ny_ * nz_ + j * nz_ + k;
                    md.nonzeros.emplace_back(row, row, diag);

                    int im = (i == 0) ? nx_ - 1 : i - 1;
                    int ip = (i == nx_ - 1) ? 0 : i + 1;
                    md.nonzeros.emplace_back(row, im * ny_ * nz_ + j * nz_ + k, -inv_dx2);
                    md.nonzeros.emplace_back(row, ip * ny_ * nz_ + j * nz_ + k, -inv_dx2);

                    int jm = (j == 0) ? ny_ - 1 : j - 1;
                    int jp = (j == ny_ - 1) ? 0 : j + 1;
                    md.nonzeros.emplace_back(row, i * ny_ * nz_ + jm * nz_ + k, -inv_dy2);
                    md.nonzeros.emplace_back(row, i * ny_ * nz_ + jp * nz_ + k, -inv_dy2);

                    int km = (k == 0) ? nz_ - 1 : k - 1;
                    int kp = (k == nz_ - 1) ? 0 : k + 1;
                    md.nonzeros.emplace_back(row, i * ny_ * nz_ + j * nz_ + km, -inv_dz2);
                    md.nonzeros.emplace_back(row, i * ny_ * nz_ + j * nz_ + kp, -inv_dz2);
                }
            }
        }
        
        auto system_matrix = gko::share(gko::matrix::Csr<value_type, int>::create(exec_));
        system_matrix->read(md); 

        // 2. CONFIGURE SOLVER
        std::shared_ptr<gko::LinOpFactory> solver_factory;
        if (solver_type == "cg") {
            solver_factory = gko::solver::Cg<value_type>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(max_iters).on(exec_),
                    gko::stop::ResidualNorm<value_type>::build().with_reduction_factor(tolerance).on(exec_))
                .on(exec_);
        } else {
            solver_factory = gko::solver::Gmres<value_type>::build()
                .with_criteria(
                    gko::stop::Iteration::build().with_max_iters(max_iters).on(exec_),
                    gko::stop::ResidualNorm<value_type>::build().with_reduction_factor(tolerance).on(exec_))
                .on(exec_);
        }
        
        solver_ = solver_factory->generate(system_matrix);
        logger_ = gko::log::Convergence<value_type>::create();
        solver_->add_logger(logger_);
    }

    void solve() {
        int N = nx_ * ny_ * nz_;
        int nghost = rhs_mp->getNghost();
        auto view_rhs = rhs_mp->getView();
        auto view_lhs = lhs_mp->getView();
        int ny = ny_, nz = nz_; 

        // A. Pack data in IPPL's native memory space
        Kokkos::View<double*> staging_b("staging_b", N);
        Kokkos::View<double*> staging_x("staging_x", N);

        Kokkos::parallel_for("PackToGinkgo", rhs_mp->getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int ii = i - nghost;
                int jj = j - nghost;
                int kk = k - nghost;
                int row = ii * ny * nz + jj * nz + kk;
                staging_b(row) = view_rhs(i, j, k);
                staging_x(row) = view_lhs(i, j, k); 
            });
        Kokkos::fence();

        // B. Memory Bridge: Safely guarantee the data is on the CPU Host
        auto host_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), staging_b);
        auto host_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), staging_x);

        // C. Wrap Host memory for Ginkgo (using non-owning array views)
        auto gko_host_b = gko::matrix::Dense<value_type>::create(
            omp_exec_, gko::dim<2>{(size_t)N, 1}, 
            gko::array<value_type>::view(omp_exec_, N, host_b.data()), 1);
            
        auto gko_host_x = gko::matrix::Dense<value_type>::create(
            omp_exec_, gko::dim<2>{(size_t)N, 1}, 
            gko::array<value_type>::view(omp_exec_, N, host_x.data()), 1);

        // D. Push safely to GPU
        gko_b_->copy_from(gko_host_b.get());
        gko_x_->copy_from(gko_host_x.get());

        // E. ACTUAL GPU MATH 
        solver_->apply(gko_b_.get(), gko_x_.get());
        itCount_ = logger_->get_num_iterations();

        // F. Pull safely from GPU
        gko_host_x->copy_from(gko_x_.get());
        Kokkos::deep_copy(staging_x, host_x); // Push back to Kokkos native space

        // G. Unpack to IPPL Field
        Kokkos::parallel_for("UnpackFromGinkgo", lhs_mp->getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int ii = i - nghost;
                int jj = j - nghost;
                int kk = k - nghost;
                int row = ii * ny * nz + jj * nz + kk;
                view_lhs(i, j, k) = staging_x(row);
            });
        Kokkos::fence();
    }

private:
    FieldType* rhs_mp = nullptr;
    FieldType* lhs_mp = nullptr;
    ippl::ParameterList params_m;
    int itCount_ = 0;
    int nx_, ny_, nz_;
    double dx_, dy_, dz_;
    
    // Core state tracking
    std::shared_ptr<gko::Executor> omp_exec_;
    std::shared_ptr<gko::Executor> exec_;
    std::unique_ptr<gko::matrix::Dense<value_type>> gko_b_;
    std::unique_ptr<gko::matrix::Dense<value_type>> gko_x_;
    std::unique_ptr<gko::LinOp> solver_;
    std::shared_ptr<const gko::log::Convergence<value_type>> logger_;
};
#endif // ENABLE_GINKGO


// ============================================================================
// STEP 3: THE BENCHMARK (MAIN)
// ============================================================================
int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        constexpr unsigned int dim = 3;
        using Mesh_t               = ippl::UniformCartesian<double, 3>;
        using Centering_t          = Mesh_t::DefaultCentering;

        int pt = 4, ptY = 4;
        bool isWeak = false;
        std::string selected_solver = "native"; // Default
        
        Inform info("Config");
        if (argc >= 2) {
            double N = strtol(argv[1], NULL, 10);
            pt = ptY = 1 << (int)N;
            if (argc >= 3) {
                selected_solver = argv[2]; // Captures "native", "gko_cg", or "gko_gmres"
            }
        }
        
        info << "Grid Size (2^N): " << pt << endl;
        info << "Selected Solver Engine: " << selected_solver << endl;

        ippl::Index I(pt), Iy(ptY);
        ippl::NDIndex<dim> owned(I, Iy, I);
        std::array<bool, dim> isParallel = {true, true, true};
        ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

        double dx = 2.0 / double(pt); double dy = 2.0 / double(ptY);
        ippl::Vector<double, dim> hx = {dx, dy, dx};
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

        int shift1 = solution.getNghost();
        auto policySol = solution.getFieldRangePolicy();
        Kokkos::parallel_for("Assign solution", policySol, KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const size_t ig = i + lDom[0].first() - shift1;
            const size_t jg = j + lDom[1].first() - shift1;
            const size_t kg = k + lDom[2].first() - shift1;
            viewSol(i, j, k) = sin(sin(pi * (ig + 0.5) * hx[0])) * sin(sin(pi * (jg + 0.5) * hx[1])) * sin(sin(pi * (kg + 0.5) * hx[2]));
        });

        const int shift2 = rhs.getNghost();
        auto policyRHS   = rhs.getFieldRangePolicy();
        Kokkos::parallel_for("Assign rhs", policyRHS, KOKKOS_LAMBDA(const int i, const int j, const int k) {
            const size_t ig = i + lDom[0].first() - shift2;
            const size_t jg = j + lDom[1].first() - shift2;
            const size_t kg = k + lDom[2].first() - shift2;
            double x = (ig + 0.5) * hx[0]; double y = (jg + 0.5) * hx[1]; double z = (kg + 0.5) * hx[2];
            viewRHS(i, j, k) = pow(pi, 2) * (cos(sin(pi * z)) * sin(pi * z) * sin(sin(pi * x)) * sin(sin(pi * y))
                       + (cos(sin(pi * y)) * sin(pi * y) * sin(sin(pi * x))
                          + (cos(sin(pi * x)) * sin(pi * x) + (pow(cos(pi * x), 2) + pow(cos(pi * y), 2) + pow(cos(pi * z), 2))
                                   * sin(sin(pi * x))) * sin(sin(pi * y))) * sin(sin(pi * z)));
        });

        lhs = 0;
        int itCount = 0;

        // ====================================================================
        // THE BENCHMARK TOGGLE
        // ====================================================================
        if (selected_solver == "native") {
            ippl::PoissonCG<field_type> lapsolver;
            ippl::ParameterList params;
            params.add("max_iterations", 500);
            lapsolver.mergeParameters(params);
            lapsolver.setRhs(rhs); 
            lapsolver.setLhs(lhs);
            
            IpplTimings::TimerRef timer = IpplTimings::getTimer("1. SOLVE: Native IPPL CG");
            IpplTimings::startTimer(timer);
            lapsolver.solve();
            IpplTimings::stopTimer(timer);
            
            itCount = lapsolver.getIterationCount();
        } 
        else {
#ifdef ENABLE_GINKGO
PoissonGinkgo<field_type> lapsolver;
            ippl::ParameterList params;
            params.add("max_iterations", 500);
            
            if (selected_solver == "gko_cg") params.add("solver_type", "cg");
            else if (selected_solver == "gko_gmres") params.add("solver_type", "gmres");
            
            lapsolver.mergeParameters(params);
            lapsolver.setRhs(rhs); 
            lapsolver.setLhs(lhs);
            lapsolver.setGridParams(pt, ptY, pt, dx, dy, dx);

            // Time ONLY the setup phase
            IpplTimings::TimerRef setup_timer = IpplTimings::getTimer("1. Ginkgo Setup (One-time)");
            IpplTimings::startTimer(setup_timer);
            lapsolver.setup(); // <--- WE WILL NEED TO SPLIT THIS FUNCTION IN THE CLASS
            IpplTimings::stopTimer(setup_timer);

            // Time ONLY the solve phase
            std::string t_name = "2. SOLVE: Ginkgo " + selected_solver;
            IpplTimings::TimerRef timer = IpplTimings::getTimer(t_name.c_str());
            
            IpplTimings::startTimer(timer);
            lapsolver.solve(); 
            IpplTimings::stopTimer(timer);
            
            itCount = lapsolver.getIterationCount();
#else
            info << "ERROR: Built without Ginkgo. Use 'native'." << endl;
            std::exit(1);
#endif
        }

        // Error checking
        field_type error(mesh, layout);
        error = lhs - solution;
        double relError = norm(error) / norm(solution);
        error = -laplace(lhs) - rhs;
        double residue = norm(error) / norm(rhs);

        Inform m("Result");
        m << "\n=============================================" << endl;
        m << "Iterations:    " << itCount << endl;
        m << "Final Residue: " << residue << endl;
        m << "Relative Err:  " << relError << endl;
        m << "=============================================\n" << endl;
        
        IpplTimings::print();
    }
    ippl::finalize();
    return 0;
}