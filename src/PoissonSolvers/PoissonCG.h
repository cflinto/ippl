//
// Class PoissonCG
//   Solves the Poisson problem with the CG algorithm
//

#ifndef IPPL_POISSON_CG_H
#define IPPL_POISSON_CG_H

#include "LaplaceHelpers.h"
#include "LinearSolvers/PCG.h"
#include "Poisson.h"

#include <fstream>
#include <string>

namespace ippl {

// Expands to a lambda that acts as a wrapper for a differential operator
// fun: the function for which to create the wrapper, such as ippl::laplace
// type: the argument type, which should match the LHS type for the solver
#define IPPL_SOLVER_OPERATOR_WRAPPER(fun, type) \
    [](type arg) {                              \
        return fun(arg);                        \
    }

    template <typename FieldLHS, typename FieldRHS = FieldLHS>
    class PoissonCG : public Poisson<FieldLHS, FieldRHS> {
        using Tlhs = typename FieldLHS::value_type;

    public:
        using Base                    = Poisson<FieldLHS, FieldRHS>;
        constexpr static unsigned Dim = FieldLHS::dim;
        using typename Base::lhs_type, typename Base::rhs_type;
        using OperatorRet        = UnaryMinus<detail::meta_laplace<lhs_type>>;
        using LowerRet           = UnaryMinus<detail::meta_lower_laplace<lhs_type>>;
        using UpperRet           = UnaryMinus<detail::meta_upper_laplace<lhs_type>>;
        using UpperAndLowerRet   = UnaryMinus<detail::meta_upper_and_lower_laplace<lhs_type>>;
        using InverseDiagonalRet = double;
        using DiagRet            = double;

        PoissonCG()
            : Base()
            , algo_m(nullptr) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();
        }

        PoissonCG(lhs_type& lhs, rhs_type& rhs)
            : Base(lhs, rhs)
            , algo_m(nullptr) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();
            setSolver(*(this->lhs_mp));
        }

        void setLhs(lhs_type& lhs) override {
            Base::setLhs(lhs);
            setSolver(lhs);
        }

        void setSolver(lhs_type lhs) {
            std::string solver_type            = this->params_m.template get<std::string>("solver");
            typename lhs_type::Mesh_t mesh     = lhs.get_mesh();
            typename lhs_type::Layout_t layout = lhs.getLayout();
            double beta                        = 0;
            double alpha                       = 0;
            if (solver_type == "preconditioned") {
                algo_m = std::move(
                    std::make_unique<PCG<OperatorRet, LowerRet, UpperRet, UpperAndLowerRet,
                                         InverseDiagonalRet, DiagRet, FieldLHS, FieldRHS>>());
                std::string preconditioner_type =
                    this->params_m.template get<std::string>("preconditioner_type");
                int level    = this->params_m.template get<int>("newton_level");
                int degree   = this->params_m.template get<int>("chebyshev_degree");
                int inner    = this->params_m.template get<int>("gauss_seidel_inner_iterations");
                int outer    = this->params_m.template get<int>("gauss_seidel_outer_iterations");
                double omega = this->params_m.template get<double>("ssor_omega");
                int richardson_iterations =
                    this->params_m.template get<int>("richardson_iterations");
                int communication = this->params_m.template get<int>("communication");
                // Analytical eigenvalues for the d dimensional laplace operator
                // Going brute force through all possible eigenvalues seems to be the only way to
                // find max and min

                unsigned long n;
                double h;
                for (unsigned int d = 0; d < Dim; ++d) {
                    n                = mesh.getGridsize(d);
                    h                = mesh.getMeshSpacing(d);
                    double local_min = 4 / std::pow(h, 2);  // theoretical maximum
                    double local_max = 0;
                    double test;
                    for (unsigned int i = 1; i < n; ++i) {
                        test = 4. / std::pow(h, 2) * std::pow(std::sin(i * M_PI * h / 2.), 2);
                        if (test > local_max) {
                            local_max = test;
                        }
                        if (test < local_min) {
                            local_min = test;
                        }
                    }
                    beta += local_max;
                    alpha += local_min;
                }
                if (communication) {
                    algo_m->setPreconditioner(
                        IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-lower_laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-upper_laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-upper_and_lower_laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(negative_inverse_diagonal_laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(diagonal_laplace, lhs_type), alpha, beta,
                        preconditioner_type, level, degree, richardson_iterations, inner, outer,
                        omega);
                } else {
                    algo_m->setPreconditioner(
                        IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-lower_laplace_no_comm, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-upper_laplace_no_comm, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(-upper_and_lower_laplace_no_comm, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(negative_inverse_diagonal_laplace, lhs_type),
                        IPPL_SOLVER_OPERATOR_WRAPPER(diagonal_laplace, lhs_type), alpha, beta,
                        preconditioner_type, level, degree, richardson_iterations, inner, outer,
                        omega);
                }
            } else {
                algo_m = std::move(
                    std::make_unique<CG<OperatorRet, LowerRet, UpperRet, UpperAndLowerRet,
                                        InverseDiagonalRet, DiagRet, FieldLHS, FieldRHS>>());
            }
            algo_m->initializeFields(lhs.get_mesh(), lhs.getLayout());
        }

        void solve() override {
            // \todo TODO add a check for mesh changes for alpha and beta for preconditioners

            algo_m->setOperator(IPPL_SOLVER_OPERATOR_WRAPPER(-laplace, lhs_type));
            
            // This is the line that actually runs the solver math
            algo_m->operator()(*(this->lhs_mp), *(this->rhs_mp), this->params_m);

            // =================================================================
            // START OF DATA DUMP INSTRUMENTATION
            // =================================================================
            {
                if (ippl::Comm->rank() == 0) {
                    auto& fl = this->rhs_mp->getLayout();
                    const auto& local_domain = fl.getLocalNDIndex();
                    std::printf("IPPL LAYOUT DECOMPOSITION FOR STANDALONE:\n");
                    std::printf("local_Nx = %d\n", (int)local_domain[0].length());
                    std::printf("local_Ny = %d\n", (int)local_domain[1].length());
                    std::printf("local_Nz = %d\n", (int)local_domain[2].length());
                    std::printf("========================================\n\n");
                }

                // Print the exact layout boundaries for every rank
                auto& fl = this->rhs_mp->getLayout();
                const auto& local_domain = fl.getLocalNDIndex();
                std::printf("LAYOUT_CONSTR rank=%d start=%d,%d,%d size=%d,%d,%d\n",
                            ippl::Comm->rank(),
                            (int)local_domain[0].first(), (int)local_domain[1].first(), (int)local_domain[2].first(),
                            (int)local_domain[0].length(), (int)local_domain[1].length(), (int)local_domain[2].length());

                // 1. Get the underlying Kokkos Views for the RHS (Input) and LHS (Output)
                auto view_rhs = this->rhs_mp->getView(); 
                auto view_lhs = this->lhs_mp->getView(); 

                // 2. Create Host Mirrors (Safely copies memory to CPU if running on GPUs)
                auto host_rhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view_rhs);
                auto host_lhs = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view_lhs);

                // 3. Get the MPI rank to prevent files from overwriting each other
                int rank = ippl::Comm->rank();
                
                // Track which solve step this is
                static int step = 0;
                
                // 4. Generate unique filenames
                std::string in_file = "poisson_rhs_step" + std::to_string(step) + "_rank" + std::to_string(rank) + ".bin";
                std::string out_file = "poisson_lhs_step" + std::to_string(step) + "_rank" + std::to_string(rank) + ".bin";

                // 5. Dump the raw binary arrays (stripping IPPL abstraction)
                std::ofstream fout_rhs(in_file, std::ios::binary);
                if (fout_rhs.is_open()) {
                    fout_rhs.write(reinterpret_cast<const char*>(host_rhs.data()), host_rhs.span() * sizeof(Tlhs));
                    fout_rhs.close();
                }

                std::ofstream fout_lhs(out_file, std::ios::binary);
                if (fout_lhs.is_open()) {
                    fout_lhs.write(reinterpret_cast<const char*>(host_lhs.data()), host_lhs.span() * sizeof(Tlhs));
                    fout_lhs.close();
                }
                
                step++;
            }
            // =================================================================
            // END OF DATA DUMP INSTRUMENTATION
            // =================================================================

            int output = this->params_m.template get<int>("output_type");
            if (output & Base::GRAD) {
                *(this->grad_mp) = -grad(*(this->lhs_mp));
            }
        }

        /*!
         * Query how many iterations were required to obtain the solution
         * the last time this solver was used
         * @return Iteration count of last solve
         */
        int getIterationCount() { return algo_m->getIterationCount(); }

        /*!
         * Query the residue
         * @return Residue norm from last solve
         */
        Tlhs getResidue() const { return algo_m->getResidue(); }

    protected:
        std::unique_ptr<CG<OperatorRet, LowerRet, UpperRet, UpperAndLowerRet, InverseDiagonalRet,
                           DiagRet, FieldLHS, FieldRHS>>
            algo_m;

        void setDefaultParameters() override {
            this->params_m.add("max_iterations", 2000);
            this->params_m.add("tolerance", (Tlhs)1e-13);
            this->params_m.add("solver", "non-preconditioned");
        }
    };

}  // namespace ippl

#endif
