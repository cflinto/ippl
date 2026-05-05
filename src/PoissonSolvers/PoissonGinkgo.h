#ifndef IPPL_POISSON_GINKGO_H
#define IPPL_POISSON_GINKGO_H

#ifdef ENABLE_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <memory>
#include "PoissonSolvers/Poisson.h"
#include "PoissonSolvers/IpplLaplaceOp.h" 
#include "Utility/IpplTimings.h"

namespace ippl {

// We now pass the Vector Field type directly so we don't have to guess it!
template <typename FieldLHS, typename FieldRHS = FieldLHS, typename VFieldType = FieldLHS>
class PoissonGinkgo : public Poisson<FieldLHS, FieldRHS> {
    using Tlhs = typename FieldLHS::value_type;

public:
    using Base = Poisson<FieldLHS, FieldRHS>;
    constexpr static unsigned Dim = FieldLHS::dim;

    PoissonGinkgo() {
        this->params_m.add("solver_type", "cg");
        this->params_m.add("max_iterations", 1000);
        this->params_m.add("tolerance", (Tlhs)1e-13);
        this->params_m.add("preconditioner", "none"); 
    }

    void mergeParameters(const ParameterList& params) { this->params_m.merge(params); }
    void setRhs(FieldRHS& rhs) override { this->rhs_mp = &rhs; }
    void setLhs(FieldLHS& lhs) override { this->lhs_mp = &lhs; }
    
    // Gradient method now uses the provided VFieldType safely
    void setGradient(VFieldType& grad) { this->grad_mp = &grad; }
    
    int getIterationCount() const { return itCount_; }

    void setup() {
        exec_ = gko::OmpExecutor::create();
        
        auto& mesh = this->rhs_mp->get_mesh();
        int nx = mesh.getGridsize(0);
        int ny = mesh.getGridsize(1);
        int nz = mesh.getGridsize(2);
        int N = nx * ny * nz;

        int nghost = this->lhs_mp->getNghost();

        system_matrix_ = gko::share(IpplLaplaceOp<FieldLHS>::create(
            exec_, gko::dim<2>{(size_t)N, (size_t)N}, *(this->lhs_mp), nx, ny, nz, nghost));

        int max_iters = this->params_m.template get<int>("max_iterations");
        double tol = this->params_m.template get<Tlhs>("tolerance");
        std::string solver_type = this->params_m.template get<std::string>("solver_type");

        auto stop_criterion = gko::stop::Iteration::build().with_max_iters(max_iters).on(exec_);
        auto res_criterion = gko::stop::ResidualNorm<Tlhs>::build().with_reduction_factor(tol).on(exec_);

        auto shared_stop = gko::share(std::move(stop_criterion));
        auto shared_res = gko::share(std::move(res_criterion));

        if (solver_type == "cg") {
            auto builder = gko::solver::Cg<Tlhs>::build().with_criteria(shared_stop, shared_res);
            solver_ = builder.on(exec_)->generate(system_matrix_);
        } else {
            auto builder = gko::solver::Gmres<Tlhs>::build().with_criteria(shared_stop, shared_res);
            solver_ = builder.on(exec_)->generate(system_matrix_);
        }
        
        logger_ = gko::log::Convergence<Tlhs>::create();
        solver_->add_logger(logger_);
    }

    void solve() override {
        if (!solver_) {
            this->setup();
        }

        int N = system_matrix_->get_size()[0];

        auto view_rhs = this->rhs_mp->getView();
        int nghost = this->rhs_mp->getNghost();
        int ny = this->rhs_mp->get_mesh().getGridsize(1);
        int nz = this->rhs_mp->get_mesh().getGridsize(2);
        
        Kokkos::View<double*> staging_b("staging_b", N);
        Kokkos::View<double*> staging_x("staging_x", N);
        
        auto local_b = staging_b;
        auto local_x = staging_x;
        
        Kokkos::parallel_for("PackRHS", this->rhs_mp->getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int row = (i - nghost) * ny * nz + (j - nghost) * nz + (k - nghost);
                local_b(row) = view_rhs(i, j, k);
                local_x(row) = 0.0;
            });
        Kokkos::fence();

        auto gko_b = gko::share(gko::matrix::Dense<Tlhs>::create(
            exec_, gko::dim<2>{(size_t)N, 1}, gko::array<Tlhs>::view(exec_, N, staging_b.data()), 1));
        auto gko_x = gko::share(gko::matrix::Dense<Tlhs>::create(
            exec_, gko::dim<2>{(size_t)N, 1}, gko::array<Tlhs>::view(exec_, N, staging_x.data()), 1));

        solver_->apply(gko_b.get(), gko_x.get());
        itCount_ = logger_->get_num_iterations();

        auto view_lhs = this->lhs_mp->getView();
        Kokkos::parallel_for("UnpackLHS", this->lhs_mp->getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int row = (i - nghost) * ny * nz + (j - nghost) * nz + (k - nghost);
                view_lhs(i, j, k) = local_x(row);
            });
        Kokkos::fence();

        if (this->grad_mp) {
            this->lhs_mp->fillHalo();
            *(this->grad_mp) = -grad(*(this->lhs_mp));
        }
    }

private:
    int itCount_ = 0;
    VFieldType* grad_mp = nullptr; 
    
    std::shared_ptr<gko::Executor> exec_;
    std::shared_ptr<const gko::LinOp> system_matrix_;
    std::unique_ptr<gko::LinOp> solver_;
    std::shared_ptr<const gko::log::Convergence<Tlhs>> logger_;
};

} // namespace ippl

#endif // ENABLE_GINKGO
#endif // IPPL_POISSON_GINKGO_H
