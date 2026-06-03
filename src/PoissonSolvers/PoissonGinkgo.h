#ifndef IPPL_POISSON_GINKGO_H
#define IPPL_POISSON_GINKGO_H

#ifdef ENABLE_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <memory>
#include "PoissonSolvers/Poisson.h"
#include "Utility/IpplTimings.h"

namespace ippl {

template <typename FieldLHS, typename FieldRHS = FieldLHS, typename VFieldType = FieldLHS>
class PoissonGinkgo : public Poisson<FieldLHS, FieldRHS> {
    using Tlhs = typename FieldLHS::value_type;

public:
    using Base = Poisson<FieldLHS, FieldRHS>;
    constexpr static unsigned Dim = FieldLHS::dim;

    PoissonGinkgo() {
        this->params_m.add("max_iterations", 1000);
        this->params_m.add("tolerance", (Tlhs)1e-13);
        this->params_m.add("preconditioner", "jacobi"); // Can be "jacobi", "ilu", or "none"
    }

    void mergeParameters(const ParameterList& params) { this->params_m.merge(params); }
    void setRhs(FieldRHS& rhs) override { this->rhs_mp = &rhs; }
    void setLhs(FieldLHS& lhs) override { this->lhs_mp = &lhs; }
    void setGradient(VFieldType& grad) { this->grad_mp = &grad; }
    int getIterationCount() const { return itCount_; }

    void setup() {
        exec_ = gko::OmpExecutor::create();
        
        auto& mesh = this->rhs_mp->get_mesh();
        nx_ = mesh.getGridsize(0); ny_ = mesh.getGridsize(1); nz_ = mesh.getGridsize(2);
        
        auto& fl = this->rhs_mp->getLayout();
        const auto& local_domain = fl.getLocalNDIndex();
        nx_local_ = local_domain[0].length();
        ny_local_ = local_domain[1].length();
        nz_local_ = local_domain[2].length();
        local_size_ = nx_local_ * ny_local_ * nz_local_;

        std::string prec_type = this->params_m.template get<std::string>("preconditioner");

        if (prec_type != "none") {
            // Generate a local sub-matrix block for preconditioning on this rank
            gko::matrix_data<double, int> md{gko::dim<2>{(size_t)local_size_, (size_t)local_size_}};
            double dx = mesh.getMeshSpacing(0); double dy = mesh.getMeshSpacing(1); double dz = mesh.getMeshSpacing(2);
            double inv_dx2 = 1.0 / (dx * dx); double inv_dy2 = 1.0 / (dy * dy); double inv_dz2 = 1.0 / (dz * dz);
            double diag = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2);

            // Simple local 7-point stencil setup for the preconditioner block
            for (int i = 0; i < nx_local_; ++i) {
                for (int j = 0; j < ny_local_; ++j) {
                    for (int k = 0; k < nz_local_; ++k) {
                        int row = i * ny_local_ * nz_local_ + j * nz_local_ + k;
                        md.nonzeros.emplace_back(row, row, diag);
                        if (i > 0) md.nonzeros.emplace_back(row, (i-1)*ny_local_*nz_local_ + j*nz_local_ + k, -inv_dx2);
                        if (i < nx_local_-1) md.nonzeros.emplace_back(row, (i+1)*ny_local_*nz_local_ + j*nz_local_ + k, -inv_dx2);
                        if (j > 0) md.nonzeros.emplace_back(row, i*ny_local_*nz_local_ + (j-1)*nz_local_ + k, -inv_dy2);
                        if (j < ny_local_-1) md.nonzeros.emplace_back(row, i*ny_local_*nz_local_ + (j+1)*nz_local_ + k, -inv_dy2);
                        if (k > 0) md.nonzeros.emplace_back(row, i*ny_local_*nz_local_ + j*nz_local_ + (k-1), -inv_dz2);
                        if (k < nz_local_-1) md.nonzeros.emplace_back(row, i*ny_local_*nz_local_ + j*nz_local_ + (k+1), -inv_dz2);
                    }
                }
            }
            auto local_mat = gko::share(gko::matrix::Csr<double, int>::create(exec_));
            local_mat->read(md);

            if (prec_type == "jacobi") {
                prec_factory_ = gko::share(gko::preconditioner::Jacobi<double, int>::build().on(exec_)->generate(local_mat));
            } else if (prec_type == "ilu") {
                using L_Solver = gko::solver::LowerTrs<double, int>;
                using U_Solver = gko::solver::UpperTrs<double, int>;
                prec_factory_ = gko::share(gko::preconditioner::Ilu<L_Solver, U_Solver>::build().on(exec_)->generate(local_mat));
            }
        }
    }

    // High-performance local preconditioning hook
    void applyPreconditioner(FieldLHS& r, FieldLHS& z) {
        if (!prec_factory_) {
            z = r;
            return;
        }
        auto view_r = r.getView(); auto view_z = z.getView();
        int nghost = r.getNghost();

        auto gko_r = gko::matrix::Dense<double>::create(exec_, gko::dim<2>{(size_t)local_size_, 1});
        auto gko_z = gko::matrix::Dense<double>::create(exec_, gko::dim<2>{(size_t)local_size_, 1});

        double* r_vals = gko_r->get_values(); double* z_vals = gko_z->get_values();
        int ny = ny_local_, nz = nz_local_;

        Kokkos::parallel_for("PackPrec", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{nx_local_,ny_local_,nz_local_}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                r_vals[i*ny*nz + j*nz + k] = view_r(i+nghost, j+nghost, k+nghost);
            });
        Kokkos::fence();

        prec_factory_->apply(gko_r.get(), gko_z.get());

        Kokkos::parallel_for("UnpackPrec", Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{nx_local_,ny_local_,nz_local_}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                view_z(i+nghost, j+nghost, k+nghost) = z_vals[i*ny*nz + j*nz + k];
            });
        Kokkos::fence();
    }
    
    // Natively distributed dot product using IPPL Comm
    double getDotProduct(FieldLHS& a, FieldLHS& b) {
        auto view_a = a.getView();
        auto view_b = b.getView();
        int nghost = a.getNghost();
        int nx = nx_local_, ny = ny_local_, nz = nz_local_;

        double local_dot = 0.0;
        Kokkos::parallel_reduce("DotProduct", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nx, ny, nz}),
            KOKKOS_LAMBDA(const int i, const int j, const int k, double& lsum) {
                lsum += view_a(i + nghost, j + nghost, k + nghost) * view_b(i + nghost, j + nghost, k + nghost);
            }, Kokkos::Sum<double>(local_dot));
        Kokkos::fence();

        double global_dot = 0.0;
        ippl::Comm->reduce(local_dot, global_dot, 1, std::plus<double>());
        return global_dot;
    }

    void solve() override {
        if (!exec_) this->setup();

        FieldLHS& x = *(this->lhs_mp); FieldRHS& b = *(this->rhs_mp);
        FieldLHS r, d, q, s;
        r.initialize(b.get_mesh(), b.getLayout()); d.initialize(b.get_mesh(), b.getLayout());
        q.initialize(b.get_mesh(), b.getLayout()); s.initialize(b.get_mesh(), b.getLayout());

        int max_iters = this->params_m.template get<int>("max_iterations");
        double tol = this->params_m.template get<Tlhs>("tolerance");

        // 1. Distributed PCG Initializations
        x.fillHalo();
        r = b - (-laplace(x)); 
        this->applyPreconditioner(r, d); 
        
        double delta_new = getDotProduct(r, d); 
        double delta_0 = delta_new;
        itCount_ = 0;

        // 2. Globally Synchronized Solver Loop
        while (itCount_ < max_iters && delta_new > tol * tol * delta_0) {
            d.fillHalo(); // Crucial for distributed derivatives!
            q = -laplace(d);
            double alpha = delta_new / getDotProduct(d, q);
            
            x = x + alpha * d;
            r = r - alpha * q;
            
            this->applyPreconditioner(r, s);
            double delta_old = delta_new;
            delta_new = getDotProduct(r, s);
            
            double beta = delta_new / delta_old;
            d = s + beta * d;
            itCount_++;
        }

        if (this->grad_mp) {
            x.fillHalo();
            *(this->grad_mp) = -grad(x);
        }
    }

private:
    int itCount_ = 0;
    VFieldType* grad_mp = nullptr;
    int nx_ = 0, ny_ = 0, nz_ = 0;
    int nx_local_ = 0, ny_local_ = 0, nz_local_ = 0, local_size_ = 0;

    std::shared_ptr<gko::Executor> exec_;
    std::shared_ptr<gko::LinOp> prec_factory_;
};

} // namespace ippl
#endif // ENABLE_GINKGO
#endif // IPPL_POISSON_GINKGO_H
