#ifndef IPPL_POISSON_GINKGO_H
#define IPPL_POISSON_GINKGO_H

#define ENABLE_GINKGO // TODO delete

#ifdef ENABLE_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <mpi.h>

template <typename FieldType>
class PoissonGinkgo {
public:
    using value_type = double;
    using local_index_type = gko::int32;
    using global_index_type = gko::int64; // Strictly required by Ginkgo's distributed module

    PoissonGinkgo() {
        params_m.add("solver_type", "cg");
        params_m.add("tolerance", 1e-13);
        params_m.add("max_iterations", 500);
        params_m.add("matrix_free", false);
        params_m.add("preconditioner", "none"); 
    }

    void mergeParameters(const ippl::ParameterList& params) { params_m.merge(params); }
    void setRhs(FieldType& rhs) { rhs_mp = &rhs; }
    void setLhs(FieldType& lhs) { lhs_mp = &lhs; }
    
    int getIterationCount() const { return itCount_; }

    void setup() {
        comm_ = gko::experimental::mpi::communicator(MPI_COMM_WORLD);
        exec_ = gko::OmpExecutor::create();

        auto& mesh = rhs_mp->get_mesh();
        dx_ = mesh.getMeshSpacing(0);
        dy_ = mesh.getMeshSpacing(1);
        dz_ = mesh.getMeshSpacing(2);
        
        auto& fl = rhs_mp->getLayout();
        const auto& local_domain = fl.getLocalNDIndex();
        
        int nx_local = local_domain[0].length();
        int ny_local = local_domain[1].length();
        int nz_local = local_domain[2].length();
        
        const auto& global_domain = fl.getDomain();
        int nx_global = global_domain[0].length();
        int ny_global = global_domain[1].length();
        int nz_global = global_domain[2].length();

        local_size_ = nx_local * ny_local * nz_local;
        global_size_ = (global_index_type)nx_global * ny_global * nz_global;

        std::string solver_type = params_m.template get<std::string>("solver_type");
        double tolerance = params_m.template get<double>("tolerance");
        int max_iters = params_m.template get<int>("max_iterations");

        // Build MPI Partition
        auto partition = gko::share(gko::experimental::distributed::Partition<local_index_type, global_index_type>::build_from_global_size_uniform(
            exec_, comm_.size(), global_size_));

        gko::matrix_data<value_type, global_index_type> md;
        md.size = gko::dim<2>{(size_t)global_size_, (size_t)global_size_};

        double inv_dx2 = 1.0 / (dx_ * dx_); double inv_dy2 = 1.0 / (dy_ * dy_); double inv_dz2 = 1.0 / (dz_ * dz_);
        double diag = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2);

        int start_x = local_domain[0].first();
        int start_y = local_domain[1].first();
        int start_z = local_domain[2].first();

        for (int i = 0; i < nx_local; ++i) {
            for (int j = 0; j < ny_local; ++j) {
                for (int k = 0; k < nz_local; ++k) {
                    
                    int gi = start_x + i;
                    int gj = start_y + j;
                    int gk = start_z + k;
                    
                    global_index_type global_row = gi * ny_global * nz_global + gj * nz_global + gk;
                    
                    md.nonzeros.emplace_back(global_row, global_row, diag);

                    int gim = (gi == 0) ? nx_global - 1 : gi - 1;
                    int gip = (gi == nx_global - 1) ? 0 : gi + 1;
                    md.nonzeros.emplace_back(global_row, gim * ny_global * nz_global + gj * nz_global + gk, -inv_dx2);
                    md.nonzeros.emplace_back(global_row, gip * ny_global * nz_global + gj * nz_global + gk, -inv_dx2);

                    int gjm = (gj == 0) ? ny_global - 1 : gj - 1;
                    int gjp = (gj == ny_global - 1) ? 0 : gj + 1;
                    md.nonzeros.emplace_back(global_row, gi * ny_global * nz_global + gjm * nz_global + gk, -inv_dy2);
                    md.nonzeros.emplace_back(global_row, gi * ny_global * nz_global + gjp * nz_global + gk, -inv_dy2);

                    int gkm = (gk == 0) ? nz_global - 1 : gk - 1;
                    int gkp = (gk == nz_global - 1) ? 0 : gk + 1;
                    md.nonzeros.emplace_back(global_row, gi * ny_global * nz_global + gj * nz_global + gkm, -inv_dz2);
                    md.nonzeros.emplace_back(global_row, gi * ny_global * nz_global + gj * nz_global + gkp, -inv_dz2);
                }
            }
        }

        auto dist_mat = gko::share(gko::experimental::distributed::Matrix<value_type, local_index_type, global_index_type>::create(exec_, comm_));
        // Pass the shared_ptr directly to fix the read_distributed compile error
        dist_mat->read_distributed(md, partition);
        system_matrix_ = dist_mat;

        auto stop_criterion = gko::stop::Iteration::build().with_max_iters(max_iters).on(exec_);
        auto res_criterion = gko::stop::ResidualNorm<value_type>::build().with_reduction_factor(tolerance).on(exec_);

        auto shared_stop = gko::share(std::move(stop_criterion));
        auto shared_res = gko::share(std::move(res_criterion));

        if (solver_type == "cg") {
            auto cg_builder = gko::solver::Cg<value_type>::build().with_criteria(shared_stop, shared_res);
            solver_ = cg_builder.on(exec_)->generate(system_matrix_);
        } else {
            auto gmres_builder = gko::solver::Gmres<value_type>::build().with_criteria(shared_stop, shared_res);
            solver_ = gmres_builder.on(exec_)->generate(system_matrix_);
        }
        
        logger_ = gko::log::Convergence<value_type>::create();
        solver_->add_logger(logger_);
    }

    void solve() {
        if (!solver_) this->setup();

        auto view_rhs = rhs_mp->getView();
        auto view_lhs = lhs_mp->getView();
        int nghost = rhs_mp->getNghost();
        
        auto& fl = rhs_mp->getLayout();
        const auto& local_domain = fl.getLocalNDIndex();
        int nx_local = local_domain[0].length();
        int ny_local = local_domain[1].length();
        int nz_local = local_domain[2].length();
        
        // Allocate local dense arrays
        auto local_b_dense = gko::matrix::Dense<value_type>::create(exec_, gko::dim<2>{(size_t)local_size_, 1});
        auto local_x_dense = gko::matrix::Dense<value_type>::create(exec_, gko::dim<2>{(size_t)local_size_, 1});
        
        value_type* b_vals = local_b_dense->get_values();
        value_type* x_vals = local_x_dense->get_values();

        // Unmanaged memory traits prevent Kokkos from trying to double-free Ginkgo's raw pointers
        Kokkos::View<value_type*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> b_view(b_vals, local_size_);
        Kokkos::View<value_type*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> x_view(x_vals, local_size_);

        Kokkos::parallel_for("PackDistributed", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nx_local, ny_local, nz_local}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int row = i * ny_local * nz_local + j * nz_local + k;
                b_view(row) = view_rhs(i + nghost, j + nghost, k + nghost);
                x_view(row) = view_lhs(i + nghost, j + nghost, k + nghost); 
            });
        Kokkos::fence();

        // Create Distributed Vectors by moving the dense arrays
        auto gko_b = gko::experimental::distributed::Vector<value_type>::create(
            exec_, comm_, gko::dim<2>{(size_t)global_size_, 1}, std::move(local_b_dense));
            
        auto gko_x = gko::experimental::distributed::Vector<value_type>::create(
            exec_, comm_, gko::dim<2>{(size_t)global_size_, 1}, std::move(local_x_dense));

        // debug: dump the first few values
        if (comm_.rank() == 0) {
            auto b_host = gko_b->get_local_vector()->get_const_values();
            std::cout << "First 10 values of b: ";
            for (int i = 0; i < std::min(10, local_size_); ++i) {
                std::cout << b_host[i] << " ";
            }
            std::cout << std::endl;
        }

        solver_->apply(gko_b.get(), gko_x.get());
        itCount_ = logger_->get_num_iterations();

        // Unpack back into the local IPPL view
        auto out_x_dense = gko_x->get_local_vector();
        const value_type* out_vals = out_x_dense->get_const_values();
        
        Kokkos::View<const value_type*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> out_view(out_vals, local_size_);

        Kokkos::parallel_for("UnpackDistributed", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nx_local, ny_local, nz_local}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int row = i * ny_local * nz_local + j * nz_local + k;
                view_lhs(i + nghost, j + nghost, k + nghost) = out_view(row);
            });
        Kokkos::fence();
    }

private:
    FieldType* rhs_mp = nullptr; 
    FieldType* lhs_mp = nullptr;
    
    ippl::ParameterList params_m;
    int itCount_ = 0;
    
    double dx_ = 0.0, dy_ = 0.0, dz_ = 0.0;
    int local_size_ = 0;
    global_index_type global_size_ = 0;
    
    gko::experimental::mpi::communicator comm_{MPI_COMM_NULL};
    std::shared_ptr<gko::Executor> exec_;
    std::shared_ptr<const gko::LinOp> system_matrix_;
    std::unique_ptr<gko::LinOp> solver_;
    std::shared_ptr<const gko::log::Convergence<value_type>> logger_;
};
#endif // ENABLE_GINKGO
#endif // IPPL_POISSON_GINKGO_H
