#ifndef IPPL_LAPLACE_OP_H
#define IPPL_LAPLACE_OP_H

#ifdef ENABLE_GINKGO

#include <ginkgo/ginkgo.hpp>
#include <memory>
#include "Utility/IpplTimings.h"

namespace ippl {

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
        // Fix the Assertion 'layout_m != 0' failed by initializing properly!
        temp_in_ = std::make_shared<FieldType>();
        temp_in_->initialize(templated_field.get_mesh(), templated_field.getLayout());
        
        temp_out_ = std::make_shared<FieldType>();
        temp_out_->initialize(templated_field.get_mesh(), templated_field.getLayout());
        
        temp_in_->setFieldBC(templated_field.getFieldBC());
        temp_out_->setFieldBC(templated_field.getFieldBC());
    }

protected:
    void apply_impl(const gko::LinOp* in, gko::LinOp* out) const override {
        static IpplTimings::TimerRef packTimer = IpplTimings::getTimer("Ginkgo MF: Pack/Unpack");
        static IpplTimings::TimerRef mathTimer = IpplTimings::getTimer("Ginkgo MF: Laplace Math");

        // The vectors passed by Ginkgo's distributed CG are distributed vectors!
        auto dist_in = gko::as<gko::experimental::distributed::Vector<double>>(in);
        auto dist_out = gko::as<gko::experimental::distributed::Vector<double>>(out);
        
        // Get the local dense pieces of those vectors
        auto dense_in = dist_in->get_local_vector();
        auto dense_out = dist_out->get_local_vector();

        auto view_in = temp_in_->getView();
        const double* gko_in_data = dense_in->get_const_values();
        
        int nghost = temp_in_->getNghost();
        auto& fl = temp_in_->getLayout();
        const auto& local_domain = fl.getLocalNDIndex();
        int nx_local = local_domain[0].length();
        int ny_local = local_domain[1].length();
        int nz_local = local_domain[2].length();
        
        IpplTimings::startTimer(packTimer);
        Kokkos::parallel_for("CopyIn", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nx_local, ny_local, nz_local}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int row = i * ny_local * nz_local + j * nz_local + k;
                view_in(i + nghost, j + nghost, k + nghost) = gko_in_data[row];
            });
        Kokkos::fence();
        IpplTimings::stopTimer(packTimer);
        
        IpplTimings::startTimer(mathTimer);
        *temp_out_ = -laplace(*temp_in_);
        Kokkos::fence();
        IpplTimings::stopTimer(mathTimer);

        IpplTimings::startTimer(packTimer);
        auto view_out = temp_out_->getView();
        double* gko_out_data = dense_out->get_values();
        Kokkos::parallel_for("CopyOut", 
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {nx_local, ny_local, nz_local}),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int row = i * ny_local * nz_local + j * nz_local + k;
                gko_out_data[row] = view_out(i + nghost, j + nghost, k + nghost);
            });
        Kokkos::fence();
        IpplTimings::stopTimer(packTimer);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* in,
                    const gko::LinOp* beta, gko::LinOp* out) const override {
        auto dist_out = gko::as<gko::experimental::distributed::Vector<double>>(out);
        auto dense_alpha = gko::as<gko::matrix::Dense<double>>(alpha);
        auto dense_beta = gko::as<gko::matrix::Dense<double>>(beta);

        if (!temp_math_result_) {
            // Clone the vector to guarantee exact MPI distributed sizing
            temp_math_result_ = std::dynamic_pointer_cast<gko::experimental::distributed::Vector<double>>(
                gko::share(dist_out->clone())
            );
        }
        
        this->apply_impl(in, temp_math_result_.get());
        dist_out->scale(dense_beta);
        dist_out->add_scaled(dense_alpha, temp_math_result_.get());
    }

private:
    mutable std::shared_ptr<gko::experimental::distributed::Vector<double>> temp_math_result_;
    mutable std::shared_ptr<FieldType> temp_in_;
    mutable std::shared_ptr<FieldType> temp_out_;
};

} // namespace ippl

#endif // ENABLE_GINKGO
#endif // IPPL_LAPLACE_OP_H
