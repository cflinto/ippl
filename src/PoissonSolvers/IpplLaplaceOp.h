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
                  FieldType& templated_field, int nx, int ny, int nz, int nghost)
        : gko::EnableLinOp<IpplLaplaceOp<FieldType>>(exec, size),
          nx_(nx), ny_(ny), nz_(nz), nghost_(nghost)
    {
        temp_in_ = std::make_shared<FieldType>(templated_field.get_mesh(), templated_field.getLayout());
        temp_out_ = std::make_shared<FieldType>(templated_field.get_mesh(), templated_field.getLayout());
        temp_in_->setFieldBC(templated_field.getFieldBC());
        temp_out_->setFieldBC(templated_field.getFieldBC());
    }

protected:
    void apply_impl(const gko::LinOp* in, gko::LinOp* out) const override {
        static IpplTimings::TimerRef packTimer = IpplTimings::getTimer("Ginkgo MF: Pack/Unpack");
        static IpplTimings::TimerRef mathTimer = IpplTimings::getTimer("Ginkgo MF: Laplace Math");

        auto dense_in = gko::as<gko::matrix::Dense<double>>(in);
        auto dense_out = gko::as<gko::matrix::Dense<double>>(out);
        auto view_in = temp_in_->getView();
        const double* gko_in_data = dense_in->get_const_values();
        int nghost = nghost_; int ny = ny_; int nz = nz_;
        
        IpplTimings::startTimer(packTimer);
        Kokkos::parallel_for("CopyIn", temp_in_->getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int row = (i - nghost) * ny * nz + (j - nghost) * nz + (k - nghost);
                view_in(i, j, k) = gko_in_data[row];
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
        Kokkos::parallel_for("CopyOut", temp_out_->getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                int row = (i - nghost) * ny * nz + (j - nghost) * nz + (k - nghost);
                gko_out_data[row] = view_out(i, j, k);
            });
        Kokkos::fence();
        IpplTimings::stopTimer(packTimer);
    }

    void apply_impl(const gko::LinOp* alpha, const gko::LinOp* in,
                    const gko::LinOp* beta, gko::LinOp* out) const override {
        auto dense_out = gko::as<gko::matrix::Dense<double>>(out);
        auto dense_alpha = gko::as<gko::matrix::Dense<double>>(alpha);
        auto dense_beta = gko::as<gko::matrix::Dense<double>>(beta);

        if (!temp_math_result_) {
            temp_math_result_ = gko::share(gko::matrix::Dense<double>::create(this->get_executor(), dense_out->get_size()));
        }
        
        this->apply_impl(in, temp_math_result_.get());
        dense_out->scale(dense_beta);
        dense_out->add_scaled(dense_alpha, temp_math_result_.get());
    }

private:
    mutable std::shared_ptr<gko::matrix::Dense<double>> temp_math_result_;
    mutable std::shared_ptr<FieldType> temp_in_;
    mutable std::shared_ptr<FieldType> temp_out_;
    int nx_, ny_, nz_, nghost_;
};

} // namespace ippl

#endif // ENABLE_GINKGO
#endif // IPPL_LAPLACE_OP_H
