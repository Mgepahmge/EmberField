/** @file SpmvPlan.cu @brief 实现共享只读快照与可复用cuSPARSE资源。 */
#include <Numerics/LinearAlgebra/SpmvPlan.cuh>
#include <cusparse.h>
/** @brief 线性代数内部稀疏执行资源实现。 */
namespace emberfield::linalg::detail {
    namespace {
        /** @brief 检查库状态。@param status 库返回值。@pre 无。
         * @return 无，失败runtime_error。@code check(CUSPARSE_STATUS_SUCCESS); @endcode */
        void check(cusparseStatus_t status) {
            if (status != CUSPARSE_STATUS_SUCCESS) {
                throw std::runtime_error("cuSPARSE SpMV failure: " + std::to_string(static_cast<int>(status)));
            }
        }
        /** @brief 检查CUDA状态。@param status 返回值。@pre 无。@return 无，失败CudaError。
         * @code check(cudaGetLastError()); @endcode */
        void check(cudaError_t status) {
            if (status != cudaSuccess) {
                throw CudaError(status, "SpMV");
            }
        }
    } // namespace
    /** @brief 描述符先释放，随后释放其数组。@tparam T 精度。@tparam I 索引。
     * @code Impl state(frozen); @endcode */
    template <Scalar T, CsrIndex I>
    struct SpmvPlan<T, I>::Impl {
        CanonicalCsrSnapshot<T, I> snapshot_; ///< 矩阵强所有权。
        Vector<T> input_, output_; ///< 初始合法绑定。
        CudaBuffer<std::byte> workspace_; ///< 固定操作工作区。
        cusparseHandle_t sparse_{}; ///< 稀疏句柄。
        cusparseSpMatDescr_t a_{}; ///< 固定矩阵描述符。
        cusparseDnVecDescr_t x_{}, y_{}; ///< 可重绑稠密描述符。
        int device_{}; ///< 所属设备。
        /** @brief 构建资源。@param snapshot 不可变来源。@pre 已切换所属设备。
         * @return 无。@code Impl state(frozen); @endcode */
        explicit Impl(CanonicalCsrSnapshot<T, I> snapshot) :
            snapshot_(std::move(snapshot)), input_(snapshot_.matrix().columnCount()),
            output_(snapshot_.matrix().rowCount()) {
            const auto& matrix_ = snapshot_.matrix();
            check(cudaGetDevice(&device_));
            try {
                check(cusparseCreate(&sparse_));
                if (matrix_.rowCount() && matrix_.columnCount() && matrix_.nonZeroCount()) {
                    const auto index = sizeof(I) == 4 ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I;
                    const auto type = std::same_as<T, float> ? CUDA_R_32F : CUDA_R_64F;
                    check(cusparseCreateCsr(&a_, matrix_.rowCount(), matrix_.columnCount(), matrix_.nonZeroCount(),
                                            const_cast<I*>(matrix_.rowOffsets().deviceData()),
                                            const_cast<I*>(matrix_.columnIndices().deviceData()),
                                            const_cast<T*>(matrix_.values().deviceData()), index, index,
                                            CUSPARSE_INDEX_BASE_ZERO, type));
                    check(cusparseCreateDnVec(&x_, input_.size(), input_.deviceData(), type));
                    check(cusparseCreateDnVec(&y_, output_.size(), output_.deviceData(), type));
                    const T one = 1, zero = 0;
                    std::size_t size{};
                    check(cusparseSpMV_bufferSize(sparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, a_, x_, &zero, y_,
                                                  type, CUSPARSE_SPMV_ALG_DEFAULT, &size));
                    workspace_ = CudaBuffer<std::byte>(size);
                }
            }
            catch (...) {
                release();
                throw;
            }
        }
        /** @brief 释放库资源，无参数；当前设备为所属设备。@return 无，不抛异常。
         * @code state.release(); @endcode */
        void release() noexcept {
            if (y_) {
                cusparseDestroyDnVec(y_);
            }
            if (x_) {
                cusparseDestroyDnVec(x_);
            }
            if (a_) {
                cusparseDestroySpMat(a_);
            }
            if (sparse_) {
                cusparseDestroy(sparse_);
            }
        }
        /** @brief RAII释放，无参数无前提。@return 无。@code { Impl s(frozen); } @endcode */
        ~Impl() noexcept {
            int previous{};
            cudaGetDevice(&previous);
            cudaSetDevice(device_);
            release();
            cudaSetDevice(previous);
        }
    };
    template <Scalar T, CsrIndex I>
    SpmvPlan<T, I>::SpmvPlan(CanonicalCsrSnapshot<T, I> snapshot) {
        librarySize(snapshot.matrix().rowCount());
        librarySize(snapshot.matrix().columnCount());
        OperationScope scope(snapshot.matrix().rowOffsets().deviceId());
        impl_ = std::make_unique<Impl>(std::move(snapshot));
    }
    template <Scalar T, CsrIndex I>
    SpmvPlan<T, I>::~SpmvPlan() noexcept = default;
    template <Scalar T, CsrIndex I>
    const CsrMatrix<T, I>& SpmvPlan<T, I>::matrix() const noexcept {
        return impl_->snapshot_.matrix();
    }
    template <Scalar T, CsrIndex I>
    void SpmvPlan<T, I>::multiply(const Vector<T>& x, Vector<T>& y) {
        EMBERFIELD_NVTX_SCOPE("emberfield::linalg::detail::SpmvPlan::multiply");
        const auto& a = matrix();
        if (x.size() != a.columnCount() || y.size() != a.rowCount() || x.memoryType() != MemoryType::Global ||
            y.memoryType() != MemoryType::Global || (x.size() && x.deviceId() != impl_->device_) ||
            (y.size() && y.deviceId() != impl_->device_) || (x.size() && x.data() == y.data())) {
            throw std::invalid_argument("SpMV shape, device, storage or alias mismatch");
        }
        OperationScope scope(impl_->device_);
        if (!impl_->a_) {
            zero(y.deviceData(), y.size());
            return;
        }
        check(cusparseDnVecSetValues(impl_->x_, const_cast<T*>(x.deviceData())));
        check(cusparseDnVecSetValues(impl_->y_, y.deviceData()));
        const T one = 1, zeroValue = 0;
        check(cusparseSpMV(impl_->sparse_, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, impl_->a_, impl_->x_, &zeroValue,
                           impl_->y_, std::same_as<T, float> ? CUDA_R_32F : CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                           impl_->workspace_.deviceData()));
        check(cudaStreamSynchronize(nullptr));
    }
    template class SpmvPlan<float, std::int32_t>;
    template class SpmvPlan<double, std::int32_t>;
    template class SpmvPlan<float, std::int64_t>;
    template class SpmvPlan<double, std::int64_t>;
} // namespace emberfield::linalg::detail
