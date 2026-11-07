/** @file IterationContext.cu
 * @brief 实现缓存句柄、SpMV 描述符及 Jacobi CUDA 更新内核。
 */
#include <NVTXProfiler.cuh>
#include <Numerics/LinearAlgebra/IterationContext.cuh>
#include <Numerics/LinearAlgebra/SpmvPlan.cuh>
#include <cmath>
#include <cublas_v2.h>
#include <cusparse.h>

namespace emberfield::linalg {
    /** @brief 提供执行上下文的内部错误检查与 CUDA 内核。 */
    namespace {
        /** @brief 检查库状态。@tparam S 状态枚举。@param status 返回状态。
         * @pre 零表示成功。@note 无返回值。@code check(cublasCreate(&handle)); @endcode
         */
        template <class S>
            requires std::is_enum_v<S>
        void check(S status) {
            if (static_cast<int>(status) != 0) {
                throw std::runtime_error("CUDA iteration library failure: " + std::to_string(static_cast<int>(status)));
            }
        }
        /** @brief 逐元素更新 Jacobi 近似解。@tparam T 精度。@param n 长度。@param w 权重。
         * @param d 逆对角线。@param b 右端项。@param a 矩阵向量积。@param x 近似解。
         * @pre 设备地址有效且等长。@note 无返回值。@code update<<<1,256>>>(n,w,d,b,a,x); @endcode
         */
        template <Scalar T>
        __global__ void update(int n, T w, const T* d, const T* b, const T* a, T* x) {
            const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < static_cast<unsigned>(n)) {
                x[i] += w * d[i] * (b[i] - a[i]);
            }
        }
    } // namespace

    /** @brief 独占库资源及矩阵快照，释放时恢复调用方设备。
     * @code IterationContext<double>::Impl state(matrix); @endcode
     */
    template <Scalar T, CsrIndex I>
    struct IterationContext<T, I>::Impl {
        detail::SpmvPlan<T, I> spmv_; ///< 共享快照与独立稀疏执行资源。
        const CsrMatrix<T, I>& matrix_; ///< 借用spmv_拥有的矩阵，兼容旧只读接口。
        cublasHandle_t blas_{}; ///< 持久化BLAS句柄。
        int device_{}; ///< 资源所属设备。
        /** @brief 建立所有资源。@param matrix 输入。
         * @pre 当前设备正确。@note 无返回值。@code Impl state(matrix); @endcode
         */
        explicit Impl(detail::CanonicalCsrSnapshot<T, I> matrix) : spmv_(std::move(matrix)), matrix_(spmv_.matrix()) {
            EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::linalg::IterationContext<T, I>::Impl::Impl",
                                        emberfield::NvtxColor::Cyan);
            check(cudaGetDevice(&device_));
            try {
                check(cublasCreate(&blas_));
                check(cublasSetMathMode(blas_, CUBLAS_PEDANTIC_MATH));
            }
            catch (...) {
                release();
                throw;
            }
        }
        /** @brief 释放句柄，无参数、无额外约束。@note 无返回值。
         * @code state.release(); @endcode
         */
        void release() noexcept {
            EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::linalg::IterationContext<T, I>::Impl::release",
                                        emberfield::NvtxColor::Cyan);
            if (blas_) {
                cublasDestroy(blas_);
            }
        }
        /** @brief 在所属设备释放资源，无参数、无额外约束。@note 无返回值。
         * @code { Impl state(matrix); } @endcode
         */
        ~Impl() noexcept {
            int previous{};
            cudaGetDevice(&previous);
            cudaSetDevice(device_);
            release();
            cudaSetDevice(previous);
        }
        /** @brief 检查工作向量。@param v 向量。@pre 无额外约束。
         * @note 无返回值。@code state.validate(v); @endcode
         */
        void validate(const Vector<T>& v) const {
            detail::librarySize(v.size());
            if (v.memoryType() != MemoryType::Global || (v.size() && v.deviceId() != device_)) {
                throw std::invalid_argument("Iteration workspace requires same-device global vectors");
            }
        }
    };

    template <Scalar T, CsrIndex I>
    IterationContext<T, I>::IterationContext(const CsrMatrix<T, I>& matrix) {
        detail::librarySize(matrix.rowCount());
        detail::librarySize(matrix.columnCount());
        detail::OperationScope scope(matrix.rowOffsets().deviceId());
        impl_ = std::make_unique<Impl>(detail::freezeCanonical(matrix));
    }
    template <Scalar T, CsrIndex I>
    IterationContext<T, I>::IterationContext(detail::CanonicalCsrSnapshot<T, I> snapshot) {
        detail::librarySize(snapshot.matrix().rowCount());
        detail::librarySize(snapshot.matrix().columnCount());
        detail::OperationScope scope(snapshot.matrix().rowOffsets().deviceId());
        impl_ = std::make_unique<Impl>(std::move(snapshot));
    }
    template <Scalar T, CsrIndex I>
    IterationContext<T, I>::~IterationContext() noexcept = default;
    template <Scalar T, CsrIndex I>
    const CsrMatrix<T, I>& IterationContext<T, I>::matrix() const noexcept {
        return impl_->matrix_;
    }
    template <Scalar T, CsrIndex I>
    void IterationContext<T, I>::multiply(const Vector<T>& x, Vector<T>& y) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::linalg::IterationContext<T, I>::multiply",
                                    emberfield::NvtxColor::Cyan);
        impl_->spmv_.multiply(x, y);
    }
    template <Scalar T, CsrIndex I>
    void IterationContext<T, I>::clear(Vector<T>& x) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::linalg::IterationContext<T, I>::clear", emberfield::NvtxColor::Cyan);
        impl_->validate(x);
        detail::OperationScope scope(impl_->device_);
        detail::zero(x.deviceData(), x.size());
    }
    template <Scalar T, CsrIndex I>
    void IterationContext<T, I>::residual(const Vector<T>& rhs, const Vector<T>& x, Vector<T>& out) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::linalg::IterationContext<T, I>::residual",
                                    emberfield::NvtxColor::Cyan);
        impl_->validate(rhs);
        if (rhs.size() != out.size() || (rhs.size() && rhs.data() == out.data())) {
            throw std::invalid_argument("Residual shape or alias mismatch");
        }
        multiply(x, out);
        scale(T(-1), out);
        axpy(T(1), rhs, out);
    }
    template <Scalar T, CsrIndex I>
    void IterationContext<T, I>::axpy(T alpha, const Vector<T>& x, Vector<T>& y) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::linalg::IterationContext<T, I>::axpy", emberfield::NvtxColor::Cyan);
        impl_->validate(x);
        impl_->validate(y);
        if (x.size() != y.size()) {
            throw std::invalid_argument("AXPY shape mismatch");
        }
        if (x.empty()) {
            return;
        }
        detail::OperationScope scope(impl_->device_);
        if constexpr (std::same_as<T, float>) {
            check(
                cublasSaxpy(impl_->blas_, detail::librarySize(x.size()), &alpha, x.deviceData(), 1, y.deviceData(), 1));
        }
        else {
            check(
                cublasDaxpy(impl_->blas_, detail::librarySize(x.size()), &alpha, x.deviceData(), 1, y.deviceData(), 1));
        }
        check(cudaStreamSynchronize(nullptr));
    }
    template <Scalar T, CsrIndex I>
    void IterationContext<T, I>::scale(T alpha, Vector<T>& x) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::linalg::IterationContext<T, I>::scale", emberfield::NvtxColor::Cyan);
        impl_->validate(x);
        if (x.empty()) {
            return;
        }
        detail::OperationScope scope(impl_->device_);
        if constexpr (std::same_as<T, float>) {
            check(cublasSscal(impl_->blas_, detail::librarySize(x.size()), &alpha, x.deviceData(), 1));
        }
        else {
            check(cublasDscal(impl_->blas_, detail::librarySize(x.size()), &alpha, x.deviceData(), 1));
        }
        check(cudaStreamSynchronize(nullptr));
    }
    template <Scalar T, CsrIndex I>
    T IterationContext<T, I>::dot(const Vector<T>& x, const Vector<T>& y) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::linalg::IterationContext<T, I>::dot", emberfield::NvtxColor::Cyan);
        impl_->validate(x);
        impl_->validate(y);
        if (x.size() != y.size()) {
            throw std::invalid_argument("Dot shape mismatch");
        }
        T result{};
        if (x.empty()) {
            return result;
        }
        detail::OperationScope scope(impl_->device_);
        if constexpr (std::same_as<T, float>) {
            check(
                cublasSdot(impl_->blas_, detail::librarySize(x.size()), x.deviceData(), 1, y.deviceData(), 1, &result));
        }
        else {
            check(
                cublasDdot(impl_->blas_, detail::librarySize(x.size()), x.deviceData(), 1, y.deviceData(), 1, &result));
        }
        return result;
    }
    template <Scalar T, CsrIndex I>
    T IterationContext<T, I>::norm(const Vector<T>& x) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::linalg::IterationContext<T, I>::norm", emberfield::NvtxColor::Cyan);
        impl_->validate(x);
        T result{};
        if (x.empty()) {
            return result;
        }
        detail::OperationScope scope(impl_->device_);
        if constexpr (std::same_as<T, float>) {
            check(cublasSnrm2(impl_->blas_, detail::librarySize(x.size()), x.deviceData(), 1, &result));
        }
        else {
            check(cublasDnrm2(impl_->blas_, detail::librarySize(x.size()), x.deviceData(), 1, &result));
        }
        return result;
    }
    template <Scalar T, CsrIndex I>
    void IterationContext<T, I>::jacobi(T weight, const Vector<T>& inverse, const Vector<T>& rhs,
                                        const Vector<T>& product, Vector<T>& x) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::linalg::IterationContext<T, I>::jacobi", emberfield::NvtxColor::Cyan);
        impl_->validate(inverse);
        impl_->validate(rhs);
        impl_->validate(product);
        impl_->validate(x);
        if (inverse.size() != x.size() || rhs.size() != x.size() || product.size() != x.size()) {
            throw std::invalid_argument("Jacobi shape mismatch");
        }
        if (x.empty()) {
            return;
        }
        detail::OperationScope scope(impl_->device_);
        const int n = detail::librarySize(x.size());
        update<<<(static_cast<unsigned>(n) + 255) / 256, 256>>>(n, weight, inverse.deviceData(), rhs.deviceData(),
                                                                product.deviceData(), x.deviceData());
        check(cudaGetLastError());
        check(cudaStreamSynchronize(nullptr));
    }
    template class IterationContext<float, std::int32_t>;
    template class IterationContext<double, std::int32_t>;
    template class IterationContext<float, std::int64_t>;
    template class IterationContext<double, std::int64_t>;
} // namespace emberfield::linalg
