/** @file CanonicalCsrSnapshot.cuh @brief 不外泄可写别名的规范有限CSR快照。 */
#pragma once
#include <Numerics/LinearAlgebra/Operations.cuh>
#include <memory>
/** @brief 线性代数内部不可变矩阵导入与拥有语义。 */
namespace emberfield::linalg::detail {
    template <Scalar T, CsrIndex I>
    class CanonicalCsrSnapshot;
    /** @brief 导入独立规范快照，只有成功规范化及有限校验后才发布。
     * @tparam T Scalar精度。@tparam I CSR索引。@param matrix 合法通用CSR。
     * @pre 调用期间输入无并发写。@return 独立只读快照，原输入随后可销毁或修改。
     * @code auto frozen = freezeCanonical(matrix); @endcode */
    template <Scalar T, CsrIndex I>
    CanonicalCsrSnapshot<T, I> freezeCanonical(const CsrMatrix<T, I>& matrix);
    /** @brief 多消费者共享只读拥有的CSR，复制不复制数组。
     * @tparam T Scalar精度。@tparam I CSR索引。
     * @code auto frozen = freezeCanonical(matrix); auto shared = frozen; @endcode */
    template <Scalar T, CsrIndex I>
    class CanonicalCsrSnapshot {
    public:
        /** @brief 读取拥有的矩阵，无参数无额外约束。@return 只读借用，生命周期由快照覆盖。
         * @code auto& matrix = frozen.matrix(); @endcode */
        const CsrMatrix<T, I>& matrix() const noexcept { return *matrix_; }

    private:
        friend CanonicalCsrSnapshot<T, I> freezeCanonical<T, I>(const CsrMatrix<T, I>&);
        /** @brief 接管已检查候选。@param matrix 独占候选。@pre 无外泄写别名且规范有限。
         * @return 无。@code CanonicalCsrSnapshot next(std::move(matrix)); @endcode */
        explicit CanonicalCsrSnapshot(CsrMatrix<T, I> matrix) :
            matrix_(std::make_shared<const CsrMatrix<T, I>>(std::move(matrix))) {}
        std::shared_ptr<const CsrMatrix<T, I>> matrix_; ///< 只读共同拥有者；无空默认状态。
    };
    template <Scalar T, CsrIndex I>
    CanonicalCsrSnapshot<T, I> freezeCanonical(const CsrMatrix<T, I>& matrix) {
        OperationScope scope(matrix.rowOffsets().deviceId());
        auto candidate = canonical<I>(matrix);
        if (!finiteValues(candidate.nonZeroCount(), candidate.deviceValues())) {
            throw std::invalid_argument("Matrix contains nonfinite entries");
        }
        return CanonicalCsrSnapshot<T, I>(std::move(candidate));
    }
} // namespace emberfield::linalg::detail
