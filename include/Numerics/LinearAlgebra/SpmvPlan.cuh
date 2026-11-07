/** @file SpmvPlan.cuh @brief 规范CSR快照的同步可复用稀疏执行资源。 */
#pragma once
#include <Numerics/LinearAlgebra/CanonicalCsrSnapshot.cuh>
/** @brief 线性代数内部可复用稀疏执行组件。 */
namespace emberfield::linalg::detail {
    /** @brief 独占描述符及工作区，拥有矩阵快照；同实例不可并发。
     * @tparam T Scalar精度。@tparam I CSR索引。
     * @code SpmvPlan<double,int> plan(freezeCanonical(matrix)); plan.multiply(x,y); @endcode */
    template <Scalar T, CsrIndex I>
    class SpmvPlan {
    public:
        /** @brief 建立一次库资源。@param snapshot 规范有限快照。@pre 设备可用、尺寸不超INT_MAX。
         * @return 无，资源异常传播。@code SpmvPlan<double,int> p(frozen); @endcode */
        explicit SpmvPlan(CanonicalCsrSnapshot<T, I> snapshot);
        /** @brief 在所属设备释放资源，无参数无前提。@return 无，不抛异常。
         * @code { SpmvPlan<double,int> p(frozen); } @endcode */
        ~SpmvPlan() noexcept;
        SpmvPlan(const SpmvPlan&) = delete; ///< 不复制可变执行状态。
        SpmvPlan& operator=(const SpmvPlan&) = delete; ///< 不复制可变执行状态。
        /** @brief 获取只读矩阵，无参数无前提。@return 借用矩阵。
         * @code auto& a = plan.matrix(); @endcode */
        const CsrMatrix<T, I>& matrix() const noexcept;
        /** @brief 同步计算y=A*x，热身后无新描述符或工作区分配。
         * @param x 输入。@param y 输出。@pre 同设备global、形状匹配且不别名，外部访问已完成。
         * @return 无；参数错误invalid_argument，CUDA/库错误传播，执行失败应丢弃计划与本次输出。
         * @code plan.multiply(x,y); @endcode */
        void multiply(const Vector<T>& x, Vector<T>& y);

    private:
        struct Impl; ///< 所属设备资源，完整定义不外泄CUDA库头。
        std::unique_ptr<Impl> impl_; ///< 独占可变资源。
    };
} // namespace emberfield::linalg::detail
