/** @file IterationObserver.cuh @brief 借用式迭代遥测，不额外执行数值运算。 */
#pragma once
#include <Solver/Iterative/SolveResult.cuh>
#include <optional>
/** @brief EmberField终端观察与进度接口，不持有数值资源。 */
namespace emberfield {
    /** @brief 遥测位置，Final不代表又完成一次迭代。 */
    enum class IterationKind { Initial, Iteration, Final };
    /** @brief 已有残差的计算来源。 */
    enum class ResidualKind { Recursive, True };
    /** @brief 单次回调标量快照。@code IterationInfo<double> info; @endcode */
    template <linalg::Scalar T>
    struct IterationInfo {
        int iteration_{}; ///< 已有求解器报告的迭代编号。
        T residualNorm_{}; ///< 当前已有残差范数。
        T tolerance_{}; ///< 本次apply的真实停止阈值。
        IterationKind kind_{IterationKind::Initial}; ///< 回调位置。
        ResidualKind residualKind_{ResidualKind::True}; ///< 残差来源。
        std::optional<SolveStatus> status_; ///< 仅Final携带终止状态。
    };
    /** @brief 同步借用的无异常观察者，不得重入或改动求解器。
     * @code struct Observer : IterationObserver<double> {}; @endcode */
    template <linalg::Scalar T>
    class IterationObserver {
    public:
        virtual ~IterationObserver() = default; ///< 多态释放，无参数无前提无返回。
        /** @brief 接收已有标量。@param info 临时快照。
         * @pre 不抛异常，不保留引用，不访问求解器。@note 无返回值。
         * @code observer.onIteration(info); @endcode */
        virtual void onIteration(const IterationInfo<T>& info) noexcept { (void)info; }
    };
} // namespace emberfield
