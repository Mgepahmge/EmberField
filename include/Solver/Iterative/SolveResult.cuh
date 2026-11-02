/** @file SolveResult.cuh
 * @brief 定义SolveResult的独立接口与契约。
 */
#pragma once
#include <Numerics/LinearAlgebra/IterationContext.cuh>
namespace emberfield {
    /** @brief 标记求解终止原因。 */
    enum class SolveStatus { Converged, MaxIterations, Breakdown, NonFinite };
    /** @brief 保存真实残差和终止原因。
     * @code auto result = solver.apply(rhs, x); @endcode
     */
    template <linalg::Scalar T>
    struct SolveResult {
        SolveStatus status_{SolveStatus::MaxIterations}; ///< 终止状态。
        int iterations_{}; ///< 已执行迭代次数。
        T residualNorm_{}; ///< 最终真实残差二范数。
        T relativeResidual_{}; ///< 相对右端项的残差；零右端项时等于绝对残差。
    };
} // namespace emberfield
