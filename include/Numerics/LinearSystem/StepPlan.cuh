/** @file StepPlan.cuh @brief 固定空间矩阵、容量、实际步长及约束集合的后向Euler执行计划。 */
#pragma once
#include <Numerics/LinearSystem/Operations.cuh>
#include <memory>
/** @brief 固定时间离散资源的内部执行组件。 */
namespace emberfield::systems::detail {
    /** @brief 拥有独立A0/A与M/dt乘法资源；不借用可变来源矩阵。
     * @details 同实例串行；dt精确固定，值g和载荷可变化，旧系统独立拥有。
     * @code BackwardEulerPlan plan(s,m,dt,dofs); auto system=plan.build(f,old,g); @endcode */
    class BackwardEulerPlan {
    public:
        /** @brief 创建固定系统资源。@param spatial S。@param capacity M。@param dt 实际步长。
         * @param dofs 有序唯一约束集合。@pre 同设备global、规范含对角、同模式、dt正有限。
         * @return 无，保持原构建错误类别。@code BackwardEulerPlan p(s,m,dt,dofs); @endcode */
        BackwardEulerPlan(const Matrix& spatial, const Matrix& capacity, double dt,
                          std::span<const std::uint64_t> dofs);
        /** @brief RAII释放，无参数无前提。@return 无。
         * @code { BackwardEulerPlan p(s,m,dt,dofs); } @endcode */
        ~BackwardEulerPlan() noexcept;
        BackwardEulerPlan(const BackwardEulerPlan&) = delete; ///< 独占执行状态。
        BackwardEulerPlan& operator=(const BackwardEulerPlan&) = delete; ///< 不复制。
        /** @brief 从未消元A0重建本次右端，返回独立拥有的系统。
         * @param load 当前外载。@param previous 已验证历史温度。@param prescribed 当前g。
         * @pre 同形同设备，g长度等于固定集合；历史g由调用层检查。
         * @return 完整独立系统；参数或数值失败不发布。
         * @code auto system=plan.build(f,old,g); @endcode */
        LinearSystem build(const Vector<double>& load, const Vector<double>& previous,
                           std::span<const double> prescribed);

    private:
        struct Impl; ///< 固定矩阵、掩码、规定值暂存、M/dt执行资源。
        std::unique_ptr<Impl> impl_; ///< 独占可变设备描述符。
    };
} // namespace emberfield::systems::detail
