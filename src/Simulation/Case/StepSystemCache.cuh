/** @file StepSystemCache.cuh @brief 单个热运行实例的最近后向Euler计划缓存。 */
#pragma once
#include <Numerics/LinearSystem/StepPlan.cuh>
#include <Simulation/Case/Temporal.cuh>
/** @brief 热运行内部时间系统缓存。 */
namespace emberfield::problem::detail {
    /** @brief 同源不可变S/M、精确dt与固定约束集决定缓存身份；旧读者不受覆盖。
     * @code ThermalStepSystemCache cache; auto system=cache.build(n,old,state,step); @endcode */
    class ThermalStepSystemCache {
    public:
        /** @brief 验证历史和当前状态并重建右端。@param numeric 当前模型。@param previous 历史温度。
         * @param old 历史参数。@param step 实际时间段。@pre 同运行实例串行，来源固定。
         * @return 独立系统，失败不更新缓存。@code auto s=cache.build(n,t,old,step); @endcode */
        systems::LinearSystem build(const ThermalNumericState& numeric, const Vector<double>& previous,
                                    const EvaluatedThermalState& old, const TimeStep& step);

    private:
        std::optional<ThermalNumericState> owner_; ///< 保留最近S/M强所有权及精确来源。
        double dt_{}; ///< 最近实际步长，精确比较。
        std::unique_ptr<systems::detail::BackwardEulerPlan> plan_; ///< 单个最近系统执行资源。
    };
} // namespace emberfield::problem::detail
