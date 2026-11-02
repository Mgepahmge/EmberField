/** @file System.cuh
 * @brief 热模型分量诊断、稳态/后向Euler系统工厂及实际时间区间。
 */
#pragma once
#include <Numerics/LinearSystem/Operations.cuh>
#include <Simulation/Case/Assembly.cuh>
#include <limits>
/** @brief 组合热模型与独立线性系统操作。 */
namespace emberfield::problem {
    /** @brief 单个编号共现分量的热锚定与载荷兼容性。
     * @details 兼容性仅描述常数零空间的载荷平衡，不替代任意矩阵的正定性认证。
     * @code if(!c.anchored()) inspect(c.netLoad_); @endcode */
    struct ComponentCheck {
        std::size_t component_{}; ///< 分量编号。
        std::size_t dofCount_{}; ///< 分量自由度数。
        bool hasDirichlet_{}; ///< 存在规定温度。
        bool hasPositiveConvection_{}; ///< 存在正换热系数。
        bool hasCapacity_{}; ///< 瞬态正体热容。
        bool loadCompatible_{}; ///< 净载荷在指定平衡容差内。
        long double netLoad_{}; ///< 分量载荷代数和W。
        long double absoluteLoad_{}; ///< 分量载荷绝对值和W。
        /** @brief 查询常数模式是否被锚定，无参数无额外前提。
         * @return 三种锚定任一存在。@code bool ok=c.anchored(); @endcode */
        [[nodiscard]] bool anchored() const noexcept { return hasDirichlet_ || hasPositiveConvection_ || hasCapacity_; }
    };
    /** @brief 分量诊断集合。@code auto r=checkThermalSolvability(model); @endcode */
    struct SolvabilityReport {
        std::vector<ComponentCheck> components_; ///< 每个分量的实际诊断。
        /** @brief 查询全部分量是否锚定，无参数无额外前提。
         * @return 非空且每分量被锚定。@code if(r.canBuildSystem()) run(); @endcode */
        [[nodiscard]] bool canBuildSystem() const noexcept;
    };
    /** @brief 实际推进区间，最后一步可能较短。@code auto s=nextTimeStep(t,0,0); @endcode */
    struct TimeStep {
        std::size_t index_{}; ///< 从1开始的步编号。
        double start_{}; ///< 区间起点秒数。
        double end_{}; ///< 区间终点秒数。
        double duration_{}; ///< 实际正步长秒数。
    };
    /** @brief 检查各分量热锚定及纯Neumann平衡。@param model 完整组装模型。
     * @param relativeTolerance 非负有限相对平衡容差。@param absoluteTolerance 非负有限绝对功率容差。
     * @pre 当前设备与模型一致。@return 按分量列出锚定与净功率，不修改模型。
     * @throws std::invalid_argument 容差错误；资源和CUDA异常传播。
     * @code auto r=checkThermalSolvability(m); @endcode */
    [[nodiscard]] SolvabilityReport checkThermalSolvability(const AssembledThermalModel& model,
                                                            double relativeTolerance = 128 *
                                                                std::numeric_limits<double>::epsilon(),
                                                            double absoluteTolerance = 0);
    /** @brief 生成已对称消元的稳态A,b。@param model 稳态模型。
     * @pre 当前设备匹配。@return 可直接交给求解器的全维系统。
     * @throws std::invalid_argument 分析类型错误；domain_error 未锚定；数值和资源异常传播。
     * @code auto s=buildThermalSteadySystem(m); @endcode */
    [[nodiscard]] systems::LinearSystem buildThermalSteadySystem(const AssembledThermalModel& model);
    /** @brief 生成已消元的后向Euler系统。@param model 瞬态模型。
     * @param previous 全部DOF上一时刻绝对温度。@param actualTimeStep 当前实际步长。
     * @pre previous有限非负并满足常值温度约束，global同设备；不等同求解器初猜。
     * @return A=S+M/dt以及完整历史项修正后的b。
     * @throws std::invalid_argument 类型、历史场或步长错误；数值/资源异常传播。
     * @code auto s=buildThermalStepSystem(m,previous,step.duration_); @endcode */
    [[nodiscard]] systems::LinearSystem buildThermalStepSystem(const AssembledThermalModel& model,
                                                               const Vector<double>& previous, double actualTimeStep);
    /** @brief 计算下一实际区间，不预分配全时间序列。@param options 合法时间参数。
     * @param current 当前时间。@param completedSteps 已完成步数。
     * @pre current位于闭区间，调用者串行维护状态。@return 到终点nullopt，否则下一正区间。
     * @throws std::invalid_argument 非法参数；length_error 超步数；overflow_error 浮点无法推进。
     * @code auto s=nextTimeStep(t,current,completed); @endcode */
    [[nodiscard]] std::optional<TimeStep> nextTimeStep(const TimeOptions& options, double current,
                                                       std::size_t completedSteps);
} // namespace emberfield::problem
