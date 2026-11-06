/** @file Temporal.cuh @brief 固定离散计划、时刻参数快照、GPU演化上下文及事件游标。 */
#pragma once
#include <Simulation/Case/Assembly.cuh>
#include <Simulation/Case/System.cuh>
#include <memory>
#include <stdexcept>

/** @brief 时变物理参数的独立求值、组装及时间调度。 */
namespace emberfield::problem {
    /** @brief 固定关联与全部时间事件，不拥有可变GPU缓存。
     * @details 工厂共享PreparedThermalCase与规范CSR散加计划，事件按double精确排序去重。
     * 约束来源直接复用准备对象，不复制每个DOF的曲线。计划可供多个独立运行上下文只读共享；
     * 网格和定义不能经其他可变别名修改。K/M属于运行上下文，不属于本计划。
     * @code auto plan=buildThermalEvolutionPlan(prepared); @endcode */
    class ThermalEvolutionPlan {
    public:
        /** @brief 原组装计划，无参数无前提。@return 共享只读计划。@code auto p=plan.assemblyPlan(); @endcode */
        const std::shared_ptr<const ThermalAssemblyPlan>& assemblyPlan() const noexcept { return assemblyPlan_; }
        /** @brief 区间内去重事件，无参数无前提。@return 借用排序数组。@code auto e=plan.events(); @endcode */
        const std::vector<double>& events() const noexcept { return events_; }

    private:
        ThermalEvolutionPlan() = default; ///< 仅工厂构造合法计划。
        friend std::shared_ptr<const ThermalEvolutionPlan>
            buildThermalEvolutionPlan(std::shared_ptr<const PreparedThermalCase>, ThermalAssemblyOptions);
        std::shared_ptr<const ThermalAssemblyPlan> assemblyPlan_; ///< 全部固定离散关联。
        std::vector<double> events_; ///< 闭区间内时间表节点。
    };
    /** @brief 创建瞬态演化计划。@param prepared 已准备上下文。@param options 组装预算。
     * @pre 瞬态、同源。@return 完整共享计划；参数错误invalid_argument。
     * @code auto plan=buildThermalEvolutionPlan(p); @endcode */
    std::shared_ptr<const ThermalEvolutionPlan>
    buildThermalEvolutionPlan(std::shared_ptr<const PreparedThermalCase> prepared, ThermalAssemblyOptions options = {});
    /** @brief 单时刻不可变参数数组，定义行索引与来源plan固定。
     * @details 只有求值工厂能构造。sources按源定义行，primary/ambient按边界定义行，
     * constraints按唯一约束DOF行索引；快照拥有数组，共享计划，不含文件或CUDA资源。
     * 复制独立拥有参数数组，移动后的空来源不可用于运算。时刻始终使用绝对物理秒数。
     * @code auto state=evaluateThermalState(plan,1); @endcode */
    class EvaluatedThermalState {
    public:
        /** @brief 所属计划，无参数无前提。@return 共享计划。@code auto p=s.plan(); @endcode */
        const std::shared_ptr<const ThermalEvolutionPlan>& plan() const noexcept { return plan_; }
        /** @brief 绝对秒数，无参数无前提。@return 有限时刻。@code auto t=s.time(); @endcode */
        double time() const noexcept { return time_; }
        /** @brief 每源定义Q，无参数无前提。@return 借用数组。@code auto q=s.sources(); @endcode */
        const std::vector<double>& sources() const noexcept { return sources_; }
        /** @brief 每边界定义的q或h，无参数无前提。@return 借用数组，温度边界槽为0。@code auto h=s.primary(); @endcode
         */
        const std::vector<double>& primary() const noexcept { return primary_; }
        /** @brief 每边界定义T∞，无参数无前提。@return 借用数组，非对流槽为0。@code auto t=s.ambient(); @endcode */
        const std::vector<double>& ambient() const noexcept { return ambient_; }
        /** @brief 唯一DOF顺序的g，无参数无前提。@return 借用数组。@code auto g=s.constraints(); @endcode */
        const std::vector<double>& constraints() const noexcept { return constraints_; }

    private:
        EvaluatedThermalState() = default; ///< 仅完整求值工厂构造。
        friend EvaluatedThermalState evaluateThermalState(std::shared_ptr<const ThermalEvolutionPlan>, double);
        std::shared_ptr<const ThermalEvolutionPlan> plan_; ///< 来源所有权。
        double time_{}; ///< 快照时刻。
        std::vector<double> sources_, primary_, ambient_, constraints_; ///< 分类完整求值数组。
    };
    /** @brief 事务求值一个时刻。@param plan 固定计划。@param time 绝对秒数。
     * @pre 来源合法且时间在运行区间。@return 完整快照；参数/数值错误异常传播。
     * @code auto s=evaluateThermalState(plan,.1); @endcode */
    EvaluatedThermalState evaluateThermalState(std::shared_ptr<const ThermalEvolutionPlan> plan, double time);
    /** @brief 数值模型与对应物理快照的不可分割组合。
     * @details 共享不可变S/M/f与精确参数身份，生命周期可超过创建它的上下文。
     * 后续组装不会改变旧状态；保留多个旧状态可能额外保留设备数值资源。
     * matrixRevision只在同一上下文内标识S版本，不能跨实例比较；dt不属于该版本。
     * @code auto n=assembleThermalState(context,state); @endcode */
    class ThermalNumericState {
    public:
        /** @brief 同时刻模型，无参数无前提。@return 借用不可变模型。@code auto& m=n.model(); @endcode */
        const AssembledThermalModel& model() const {
            if (!model_) {
                throw std::logic_error("数值状态已移动");
            }
            return *model_;
        }
        /** @brief 对应快照，无参数无前提。@return 借用不可变参数。@code auto& s=n.parameters(); @endcode */
        const EvaluatedThermalState& parameters() const {
            if (!parameters_) {
                throw std::logic_error("数值状态已移动");
            }
            return *parameters_;
        }
        /** @brief 空间矩阵版本，无参数无前提。@return 同上下文内可比版本。@code auto r=n.matrixRevision(); @endcode */
        std::size_t matrixRevision() const noexcept { return matrixRevision_; }

    private:
        ThermalNumericState() = default; ///< 仅上下文发布完整状态。
        friend class ThermalEvolutionContext;
        std::shared_ptr<const AssembledThermalModel> model_; ///< 当前S/M/f不可变资源。
        std::shared_ptr<const EvaluatedThermalState> parameters_; ///< 精确对应参数。
        std::size_t matrixRevision_{}; ///< S版本，不包含dt。
    };
    /** @brief 每个运行实例独占的GPU缓存，组合共享固定计划。
     * @details 构造一次形成固定导热K和一致容量M。每次assemble精确比较分类参数：
     * Q改变只重建体源，q改变只重建通量载荷，环境温度改变只重建对流载荷，
     * h改变才重建H及S=K+H；g改变仅出现在新参数快照中。各类从完整当前贡献重建，
     * 不跨步差量累加。GPU候选全部成功后才替换缓存句柄，失败不改变已返回对象。
     * 同实例必须串行、同CUDA设备调用；可移动不可复制，移出后只允许赋值和析构。
     * 批积分与组装预算约束对应数组，不构成总进程显存硬上限。
     * @code ThermalEvolutionContext context(plan); @endcode */
    class ThermalEvolutionContext {
    public:
        /** @brief 创建固定资源。@param plan 来源计划。@pre 当前设备可用。
         * @note 无返回值，资源/参数异常传播。@code ThermalEvolutionContext c(plan); @endcode */
        explicit ThermalEvolutionContext(std::shared_ptr<const ThermalEvolutionPlan> plan);
        /** @brief 创建可观察固定资源。@param plan 来源计划。@param observer 借用批观察者。
         * @pre 同原构造，观察者存活覆盖上下文。@note 无返回值，异常同原构造。
         * @code ThermalEvolutionContext c(plan,&observer); @endcode */
        ThermalEvolutionContext(std::shared_ptr<const ThermalEvolutionPlan> plan, ProgressObserver* observer);
        ~ThermalEvolutionContext(); ///< RAII释放资源。
        ThermalEvolutionContext(ThermalEvolutionContext&&) noexcept; ///< 移动所有权。
        ThermalEvolutionContext& operator=(ThermalEvolutionContext&&) noexcept; ///< 移动所有权。
        ThermalEvolutionContext(const ThermalEvolutionContext&) = delete; ///< 不复制可变缓存。
        ThermalEvolutionContext& operator=(const ThermalEvolutionContext&) = delete; ///< 不复制可变缓存。
        /** @brief 事务更新数值。@param state 同源快照。@pre 串行同设备调用。
         * @return 完整模型与快照；旧对象不变，失败使缓存可重建。
         * @code auto n=context.assemble(state); @endcode */
        ThermalNumericState assemble(const EvaluatedThermalState& state);

    private:
        struct Impl; ///< 内部资源与分类依赖。
        std::unique_ptr<Impl> impl_; ///< 独占缓存。
    };
    /** @brief 独立数值更新入口。@param context 运行缓存。@param state 同源参数。
     * @pre 同assemble。@return 完整数值状态。@code auto n=assembleThermalState(c,s); @endcode */
    inline ThermalNumericState assembleThermalState(ThermalEvolutionContext& context,
                                                    const EvaluatedThermalState& state) {
        return context.assemble(state);
    }
    /** @brief 保留名义网格并插入事件的串行时间游标。
     * @details 名义编号与实际步号分开，避免事件插入后改变后续start+k*step网格。
     * 构造在maxSteps范围内预演并统计实际步数，不展开完整时间数组；超限在运行前拒绝。
     * peek不推进，只有求解、接受及结果发布成功后才accept，实际duration可小于名义step。
     * 所有参数采用步末右连续值，事件对齐不表示精确积分跳变载荷。
     * @code ThermalTimeCursor cursor(plan); while(auto s=cursor.peek()) cursor.accept(*s); @endcode */
    class ThermalTimeCursor {
    public:
        /** @brief 校验并预检步数。@param plan 来源。@pre 非空瞬态。
         * @note 无返回；超maxSteps抛length_error，停滞抛overflow_error。@code ThermalTimeCursor c(p); @endcode */
        explicit ThermalTimeCursor(std::shared_ptr<const ThermalEvolutionPlan> plan);
        /** @brief 查看下一步，无参数无前提。@return 步或终点空值，不推进。
         * @code auto s=cursor.peek(); @endcode */
        std::optional<TimeStep> peek() const;
        /** @brief 接受精确候选。@param step peek返回步。@pre 未发生其他推进。
         * @note 无返回；错配invalid_argument。@code cursor.accept(step); @endcode */
        void accept(const TimeStep& step);
        /** @brief 预计实际步数，无参数无前提。@return 预检步数。@code auto n=c.plannedSteps(); @endcode */
        std::size_t plannedSteps() const noexcept { return planned_; }

    private:
        std::shared_ptr<const ThermalEvolutionPlan> plan_; ///< 事件所有权。
        double current_{}; ///< 当前接受时刻。
        std::size_t nominal_{1}, event_{}, completed_{}, planned_{}; ///< 独立游标及预算计数。
    };
    /** @brief 用当前快照与历史约束构造系统。@param numeric 当前模型参数。@param previous 原历史温度。
     * @param previousState 历史参数。@param step 实际时间区间。
     * @pre 全部同源、同设备、时间精确匹配。@return 已对称消元A/b；错误异常传播。
     * @details 先以旧g校验previous，再构造A=S+M/dt、b=f+M*previous/dt，最后按当前g消元。
     * 不覆盖历史约束分量，因此保留一致质量矩阵的自由-约束耦合历史项。
     * @code auto a=buildThermalStepSystem(n,old,oldState,step); @endcode */
    systems::LinearSystem buildThermalStepSystem(const ThermalNumericState& numeric, const Vector<double>& previous,
                                                 const EvaluatedThermalState& previousState, const TimeStep& step);
} // namespace emberfield::problem
