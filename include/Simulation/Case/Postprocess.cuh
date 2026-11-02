/** @file Postprocess.cuh
 * @brief 热后处理计划、物理身份映射、监测及未消元热平衡。
 */
#pragma once
#include <DataStructure/Field/Snapshot.cuh>
#include <Simulation/Case/Assembly.cuh>
namespace emberfield::problem {
    /** @brief 单个边界功率。@code BoundaryPower p; @endcode */
    struct BoundaryPower {
        std::string id_; ///< 自然条件id；整体反力为空。
        std::string kind_; ///< flux/convection/temperature。
        double powerOutW_{}; ///< 正为向外。
    };
    /** @brief 热量守恒报告，缺失值不伪装成零。@code HeatBalanceReport r; @endcode */
    struct HeatBalanceReport {
        std::optional<double> storedEnergyJ_, storagePowerW_, sourcePowerW_, prescribedFluxOutW_, convectionOutW_,
            dirichletOutW_, errorW_, toleranceW_; ///< SI能量和功率。
        std::string status_{"disabled"}; ///< disabled/unavailable/passed/warning。
        std::vector<BoundaryPower> boundaries_; ///< 声明顺序的自然功率及唯一反力集合。
    };
    /** @brief 监测长表值。@code MonitorValue v; @endcode */
    struct MonitorValue {
        std::string id_, kind_, quantity_, unit_; ///< 身份、统计量和单位。
        double value_{}; ///< 有限值。
    };
    /** @brief 一个接受态的完整后处理结果。@code auto r=plan.evaluate(m,t,nullptr,{},0,{}); @endcode */
    struct ThermalFields {
        std::optional<field::FieldSnapshot> snapshot_; ///< 按需拥有的独立物理场，未保存帧时为空。
        HeatBalanceReport energy_; ///< 能量诊断。
        std::vector<MonitorValue> monitors_; ///< 顺序监测。
        double minimumK_{}, maximumK_{}; ///< 全活动极值。
    };
    /** @brief 共享准备对象并复用静态映射和定位结果；不持有求解器或文件流。
     * @details 构造在求解前验证全部监测；P1体积分与Fourier运算采用有界GPU批次。
     * 独占实现禁止复制，移动后只允许赋值与析构；同实例不可并发evaluate。
     * @code ThermalPostprocessPlan plan(prepared); auto g=plan.grid(); @endcode */
    class ThermalPostprocessPlan {
    public:
        /** @brief 建立静态身份与监测定位。@param prepared 非空同源上下文。
         * @pre 当前CUDA设备有效。@note 无返回值。非法监测invalid_argument，超限length_error。
         * @code ThermalPostprocessPlan p(prepared); @endcode */
        explicit ThermalPostprocessPlan(std::shared_ptr<const PreparedThermalCase> prepared);
        ~ThermalPostprocessPlan() noexcept; ///< 自动释放主机状态，不写文件。
        ThermalPostprocessPlan(ThermalPostprocessPlan&&) noexcept; ///< 移动独占状态。
        ThermalPostprocessPlan& operator=(ThermalPostprocessPlan&&) noexcept; ///< 移动赋值。
        ThermalPostprocessPlan(const ThermalPostprocessPlan&) = delete; ///< 禁止复制。
        ThermalPostprocessPlan& operator=(const ThermalPostprocessPlan&) = delete; ///< 禁止复制。
        /** @brief 查询不可变网格，无参数无额外前提。@return 共享所有权。@code auto g=p.grid(); @endcode */
        const std::shared_ptr<const field::VisualizationGrid>& grid() const noexcept;
        /** @brief 输出已解析监测字典JSON，无参数无前提。@return JSON数组文本。@code auto s=p.monitorDictionary();
         * @endcode */
        std::string monitorDictionary() const;
        /** @brief 独立重建配置所选场，不计算能量或监测。@param temperature 全DOF温度。
         * @param step 步。@param time 瞬态时间。@param initial 初态。
         * @pre 同源global有限非负状态。@return 拥有场数组的快照，错误不发布部分结果。
         * @code auto f=p.evaluateThermalFields(t,0,{},false); @endcode */
        field::FieldSnapshot evaluateThermalFields(const Vector<double>& temperature, std::uint64_t step,
                                                   std::optional<double> time, bool initial = false);
        /** @brief 独立评估静态监测计划。@param temperature 同源温度。
         * @pre global有限非负。@return 全部监测值；设备或数值错误传播。
         * @code auto v=p.evaluateMonitors(t); @endcode */
        std::vector<MonitorValue> evaluateMonitors(const Vector<double>& temperature);
        /** @brief 独立评估未消元离散热量平衡。@param model 同源模型。
         * @param temperature 当前温度。@param previous 紧邻前态或空。@param dt 实际步长或空。
         * @param initial 是否给定初态。@pre 瞬态接受态必须有前态和正dt。
         * @return 完整热量报告，超容差为warning；参数错误invalid_argument。
         * @code auto e=p.evaluateHeatBalance(m,t,&old,.1); @endcode */
        HeatBalanceReport evaluateHeatBalance(const AssembledThermalModel& model, const Vector<double>& temperature,
                                              const Vector<double>* previous, std::optional<double> dt,
                                              bool initial = false);
        /** @brief 评估完整一帧。@param model 未消元同源模型。@param temperature 接受的global温度。
         * @param previous 瞬态前态，初态/稳态为空。@param dt 实际步长，初态/稳态为空。
         * @param step 步号。@param time 瞬态物理时刻，稳态为空。@param initial 是否初态。
         * @param includeFrame 是否生成完整主机场快照；false时仅回读批次统计与监测。
         * @pre 温度有限非负且满足规定值，当前设备一致。@return 全部后处理成功的完整状态。
         * @throws std::invalid_argument 不同源或非法状态；数值、资源、CUDA异常传播。
         * @code auto r=p.evaluate(m,t,&previous,dt,1,time); @endcode */
        ThermalFields evaluate(const AssembledThermalModel& model, const Vector<double>& temperature,
                               const Vector<double>* previous, std::optional<double> dt, std::uint64_t step,
                               std::optional<double> time, bool initial = false, bool includeFrame = true);

    private:
        struct Impl; ///< 模块内部几何、定位和预算状态。
        std::unique_ptr<Impl> impl_; ///< 独占实现。
    };
} // namespace emberfield::problem
