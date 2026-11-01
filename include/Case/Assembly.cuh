/** @file Assembly.cuh
 * @brief 将独立符号计划和GPU局部运算组合为可复用的全局热模型。
 */
#pragma once
#include <Assembly/Operations.cuh>
#include <Case/GpuPreparation.cuh>
/** @brief 物理算例的组装适配，不承担通用稀疏运算。 */
namespace thermofem::problem {
    /** @brief 有界流式组装选项。@code ThermalAssemblyOptions o; o.batchElements_=256; @endcode */
    struct ThermalAssemblyOptions {
        assembly::AssemblyLimits limits_; ///< 全局符号预算。
        thermal::GpuBatchLimits gpuLimits_; ///< 单个GPU积分批预算。
        std::size_t batchElements_{4096}; ///< 批内实体上限，受GPU数组预算进一步限制。
    };
    namespace detail {
        class ThermalAssemblyAccess;
    }
    /** @brief 共享准备对象并保存一次建立、多次使用的符号计划。
     * @details 模板顺序为全部活动体行、全部自然关联行。自然条件重复定义各自保留，
     * 不先合并h或环境温度；侧模板严格使用迹自由度顺序。
     * @code auto p=buildThermalAssemblyPlan(prepared); @endcode */
    class ThermalAssemblyPlan {
    public:
        /** @brief 准备上下文，无参数无额外前提。@return 共享只读对象。@code auto p=a.prepared(); @endcode */
        const std::shared_ptr<const PreparedThermalCase>& prepared() const noexcept { return prepared_; }
        /** @brief 符号计划，无参数无额外前提。@return 共享只读对象。@code auto p=a.plan(); @endcode */
        const std::shared_ptr<const assembly::AssemblyPlan>& plan() const noexcept { return plan_; }
        /** @brief 批与预算选项，无参数无额外前提。@return 只读引用。@code auto o=a.options(); @endcode */
        const ThermalAssemblyOptions& options() const noexcept { return options_; }

    private:
        friend class detail::ThermalAssemblyAccess;
        ThermalAssemblyPlan() = default; ///< 仅工厂构建。
        std::shared_ptr<const PreparedThermalCase> prepared_; ///< 空间、几何和条件的共同所有权。
        std::shared_ptr<const assembly::AssemblyPlan> plan_; ///< 全域符号映射。
        ThermalAssemblyOptions options_; ///< 实际执行选项。
    };
    /** @brief 组合拥有未消元S=K+H、可选一致容量M以及完整外载荷f。
     * @details 全部数值保留于当前设备global存储，S与M共享相同CSR模式。
     * 模型不保存求解器初猜或上一时刻温度，可用于多个时间步长。
     * @code auto m=assembleThermalModel(plan); @endcode */
    class AssembledThermalModel {
    public:
        /** @brief 来源计划，无参数无额外前提。@return 共享只读对象。@code auto p=m.plan(); @endcode */
        const std::shared_ptr<const ThermalAssemblyPlan>& plan() const noexcept { return plan_; }
        /** @brief 空间矩阵，无参数无额外前提。@return S的只读引用。@code auto& s=m.spatial(); @endcode */
        const assembly::Matrix& spatial() const noexcept { return spatial_; }
        /** @brief 一致容量，无参数无额外前提。@return 瞬态有值。@code auto& c=m.capacity(); @endcode */
        const std::optional<assembly::Matrix>& capacity() const noexcept { return capacity_; }
        /** @brief 完整载荷，无参数无额外前提。@return f的只读引用。@code auto& f=m.load(); @endcode */
        const Vector<double>& load() const noexcept { return load_; }

    private:
        friend class detail::ThermalAssemblyAccess;
        AssembledThermalModel() = default; ///< 工厂候选。
        std::shared_ptr<const ThermalAssemblyPlan> plan_; ///< 来源共同所有权。
        assembly::Matrix spatial_; ///< 未消元导热与对流之和。
        std::optional<assembly::Matrix> capacity_; ///< 瞬态一致容量。
        Vector<double> load_; ///< 体源、外向通量和环境载荷。
    };
    /** @brief 构建热模型符号映射。@param prepared 已成功准备的上下文。@param options 正预算及批大小。
     * @pre 网格和绑定对象不经可变别名修改。@return 共享只读计划。
     * @throws std::invalid_argument 空上下文或非法选项；计数超限length_error，资源异常传播。
     * @code auto plan=buildThermalAssemblyPlan(prepared); @endcode */
    [[nodiscard]] std::shared_ptr<const ThermalAssemblyPlan>
    buildThermalAssemblyPlan(std::shared_ptr<const PreparedThermalCase> prepared, ThermalAssemblyOptions options = {});
    /** @brief 流式GPU积分并组装完整热模型。@param plan 成功的符号计划。
     * @pre 当前CUDA设备有效且来源不变。@return 独立拥有S/M/f的模型，只在全部成功后发布。
     * @throws std::invalid_argument 空计划；积分、数值和CUDA异常传播。
     * @code auto model=assembleThermalModel(plan); @endcode */
    [[nodiscard]] AssembledThermalModel assembleThermalModel(std::shared_ptr<const ThermalAssemblyPlan> plan);
} // namespace thermofem::problem
