/** @file Definition.cuh
 * @brief 定义与网格、JSON及CUDA无关的热传导物理算例值对象。
 */
#pragma once
#include <Simulation/Case/Diagnostics.cuh>
#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

/** @brief 提供类型化物理模型；数值统一使用SI与绝对温度K。 */
namespace emberfield::problem {
    /** @brief 常量标量场；未来变体通过ScalarField扩展，不执行脚本。
     * @code ScalarField field=ConstantScalar{300}; @endcode */
    struct ConstantScalar {
        double value_{}; ///< 有限数值，单位由使用位置确定。
    };
    using ScalarField = std::variant<ConstantScalar>; ///< 封闭的类型化标量场。
    /** @brief 计算常量场。@param field 场。@pre 已通过有限性与物理范围校验。
     * @return 标量值，无分配。@code auto t=evaluate(ScalarField{ConstantScalar{300}}); @endcode */
    [[nodiscard]] double evaluate(const ScalarField& field) noexcept;
    /** @brief 选择器运算。 */
    enum class RegionKind { PhysicalGroup, Union, Intersection, Difference };
    /** @brief 引用物理组或递归集合，差集operands依次为base与remove。
     * @code RegionSelector r; r.dimension_=2; r.tag_=11; @endcode */
    struct RegionSelector {
        RegionKind kind_{RegionKind::PhysicalGroup}; ///< 运算类型。
        int dimension_{-1}; ///< 物理组维数；集合节点使用-1，由子树确定。
        std::optional<std::int64_t> tag_; ///< 物理标签；与name互斥。
        std::optional<std::string> name_; ///< 物理组名称。
        std::vector<RegionSelector> operands_; ///< 并交非空，差集恰两个，叶节点为空。
    };
    /** @brief 网格文件来源，绑定器不读取此路径。
     * @code MeshSource source; source.path_="mesh.msh"; @endcode */
    struct MeshSource {
        std::filesystem::path path_; ///< 文件路径，读取器规范化为绝对路径。
        std::string format_{"msh"}; ///< 当前只支持msh。
        double coordinateScale_{1}; ///< 网格坐标到米的正缩放。
    };
    /** @brief 稳态与瞬态分析。 */
    enum class AnalysisType { Steady, Transient };
    /** @brief 第一版支持的物理空间模型。 */
    enum class SpaceModel { Planar2d, Solid3d };
    /** @brief 物理分析定义，不包含离散空间和求解器。
     * @code Analysis a; a.thickness_=0.01; @endcode */
    struct Analysis {
        AnalysisType type_{AnalysisType::Steady}; ///< 时间类型。
        SpaceModel space_{SpaceModel::Planar2d}; ///< 空间模型。
        RegionSelector domain_; ///< 非空活动域。
        std::optional<double> thickness_; ///< 二维厚度，米；三维禁止。
        double geometryTolerance_{1e-12}; ///< XY平面检查容差，米。
    };
    /** @brief 可复用常系数材料；热容量的两种表示互斥。
     * @code Material m; m.id_="steel"; m.conductivity_=ConstantScalar{15}; @endcode */
    struct Material {
        std::string id_; ///< 材料唯一id。
        std::string displayName_; ///< 可选显示名。
        ScalarField conductivity_{ConstantScalar{0}}; ///< 导热率W/(m K)，严格正。
        std::optional<ScalarField> density_; ///< 密度kg/m³。
        std::optional<ScalarField> specificHeat_; ///< 比热J/(kg K)。
        std::optional<ScalarField> volumetricHeatCapacity_; ///< 体热容J/(m³ K)。
    };
    /** @brief 独立的材料分配，禁止重叠与缺失。
     * @code MaterialAssignment a; a.id_="body"; a.material_="steel"; @endcode */
    struct MaterialAssignment {
        std::string id_; ///< 分配唯一id。
        RegionSelector region_; ///< 活动域内区域。
        std::string material_; ///< 材料id。
    };
    /** @brief 体积热源，多条源在交叠区域相加。
     * @code VolumeSource s; s.value_=ConstantScalar{100}; @endcode */
    struct VolumeSource {
        std::string id_; ///< 源唯一id。
        RegionSelector region_; ///< 作用域。
        ScalarField value_{ConstantScalar{0}}; ///< 发热率W/m³，可正可负。
    };
    /** @brief 已知温度边界参数。
     * @code TemperatureBoundary b{ConstantScalar{300}}; @endcode */
    struct TemperatureBoundary {
        ScalarField temperature_{ConstantScalar{0}}; ///< 绝对温度K，不小于0。
    };
    /** @brief 外向热流参数，正号表示热量离开活动域。
     * @code FluxBoundary b{ConstantScalar{-100}}; @endcode */
    struct FluxBoundary {
        ScalarField flux_{ConstantScalar{0}}; ///< W/m²，有符号。
    };
    /** @brief 对流换热参数，不表示求解流场。
     * @code ConvectionBoundary b{ConstantScalar{20},ConstantScalar{293.15}}; @endcode */
    struct ConvectionBoundary {
        ScalarField coefficient_{ConstantScalar{0}}; ///< 换热系数W/(m² K)，非负。
        ScalarField ambientTemperature_{ConstantScalar{0}}; ///< 环境温度K，非负。
    };
    using BoundaryValue = std::variant<TemperatureBoundary, FluxBoundary, ConvectionBoundary>; ///< 条件参数联合。
    /** @brief 单个边界条件；作用位置与数值参数分开。
     * @code BoundaryCondition b; b.value_=FluxBoundary{ConstantScalar{1}}; @endcode */
    struct BoundaryCondition {
        std::string id_; ///< 条件唯一id。
        RegionSelector region_; ///< 活动域余维一外侧选择。
        BoundaryValue value_; ///< 类型化条件参数。
    };
    /** @brief 自然边界在同侧叠加的显式策略。 */
    enum class NaturalOverlap { Error, Add };
    /** @brief 初始温度的区域覆盖，不影响求解器初猜。
     * @code InitialOverride i; i.temperature_=ConstantScalar{300}; @endcode */
    struct InitialOverride {
        std::string id_; ///< 覆盖唯一id。
        RegionSelector region_; ///< 不重叠的活动域选择。
        ScalarField temperature_{ConstantScalar{0}}; ///< 初始绝对温度K。
    };
    /** @brief 全域初温与互斥覆盖。
     * @code InitialState i; i.defaultTemperature_=ConstantScalar{300}; @endcode */
    struct InitialState {
        ScalarField defaultTemperature_{ConstantScalar{0}}; ///< 必需全域默认初温。
        std::vector<InitialOverride> overrides_; ///< 可选不重叠覆盖。
    };
    /** @brief 后向Euler时间区间；不实际执行推进。
     * @code TimeOptions t{0,10,0.1,1000000}; @endcode */
    struct TimeOptions {
        double start_{}; ///< 起始秒数。
        double end_{}; ///< 结束秒数，严格大于start。
        double step_{}; ///< 正步长，最后一步允许截短。
        std::size_t maxSteps_{1000000}; ///< 最大步数。
    };
    /** @brief 连续Lagrange空间与自动积分选项。
     * @code Discretization d; d.order_=1; @endcode */
    struct Discretization {
        unsigned order_{1}; ///< 温度空间阶次，独立于几何阶次。
    };
    /** @brief 已有线性求解器的方法标识。 */
    enum class LinearMethod { BiCgStab, Cg };
    /** @brief 数值初猜策略。 */
    enum class InitialGuess { Zero, PreviousStep };
    /** @brief 第一版使用恒等预条件子，AMG完整协议尚未适配。
     * @code LinearOptions options; options.method_=LinearMethod::Cg; @endcode */
    struct LinearOptions {
        LinearMethod method_{LinearMethod::BiCgStab}; ///< 方法。
        double relativeTolerance_{1e-8}; ///< 非负相对容差。
        double absoluteTolerance_{1e-12}; ///< 非负绝对容差，至少一项为正。
        std::size_t maxIterations_{1000}; ///< 正迭代上限。
        InitialGuess initialGuess_{InitialGuess::Zero}; ///< 与物理初温分离。
    };
    /** @brief 类型化编码选项，与IO编码器类型分离。 */
    enum class OutputEncoding { AppendedRaw, Ascii };
    /** @brief 可选物理场。 */
    enum class OutputField { Temperature, TemperatureGradient, HeatFlux };
    /** @brief 区域统计量。 */
    enum class RegionStatistic { TemperatureMin, TemperatureMax, TemperatureMean, StoredEnergy };
    /** @brief 场的协议名字。@param value 枚举。@pre 无。@return 稳定名字；非法invalid_argument。
     * @code auto name=outputFieldName(OutputField::Temperature); @endcode */
    std::string_view outputFieldName(OutputField value);
    /** @brief 统计量的协议名字。@param value 枚举。@pre 无。@return 稳定名字；非法invalid_argument。
     * @code auto name=regionStatisticName(RegionStatistic::TemperatureMean); @endcode */
    std::string_view regionStatisticName(RegionStatistic value);
    /** @brief 输出资源上限；仅约束输出模块增量资源。@code OutputLimits limits; @endcode */
    struct OutputLimits {
        std::size_t maxPoints_{4000000}; ///< 活动顶点。
        std::size_t maxCells_{1000000}; ///< 活动单元。
        std::size_t maxHostBytes_{1073741824}; ///< 主机缓存字节。
        std::size_t maxDeviceBytes_{536870912}; ///< 设备缓存字节。
        std::size_t batchCells_{4096}; ///< 单批单元。
        std::size_t maxFrames_{1000000}; ///< 帧数。
        std::size_t maxMonitors_{1000}; ///< 监测数。
        std::uint64_t maxOutputBytes_{107374182400ULL}; ///< 文件总字节。
    };
    /** @brief 热量诊断容差，不改变求解收敛判断。@code EnergyOptions e; @endcode */
    struct EnergyOptions {
        bool enabled_{true}; ///< 启用热量报告。
        double absoluteToleranceW_{1e-8}; ///< 绝对功率容差。
        double relativeTolerance_{1e-8}; ///< 相对容差。
    };
    /** @brief 原始节点标签定位。@code NodeMonitor m{1}; @endcode */
    struct NodeMonitor {
        std::uint64_t nodeTag_{}; ///< 正节点标签。
    };
    /** @brief 米制位置定位。@code PointMonitor m; @endcode */
    struct PointMonitor {
        std::array<double, 3> positionM_{}; ///< 米制位置。
        double toleranceM_{1e-10}; ///< 非负定位容差。
    };
    /** @brief 区域积分和极值。@code RegionMonitor m; @endcode */
    struct RegionMonitor {
        RegionSelector region_; ///< 活动同维区域。
        std::vector<RegionStatistic> statistics_{RegionStatistic::TemperatureMean}; ///< 有序统计选择。
    };
    /** @brief 监测身份和类型化定位。@code MonitorDefinition m{"sensor",NodeMonitor{1}}; @endcode */
    struct MonitorDefinition {
        std::string id_; ///< 唯一显示名称。
        std::variant<NodeMonitor, PointMonitor, RegionMonitor> value_; ///< 定位变体。
    };
    /** @brief 结果输出意图；不负责创建文件或生成场。
     * @code OutputOptions o; o.directory_="results"; @endcode */
    struct OutputOptions {
        std::filesystem::path directory_{"results"}; ///< 输出目录。
        bool overwrite_{}; ///< 显式允许覆盖。
        std::optional<std::size_t> everySteps_; ///< 瞬态默认1；稳态禁止。
        std::optional<bool> includeInitial_; ///< 瞬态默认true；稳态禁止。
        bool includeFinal_{true}; ///< 最终状态输出，稳态必须为true。
        OutputEncoding encoding_{OutputEncoding::AppendedRaw}; ///< 无压缩二进制或ASCII。
        std::vector<OutputField> fields_{OutputField::Temperature}; ///< 温度及可选梯度、热流。
        EnergyOptions energy_; ///< 能量诊断。
        std::vector<MonitorDefinition> monitors_; ///< 有序监测。
        OutputLimits limits_; ///< 分配和输出上限。
    };
    /** @brief 完整算例值对象，通过组合解耦物理、离散与运行配置。
     * @details 使用前调用validateCase；文件解析和网格绑定不在本类内。
     * 标准容器负责RAII，复制独立拥有，移动后需重新校验才可再次绑定。
     * @code CaseDefinition c; c.name_="plate"; @endcode */
    struct CaseDefinition {
        unsigned schemaVersion_{1}; ///< 语义协议主版本。
        std::string name_; ///< 显示名称。
        std::string description_; ///< 说明。
        MeshSource mesh_; ///< 文件来源与坐标缩放。
        Analysis analysis_; ///< 分析模型。
        std::vector<Material> materials_; ///< 材料库。
        std::vector<MaterialAssignment> materialAssignments_; ///< 区域分配。
        std::vector<VolumeSource> sources_; ///< 可叠加体源。
        std::vector<BoundaryCondition> boundaries_; ///< 外边界条件。
        NaturalOverlap naturalOverlap_{NaturalOverlap::Error}; ///< 同侧自然条件叠加策略。
        std::optional<InitialState> initialState_; ///< 仅瞬态。
        std::optional<TimeOptions> time_; ///< 仅瞬态，方法固定后向Euler。
        Discretization discretization_; ///< 连续Lagrange、自动积分。
        LinearOptions linearSolver_; ///< 恒等预条件、失败终止。
        OutputOptions output_; ///< VTU温度输出意图。
    };
} // namespace emberfield::problem
