/** @file PostprocessOperations.cuh
 * @brief 独立的P1场重建、Fourier热流及积分归约GPU运算。
 */
#pragma once
#include <DataStructure/Storage/Vector.cuh>
#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>
namespace emberfield::thermal {
    /** @brief 一次主机准备的P1单元数据。@code P1CellSample c; @endcode */
    struct P1CellSample {
        std::uint64_t dofs_[4]; ///< 有效局部DOF，长度dimension+1。
        double gradients_[12]; ///< a*3+k，二维z为0。
        unsigned dimension_; ///< 二或三。
        double volume_; ///< 含二维厚度的体积。
        double conductivity_; ///< 导热率。
        double capacity_; ///< 体热容，稳态0。
        double source_; ///< 合计体热源。
    };
    /** @brief 逐单元GPU重建后的主机结果。@code P1CellValue v; @endcode */
    struct P1CellValue {
        double gradient_[3]; ///< 温度梯度。
        double flux_[3]; ///< 未乘厚度的热流密度。
        double minimum_; ///< 局部节点最小温度。
        double maximum_; ///< 局部节点最大温度。
        double integral_; ///< 体积温度积分。
        double energy_; ///< 储能。
    };
    /** @brief 同时计算批内P1梯度和积分。@param cells 有效单元数据。@param temperature 同设备global状态。
     * @pre 当前设备与输入一致，无并发修改。@return 同序拥有的主机结果。
     * @throws std::invalid_argument 非法形状；CudaError设备失败；overflow_error非有限结果。
     * @code auto v=evaluateP1Gradient(cells,t); @endcode */
    std::vector<P1CellValue> evaluateP1Gradient(std::span<const P1CellSample> cells, const Vector<double>& temperature);
    /** @brief GPU计算q=-k gradT。@param gradients 三分量扁平梯度。@param conductivity 每单元正系数。
     * @pre 同序有限。@return 同序三分量热流，非有限抛overflow_error。
     * @code auto q=applyFourierLaw(g,k); @endcode */
    std::vector<std::array<double, 3>> applyFourierLaw(std::span<const std::array<double, 3>> gradients,
                                                       std::span<const double> conductivity);
    /** @brief 已积分的侧面线性功率项。@code BoundaryPowerSample s; @endcode */
    struct BoundaryPowerSample {
        std::uint64_t dofs_[3]; ///< 迹DOF。
        double weights_[3]; ///< ∫h Na，通量时为0。
        unsigned count_; ///< 二或三。
        double constant_; ///< ∫q或-∫hT∞。
    };
    /** @brief GPU侧面功率积分的线性求值。@param samples 一批预积分项。@param temperature 温度。
     * @pre 样本合法、当前设备一致。@return 逐侧向外功率；非法invalid_argument。
     * @code auto p=integrateBoundaryPowerBatch(s,t); @endcode */
    std::vector<double> integrateBoundaryPowerBatch(std::span<const BoundaryPowerSample> samples,
                                                    const Vector<double>& temperature);
    /** @brief GPU求和和选择DOF归约。@param values 设备数值。@param selected 空时归约所有，否则只归约指定索引。
     * @pre 当前设备一致、索引有效。@return 有限总和；数值失败overflow_error。
     * @code auto total=sumValues(v); @endcode */
    double sumValues(const Vector<double>& values, std::span<const std::uint64_t> selected = {});
    /** @brief 区域批归约结果。@code RegionStatistics s{}; @endcode */
    struct RegionStatistics {
        double minimum_, maximum_, integral_, energy_, volume_; ///< 极值、温度体积分、储能与体积。
    };
    /** @brief GPU归约一批区域单元。@param values 同序重建值。@param volumes 正单元体积。
     * @pre 非空等长有限数组。@return 五个标量；非法invalid_argument，非有限overflow_error。
     * @code auto s=reduceRegion(v,volume); @endcode */
    RegionStatistics reduceRegion(std::span<const P1CellValue> values, std::span<const double> volumes);
    /** @brief 节点或单纯形内插值项。@code TemperatureSample s{}; @endcode */
    struct TemperatureSample {
        std::uint64_t dofs_[4]; ///< 全局索引。
        double weights_[4]; ///< 非负重心权重，总和1。
        unsigned count_; ///< 1..4。
    };
    /** @brief GPU gather及加权插值，只回读传感器标量。@param samples 已定位样本。
     * @param temperature 当前全DOF温度。@pre global同设备。@return 同序温度。
     * @throws std::invalid_argument 非法索引/权重；数值/CUDA异常传播。
     * @code auto v=interpolateTemperature(s,t); @endcode */
    std::vector<double> interpolateTemperature(std::span<const TemperatureSample> samples,
                                               const Vector<double>& temperature);
} // namespace emberfield::thermal

/** @brief 后处理专用的有界设备执行组件，不改变独立主机适配接口。 */
namespace emberfield::thermal::detail {
    /** @brief 固定P1样本可选常驻，批中间场留在设备；不可并发。
     * @details 借用samples至析构，调用方禁止修改/搬移；最多缓存cacheBytes，超限显式流式。
     * @code P1PostprocessWorkspace work(samples,4096,256*1024*1024); @endcode */
    class P1PostprocessWorkspace {
    public:
        /** @brief 建立批工作区。@param samples 静态样本借用。@param batchCapacity 最大批。
         * @param cacheBytes 可选静态缓存字节上限，0禁用。@pre 样本生命周期覆盖本对象。
         * @return 无，非法参数invalid_argument，CUDA错误传播。
         * @code P1PostprocessWorkspace work(samples,4096,0); @endcode */
        P1PostprocessWorkspace(std::span<const P1CellSample> samples, std::size_t batchCapacity,
                               std::size_t cacheBytes);
        /** @brief 释放设备资源，无参数无额外前提。@return 无。
         * @code { P1PostprocessWorkspace work(samples,4096,0); } @endcode */
        ~P1PostprocessWorkspace() noexcept;
        P1PostprocessWorkspace(const P1PostprocessWorkspace&) = delete; ///< 独占批资源。
        P1PostprocessWorkspace& operator=(const P1PostprocessWorkspace&) = delete; ///< 不复制。
        /** @brief 设备梯度与Fourier连续计算后一次下载最终字段。
         * @param first 首单元。@param count 批长。@param temperature 温度。
         * @pre 合法非空切片，同设备global且温度已验证。@return 独立主机结果。
         * @code auto values = work.fields(0,n,t); @endcode */
        std::vector<P1CellValue> fields(std::size_t first, std::size_t count, const Vector<double>& temperature);
        /** @brief 只下载区域统计，不下载逐单元结果。
         * @param first 首单元。@param count 批长。@param temperature 温度。
         * @pre 同fields；保留原单block256归约次序。@return 区域统计。
         * @code auto stats = work.region(0,n,t); @endcode */
        RegionStatistics region(std::size_t first, std::size_t count, const Vector<double>& temperature);

    private:
        struct Impl; ///< 静态样本借用、设备输入/输出/状态。
        std::unique_ptr<Impl> impl_; ///< 独占执行状态。
    };
} // namespace emberfield::thermal::detail
