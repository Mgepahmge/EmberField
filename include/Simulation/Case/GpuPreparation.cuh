/** @file GpuPreparation.cuh
 * @brief 将已准备算例按有界批次交给独立GPU算子，不在配置层重写积分。
 */
#pragma once
#include <Simulation/Case/Preparation.cuh>
#include <Simulation/Thermal/GpuElementOperations.cuh>
namespace emberfield::problem {
    class GpuThermalCellTerms;
    class GpuThermalBoundaryTerms;
    /** @brief 按输入活动行顺序批量计算体贡献。@param prepared 联合上下文。
     * @param cellRows 非空活动行数组。@param limits 批预算。
     * @pre Mesh不可变；调用者决定流式分批，当前设备有效。
     * @return 拥有设备K、可选M和体源合计的批；越界/非法形状/数值/CUDA错误传播。
     * @code auto t=evaluateCellBatch(prepared,rows); @endcode */
    [[nodiscard]] GpuThermalCellTerms evaluateCellBatch(const PreparedThermalCase& prepared,
                                                        std::span<const std::size_t> cellRows,
                                                        thermal::GpuBatchLimits limits = {});
    /** @brief 同类型自然项批积分。@param prepared 上下文。@param naturalTermIndices 自然项行。
     * @param limits 批预算。@pre 非空且全部Flux或全部Convection，按原输入行顺序。
     * @return 设备侧贡献；混合类型invalid_argument，数值/CUDA失败传播。
     * @code auto t=evaluateBoundaryBatch(prepared,indices); @endcode */
    [[nodiscard]] GpuThermalBoundaryTerms evaluateBoundaryBatch(const PreparedThermalCase& prepared,
                                                                std::span<const std::size_t> naturalTermIndices,
                                                                thermal::GpuBatchLimits limits = {});
    /** @brief 一批体单元的设备物理贡献与活动行顺序，供GPU装配消费。
     * @code auto t=evaluateCellBatch(p,rows); @endcode */
    class GpuThermalCellTerms {
    public:
        /** @brief 结果实体到空间行；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.cellRows();
         * @endcode */
        [[nodiscard]] const std::vector<std::size_t>& cellRows() const noexcept { return cellRows_; }
        /** @brief 设备K；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.conductivity(); @endcode */
        [[nodiscard]] const thermal::GpuLocalMatrixBatch& conductivity() const noexcept { return conductivity_; }
        /** @brief 瞬态设备M；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.capacity(); @endcode */
        [[nodiscard]] const std::optional<thermal::GpuLocalMatrixBatch>& capacity() const noexcept { return capacity_; }
        /** @brief 设备源合计；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.load(); @endcode */
        [[nodiscard]] const thermal::GpuLocalVectorBatch& load() const noexcept { return load_; }

    private:
        friend GpuThermalCellTerms evaluateCellBatch(const PreparedThermalCase&, std::span<const std::size_t>,
                                                     thermal::GpuBatchLimits);
        /** @brief 接管完整结果。@param rows 行。@param k K。@param m M。@param f 源。
         * @pre 同源同序。@note 无返回值。@code GpuThermalCellTerms t(rows,k,m,f); @endcode */
        GpuThermalCellTerms(std::vector<std::size_t> rows, thermal::GpuLocalMatrixBatch k,
                            std::optional<thermal::GpuLocalMatrixBatch> m, thermal::GpuLocalVectorBatch f) :
            cellRows_(std::move(rows)), conductivity_(std::move(k)), capacity_(std::move(m)), load_(std::move(f)) {}
        std::vector<std::size_t> cellRows_; ///< 空间活动行。
        thermal::GpuLocalMatrixBatch conductivity_; ///< 设备K。
        std::optional<thermal::GpuLocalMatrixBatch> capacity_; ///< 设备M。
        thermal::GpuLocalVectorBatch load_; ///< 设备体源合计。
    };
    /** @brief 一批同类型自然项及其稳定来源索引，不提前合并h/T∞。
     * @code auto t=evaluateBoundaryBatch(p,rows); @endcode */
    class GpuThermalBoundaryTerms {
    public:
        using Contribution =
            std::variant<thermal::GpuLocalVectorBatch, thermal::GpuConvectionBatch>; ///< 通量或对流设备结果。
        /** @brief 结果到自然项行；无参数无额外前提。@return 只读值或借用引用。@code auto
         * value=obj.naturalTermIndices(); @endcode */
        [[nodiscard]] const std::vector<std::size_t>& naturalTermIndices() const noexcept {
            return naturalTermIndices_;
        }
        /** @brief 设备侧贡献；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.contribution(); @endcode
         */
        [[nodiscard]] const Contribution& contribution() const noexcept { return contribution_; }

    private:
        friend GpuThermalBoundaryTerms evaluateBoundaryBatch(const PreparedThermalCase&, std::span<const std::size_t>,
                                                             thermal::GpuBatchLimits);
        /** @brief 接管侧结果。@param rows 行。@param c 贡献。@pre 同序。
         * @note 无返回值。@code GpuThermalBoundaryTerms t(rows,c); @endcode */
        GpuThermalBoundaryTerms(std::vector<std::size_t> rows, Contribution c) :
            naturalTermIndices_(std::move(rows)), contribution_(std::move(c)) {}
        std::vector<std::size_t> naturalTermIndices_; ///< 配置自然关联行。
        Contribution contribution_; ///< 设备结果。
    };
} // namespace emberfield::problem
