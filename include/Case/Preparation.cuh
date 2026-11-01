/** @file Preparation.cuh
 * @brief 组合几何与物理条件，并把配置适配为独立局部热运算。
 */
#pragma once
#include <Case/Conditions.cuh>
#include <Thermal/ElementOperations.cuh>
namespace thermofem::problem {
    /** @brief 本次准备的独立阶段，不等同完整执行能力。 */
    enum class PreparationStage { Geometry, IntegrationCompatibility, Conditions };
    /** @brief 某阶段未执行、通过或失败。 */
    enum class StageStatus { NotRun, Passed, Failed };
    class PreparationResult;
    class PreparedThermalCase;
    class ThermalCellTerms;
    /** @brief 联合准备。@param bound 绑定。@param space 精确同源空间。@param options 策略。
     * @pre 原始Mesh不经别名修改。@return 本次阶段结果与完整准备对象；不生成A,b。
     * @throws std::invalid_argument 非法策略；系统异常传播。
     * @code auto r=prepareThermalCase(bound,space); @endcode */
    [[nodiscard]] PreparationResult prepareThermalCase(std::shared_ptr<const BoundCase> bound,
                                                       std::shared_ptr<const fem::FiniteElementSpace> space,
                                                       const PreparationOptions& options = {});
    /** @brief 计算一行体局部项。@param prepared 联合对象。@param cellRow 活动行。
     * @pre 来源不变。@return K、瞬态M及合计体源；数值失败EvaluationError，越界out_of_range。
     * @code auto t=evaluateCellTerms(prepared,0); @endcode */
    [[nodiscard]] ThermalCellTerms evaluateCellTerms(const PreparedThermalCase& prepared, std::size_t cellRow);
    /** @brief 计算单个自然侧项。@param prepared 联合对象。@param naturalTermIndex 自然关联行。
     * @pre 来源不变。@return 通量向量或对流组合，数值失败EvaluationError，越界out_of_range。
     * @code auto t=evaluateBoundaryTerm(prepared,0); @endcode */
    [[nodiscard]] std::variant<thermal::LocalVector, thermal::ConvectionContribution>
    evaluateBoundaryTerm(const PreparedThermalCase& prepared, std::size_t naturalTermIndex);
    /** @brief 组合成功几何和条件，不持久保存全部单元矩阵。
     * @code auto p=prepareThermalCase(b,s).prepared(); @endcode */
    class PreparedThermalCase {
    public:
        /** @brief 几何准备；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.geometry(); @endcode */
        [[nodiscard]] const std::shared_ptr<const fem::PreparedGeometry>& geometry() const noexcept {
            return geometry_;
        }
        /** @brief 条件准备；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.conditions(); @endcode */
        [[nodiscard]] const std::shared_ptr<const PreparedConditions>& conditions() const noexcept {
            return conditions_;
        }
        /** @brief 唯一空间；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.space(); @endcode */
        [[nodiscard]] const std::shared_ptr<const fem::FiniteElementSpace>& space() const noexcept {
            return geometry_->space();
        }
        /** @brief 绑定定义；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.bound(); @endcode */
        [[nodiscard]] const std::shared_ptr<const BoundCase>& bound() const noexcept { return conditions_->bound(); }

    private:
        friend PreparationResult prepareThermalCase(std::shared_ptr<const BoundCase>,
                                                    std::shared_ptr<const fem::FiniteElementSpace>,
                                                    const PreparationOptions&);
        PreparedThermalCase() = default; ///< 仅联合工厂候选。
        std::shared_ptr<const fem::PreparedGeometry> geometry_; ///< 同一空间上的几何。
        std::shared_ptr<const PreparedConditions> conditions_; ///< 同一空间上的物理关联。
    };
    /** @brief 保存本次阶段结论，不修改BoundCase旧pending。
     * @code auto r=prepareThermalCase(b,s); if(!r.ok()) inspect(r.diagnostics()); @endcode */
    class PreparationResult {
    public:
        /** @brief 初始化诊断预算。@param limit 正预算。@pre limit>0。@note 无返回值。
         * @code PreparationResult r(100); @endcode */
        explicit PreparationResult(std::size_t limit) : diagnostics_(limit) {}
        /** @brief 完整联合对象；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.prepared(); @endcode
         */
        [[nodiscard]] const std::shared_ptr<const PreparedThermalCase>& prepared() const noexcept { return prepared_; }
        /** @brief 本次诊断；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.diagnostics(); @endcode */
        [[nodiscard]] const Diagnostics& diagnostics() const noexcept { return diagnostics_; }
        /** @brief 各阶段状态；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.stages(); @endcode */
        [[nodiscard]] const std::array<StageStatus, 3>& stages() const noexcept { return stages_; }

        /** @brief 查询阶段。@param value 阶段枚举。@pre 无。
         * @return 结论，非法枚举out_of_range。@code auto status=r.stage(PreparationStage::Geometry); @endcode */
        [[nodiscard]] StageStatus stage(PreparationStage value) const {
            return stages_.at(static_cast<std::size_t>(value));
        }
        /** @brief 判断当前准备是否成功；无参数无额外前提。
         * @return 三阶段全部通过且有成功对象无错误。@code if(r.ok()) use(r.prepared()); @endcode */
        [[nodiscard]] bool ok() const noexcept {
            return prepared_ && !diagnostics_.hasErrors() && stages_[0] == StageStatus::Passed &&
                stages_[1] == StageStatus::Passed && stages_[2] == StageStatus::Passed;
        }

    private:
        friend PreparationResult prepareThermalCase(std::shared_ptr<const BoundCase>,
                                                    std::shared_ptr<const fem::FiniteElementSpace>,
                                                    const PreparationOptions&);
        std::shared_ptr<const PreparedThermalCase> prepared_; ///< 全通过才非空。
        Diagnostics diagnostics_; ///< 限长且保留失败标志。
        std::array<StageStatus, 3> stages_{StageStatus::NotRun, StageStatus::NotRun,
                                           StageStatus::NotRun}; ///< 阶段状态。
    };
    /** @brief 同一个体单元的物理贡献组合，不包含自然侧项。
     * @code auto t=evaluateCellTerms(p,0); @endcode */
    class ThermalCellTerms {
    public:
        /** @brief 来源活动行；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.cellRow(); @endcode */
        [[nodiscard]] std::size_t cellRow() const noexcept { return cellRow_; }
        /** @brief 导热K；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.conductivity(); @endcode */
        [[nodiscard]] const thermal::LocalMatrix& conductivity() const noexcept { return conductivity_; }
        /** @brief 瞬态一致容量M；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.capacity(); @endcode
         */
        [[nodiscard]] const std::optional<thermal::LocalMatrix>& capacity() const noexcept { return capacity_; }
        /** @brief 合计体源载荷；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.load(); @endcode */
        [[nodiscard]] const thermal::LocalVector& load() const noexcept { return load_; }

    private:
        friend ThermalCellTerms evaluateCellTerms(const PreparedThermalCase&, std::size_t);
        /** @brief 组合本单元的独立项。@param row 活动行。@param k 导热。
         * @param m 可选容量。@param f 体源合计。@pre 同序同尺寸。
         * @note 无返回值。@code ThermalCellTerms t(row,k,m,f); @endcode */
        ThermalCellTerms(std::size_t row, thermal::LocalMatrix k, std::optional<thermal::LocalMatrix> m,
                         thermal::LocalVector f) :
            cellRow_(row), conductivity_(std::move(k)), capacity_(std::move(m)), load_(std::move(f)) {}
        std::size_t cellRow_; ///< 来源行。
        thermal::LocalMatrix conductivity_; ///< K。
        std::optional<thermal::LocalMatrix> capacity_; ///< 瞬态M。
        thermal::LocalVector load_; ///< 合计体源。
    };
} // namespace thermofem::problem
