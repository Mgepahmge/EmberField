/** @file Conditions.cuh
 * @brief 将绑定物理条件关联到既有空间，不进行几何运算或矩阵消元。
 */
#pragma once
#include <Simulation/Case/PreparationOptions.cuh>
#include <Simulation/Case/TemperatureSpace.cuh>
namespace thermofem::problem {
    class DirichletConstraints;
    class PreparedConditions;
    class ConditionPreparationResult;
    /** @brief 归并规定温度。@param bound 绑定定义。@param space 同源空间。
     * @param maxAssociations 提案访问上限。@param diagnostics 追加诊断。
     * @pre 正预算。@return 有值表示成功（可空约束），冲突nullopt；系统异常传播。
     * @code auto c=buildDirichletConstraints(b,s,limit,d); @endcode */
    [[nodiscard]] std::optional<DirichletConstraints> buildDirichletConstraints(const BoundCase& bound,
                                                                                const fem::FiniteElementSpace& space,
                                                                                std::size_t maxAssociations,
                                                                                Diagnostics& diagnostics);
    /** @brief 插值并检查瞬态初态。@param bound 绑定瞬态。@param space 同源空间。
     * @param constraints 同源已归并约束。@param maxAssociations DOF访问上限。@param diagnostics 追加诊断。
     * @pre constraints来自相同bound/space。@return 完整全DOF初温，失败nullopt。
     * @code auto t=buildInitialTemperature(b,s,c,limit,d); @endcode */
    [[nodiscard]] std::optional<std::vector<double>> buildInitialTemperature(const BoundCase& bound,
                                                                             const fem::FiniteElementSpace& space,
                                                                             const DirichletConstraints& constraints,
                                                                             std::size_t maxAssociations,
                                                                             Diagnostics& diagnostics);
    /** @brief 准备完整离散条件。@param bound 绑定所有权。@param space 空间所有权。
     * @param options 正限额与合法策略。@pre 原网格不经可变别名修改。
     * @return 不可变条件或限长诊断；非法策略invalid_argument，资源异常传播。
     * @code auto c=prepareConditions(bound,space); @endcode */
    [[nodiscard]] ConditionPreparationResult prepareConditions(std::shared_ptr<const BoundCase> bound,
                                                               std::shared_ptr<const fem::FiniteElementSpace> space,
                                                               const PreparationOptions& options = {});
    /** @brief 排序唯一的全空间温度约束及压缩来源，不缩减空间。
     * @code auto c=buildDirichletConstraints(b,s,limit,d); @endcode */
    class DirichletConstraints {
    public:
        /** @brief 规定温度DOF；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.dofs(); @endcode */
        [[nodiscard]] const std::vector<fem::DofIndex>& dofs() const noexcept { return dofs_; }

        /** @brief K值；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.temperatures(); @endcode */
        [[nodiscard]] const std::vector<double>& temperatures() const noexcept { return temperatures_; }

        /** @brief 来源段边界；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.sourceOffsets();
         * @endcode */
        [[nodiscard]] const std::vector<std::size_t>& sourceOffsets() const noexcept { return sourceOffsets_; }

        /** @brief 边界定义索引；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.sourceIndices();
         * @endcode */
        [[nodiscard]] const std::vector<std::size_t>& sourceIndices() const noexcept { return sourceIndices_; }

    private:
        DirichletConstraints() = default; ///< 仅条件工厂候选。
        friend class PreparedConditions;
        friend std::optional<DirichletConstraints>
        buildDirichletConstraints(const BoundCase&, const fem::FiniteElementSpace&, std::size_t, Diagnostics&);
        std::vector<fem::DofIndex> dofs_; ///< 严格递增DOF。
        std::vector<double> temperatures_; ///< 与DOF等长。
        std::vector<std::size_t> sourceOffsets_{0}; ///< 每个DOF来源段。
        std::vector<std::size_t> sourceIndices_; ///< 唯一边界定义索引。
    };
    /** @brief 一个自然条件与一个活动外侧的有序关联。
     * @code auto& term=conditions.naturalTerms()[0]; @endcode */
    class NaturalBoundaryTerm {
    public:
        /** @brief 边界定义行；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.definitionIndex();
         * @endcode */
        [[nodiscard]] std::size_t definitionIndex() const noexcept { return definitionIndex_; }

        /** @brief 拓扑侧行；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.facetIndex(); @endcode */
        [[nodiscard]] std::size_t facetIndex() const noexcept { return facetIndex_; }

        /** @brief 母单元侧；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.use(); @endcode */
        [[nodiscard]] mesh::FacetUse use() const noexcept { return use_; }

        /** @brief 迹到母局部序；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.parentLocalDofs();
         * @endcode */
        [[nodiscard]] const std::vector<std::size_t>& parentLocalDofs() const noexcept { return parentLocalDofs_; }

        /** @brief 有序全局迹DOF；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.globalDofs();
         * @endcode */
        [[nodiscard]] const std::vector<fem::DofIndex>& globalDofs() const noexcept { return globalDofs_; }

    private:
        NaturalBoundaryTerm() = default; ///< 仅条件工厂候选。
        friend ConditionPreparationResult prepareConditions(std::shared_ptr<const BoundCase>,
                                                            std::shared_ptr<const fem::FiniteElementSpace>,
                                                            const PreparationOptions&);
        std::size_t definitionIndex_{}; ///< 定义行。
        std::size_t facetIndex_{}; ///< 外侧行。
        mesh::FacetUse use_; ///< 母单元侧。
        std::vector<std::size_t> parentLocalDofs_; ///< 侧到母局部顺序。
        std::vector<fem::DofIndex> globalDofs_; ///< 全局侧顺序。
    };
    /** @brief 组合拥有绑定、空间、材料/载荷关联及无冲突温度状态。
     * @code auto& c=*prepareConditions(b,s).conditions(); @endcode */
    class PreparedConditions {
    public:
        /** @brief 绑定来源；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.bound(); @endcode */
        [[nodiscard]] const std::shared_ptr<const BoundCase>& bound() const noexcept { return bound_; }

        /** @brief 空间来源；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.space(); @endcode */
        [[nodiscard]] const std::shared_ptr<const fem::FiniteElementSpace>& space() const noexcept { return space_; }

        /** @brief 单元材料行；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.materialIndices();
         * @endcode */
        [[nodiscard]] const std::vector<std::size_t>& materialIndices() const noexcept { return materialIndices_; }

        /** @brief 单元源关联偏移；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.sourceOffsets();
         * @endcode */
        [[nodiscard]] const std::vector<std::size_t>& sourceOffsets() const noexcept { return sourceOffsets_; }

        /** @brief 源定义行；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.sourceIndices(); @endcode
         */
        [[nodiscard]] const std::vector<std::size_t>& sourceIndices() const noexcept { return sourceIndices_; }

        /** @brief 自然侧项；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.naturalTerms(); @endcode
         */
        [[nodiscard]] const std::vector<NaturalBoundaryTerm>& naturalTerms() const noexcept { return naturalTerms_; }

        /** @brief 唯一规定温度；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.constraints();
         * @endcode */
        [[nodiscard]] const DirichletConstraints& constraints() const noexcept { return constraints_; }

        /** @brief 瞬态全DOF初态；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.initialTemperature();
         * @endcode */
        [[nodiscard]] const std::optional<std::vector<double>>& initialTemperature() const noexcept {
            return initialTemperature_;
        }

    private:
        PreparedConditions() = default; ///< 仅工厂候选。
        friend ConditionPreparationResult prepareConditions(std::shared_ptr<const BoundCase>,
                                                            std::shared_ptr<const fem::FiniteElementSpace>,
                                                            const PreparationOptions&);
        std::shared_ptr<const BoundCase> bound_; ///< 定义共享所有权。
        std::shared_ptr<const fem::FiniteElementSpace> space_; ///< 同源空间。
        std::vector<std::size_t> materialIndices_; ///< 逐活动行材料。
        std::vector<std::size_t> sourceOffsets_; ///< 关联行指针。
        std::vector<std::size_t> sourceIndices_; ///< 单元内源定义。
        std::vector<NaturalBoundaryTerm> naturalTerms_; ///< 单个条件/侧关联。
        DirichletConstraints constraints_; ///< 规定温度。
        std::optional<std::vector<double>> initialTemperature_; ///< 瞬态初态。
    };
    /** @brief 条件准备结果，截断诊断也不能变为成功。
     * @code auto r=prepareConditions(b,s); if(r.ok()) use(r.conditions()); @endcode */
    class ConditionPreparationResult {
    public:
        /** @brief 初始化诊断预算。@param limit 正上限。@pre limit>0。@note 无返回值。
         * @code ConditionPreparationResult r(100); @endcode */
        explicit ConditionPreparationResult(std::size_t limit) : diagnostics_(limit) {}
        /** @brief 完整条件对象；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.conditions(); @endcode
         */
        [[nodiscard]] const std::shared_ptr<const PreparedConditions>& conditions() const noexcept {
            return conditions_;
        }
        /** @brief 本次诊断；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.diagnostics(); @endcode */
        [[nodiscard]] const Diagnostics& diagnostics() const noexcept { return diagnostics_; }
        /** @brief 条件是否成功；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.ok(); @endcode */
        [[nodiscard]] bool ok() const noexcept { return conditions_ && !diagnostics_.hasErrors(); }

    private:
        friend ConditionPreparationResult prepareConditions(std::shared_ptr<const BoundCase>,
                                                            std::shared_ptr<const fem::FiniteElementSpace>,
                                                            const PreparationOptions&);
        std::shared_ptr<const PreparedConditions> conditions_; ///< 仅全成功非空。
        Diagnostics diagnostics_; ///< 限长诊断。
    };
} // namespace thermofem::problem
