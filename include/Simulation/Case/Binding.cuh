/** @file Binding.cuh
 * @brief 定义只读绑定结果与分阶段能力报告，不执行FEM组装。
 */
#pragma once
#include <DataStructure/Mesh/FacetTopology.cuh>
#include <Simulation/Case/Validation.cuh>
#include <memory>

namespace emberfield::problem {
    /** @brief 仍需后续组件完成的阻断检查类型。 */
    enum class PendingKind { FacetTopology, Geometry, DegreesOfFreedom, Solvability, Assembly, ResultOutput };
    /** @brief 显式记录尚未完成的计算前提。
     * @code PendingCheck p{PendingKind::Geometry,"检查Jacobian和外法向"}; @endcode */
    struct PendingCheck {
        PendingKind kind_; ///< 待检阶段。
        std::string message_; ///< 尚未证明的性质。
    };
    /** @brief 条件到网格单元的去重映射，保持定义顺序。
     * @code BoundSelection s; s.definitionIndex_=0; @endcode */
    struct BoundSelection {
        std::size_t definitionIndex_{}; ///< 所属定义数组索引。
        std::vector<mesh::CellRef> cells_; ///< 已解析的单元成员。
        std::vector<std::size_t> facets_; ///< 边界对应的活动拓扑侧索引，拓扑未完成时为空。
    };
    /** @brief 不同阶段的能力诊断，不使合法定义被误认为可执行。
     * @code bool ready=report.executable(); @endcode */
    struct CapabilityReport {
        Diagnostics diagnostics_; ///< 已知后端不支持的组合。
        std::vector<PendingCheck> pending_; ///< 尚未完成的检查及执行组件。
        /** @brief 查询执行前提是否全部成立；无参数无额外前提。
         * @return 无能力错误且无待检项时为true。@code auto ready=r.executable(); @endcode */
        [[nodiscard]] bool executable() const noexcept { return !diagnostics_.hasErrors() && pending_.empty(); }
    };
    /** @brief 完整绑定构建值，由bindCase临时构造，不直接公开为已验证结果。
     * @code BindingData data; data.mesh_=mesh; @endcode */
    struct BindingData {
        std::shared_ptr<const mesh::Mesh> mesh_; ///< 不可变网格所有权。
        CaseDefinition definition_; ///< 定义快照。
        std::vector<mesh::CellRef> active_; ///< 活动单元稳定顺序。
        std::vector<std::size_t> materialIndices_; ///< 与active等长的材料索引。
        std::vector<BoundSelection> sources_; ///< 各源成员，不互相覆盖。
        std::vector<BoundSelection> boundaries_; ///< 各边界成员及侧映射。
        std::vector<BoundSelection> initialOverrides_; ///< 初温覆盖映射。
        mesh::FacetTopology topology_; ///< 独立活动域拓扑。
        std::vector<std::size_t> insulatedFacets_; ///< 未显式指定条件的外侧，仅拓扑完整时有效。
        CapabilityReport capability_; ///< 后续执行能力与前提。
    };
    struct BindResult;
    /** @brief 拥有已绑定的不可变网格与定义快照，使用组合保护索引生命周期。
     * @details 无公有构造器绕过校验；不包含时间状态、矩阵或GPU缓存。
     * @code auto r=bindCase(definition,mesh); if(r.bound_) use(r.bound_->data()); @endcode */
    class BoundCase {
    public:
        /** @brief 查询只读绑定数据；无参数无前提。
         * @return 受当前对象生命周期限制的引用。@code auto& d=bound.data(); @endcode */
        [[nodiscard]] const BindingData& data() const noexcept { return data_; }
        /** @brief 获取已缩放的米制坐标。
         * @param node 网格连续节点索引。@pre 索引有效。
         * @return 新坐标值；越界抛out_of_range。@code auto x=bound.position(0); @endcode */
        [[nodiscard]] std::array<double, 3> position(mesh::Index node) const;

    private:
        /** @brief 接管已验证的候选结果。@param data 候选值。@pre 仅绑定器可调用。
         * @note 无返回值。@code BoundCase c(std::move(data)); @endcode */
        explicit BoundCase(BindingData data) : data_(std::move(data)) {}
        friend BindResult bindCase(const CaseDefinition&, std::shared_ptr<const mesh::Mesh>, CaseLimits);
        BindingData data_; ///< 不可变公开的拥有数据。
    };
    /** @brief 绑定成功值与诊断；失败不产生半可用对象。
     * @code if(result.diagnostics_.hasErrors()) return; @endcode */
    struct BindResult {
        std::shared_ptr<const BoundCase> bound_; ///< 成功对象；错误时为空。
        Diagnostics diagnostics_; ///< 身份绑定错误与警告，区别于后端能力报告。
    };
    /** @brief 静态校验后绑定活动域、材料和条件；不读取文件、不上传GPU。
     * @param definition 定义，成功后保存独立快照。@param mesh 不可变网格共享所有权。
     * @param limits 记录/选择器/诊断限额。@pre 限额为正；mesh不得通过其他可变别名被并发修改。
     * @return 绑定与诊断；输入错误无绑定值，能力不足留在绑定的capability中。
     * @throws std::invalid_argument 非法限额；标准资源异常向上传递。
     * @code auto r=bindCase(definition,std::make_shared<const mesh::Mesh>(data)); @endcode */
    [[nodiscard]] BindResult bindCase(const CaseDefinition& definition, std::shared_ptr<const mesh::Mesh> mesh,
                                      CaseLimits limits = {});
} // namespace emberfield::problem
