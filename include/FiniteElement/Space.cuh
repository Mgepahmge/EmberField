/** @file Space.cuh
 * @brief 定义只读连续P1有限元空间、限长诊断和独立编号接口。
 */
#pragma once
#include <FiniteElement/ReferenceElement.cuh>
#include <Mesh/FacetTopology.cuh>
#include <memory>

namespace thermofem::fem {
    using DofIndex = std::uint64_t; ///< 空间内编号；不等于节点索引或GPU矩阵int索引。
    /** @brief 自由度来源；当前只构造顶点取值。 */
    enum class DofKind { Vertex };
    /** @brief 全局自由度的来源，不强迫未来所有自由度都关联一个节点。
     * @code auto node=space.dofInfo(0).node_; @endcode */
    struct DofInfo {
        DofKind kind_{DofKind::Vertex}; ///< 来源类别。
        std::optional<mesh::Index> node_; ///< 可选Mesh节点位置；当前Vertex始终有值。
    };
    /** @brief 空间构建问题的稳定代码，独立于物理算例和JSON路径。 */
    enum class SpaceIssueCode {
        NullMesh,
        EmptyActive,
        LengthMismatch,
        CellOutOfRange,
        DuplicateCell,
        DuplicateConnectivity,
        RepeatedNode,
        DimensionMismatch,
        UnsupportedElement,
        IncompatibleGeometry,
        ResourceLimit,
        IndexOverflow,
        UnappliedRelations
    };
    /** @brief 单条空间诊断，可携带单元与局部自由度位置。
     * @code SpaceIssue issue; issue.code_=SpaceIssueCode::EmptyActive; @endcode */
    struct SpaceIssue {
        SpaceIssueCode code_{SpaceIssueCode::EmptyActive}; ///< 稳定问题代码。
        bool error_{true}; ///< 错误阻止发布；警告只说明能力边界。
        std::optional<mesh::CellRef> cell_; ///< 相关活动单元，允许为空。
        std::optional<std::size_t> localDof_; ///< 可确定时的局部自由度号。
        std::string message_; ///< 中文解释，不作为机器错误码使用。
    };
    /** @brief 限制空间构建的工作量；所有上限必须为正。
     * @code SpaceLimits limits; limits.maxCells_=100; @endcode */
    struct SpaceLimits {
        std::size_t maxCells_{1000000}; ///< 活动单元上限。
        std::size_t maxLocalDofs_{4000000}; ///< 局部DOF总数上限。
        std::size_t maxGlobalDofs_{4000000}; ///< 全局DOF数上限。
        std::size_t maxDiagnostics_{100}; ///< 保存诊断条数，截断不丢错误状态。
    };
    class FiniteElementSpace;
    /** @brief 事务构建结果；错误时不发布空间，截断后仍保留错误标志。
     * @code if(result.space_) use(*result.space_); @endcode */
    struct SpaceBuildResult {
        std::shared_ptr<const FiniteElementSpace> space_; ///< 成功发布的只读空间。
        std::vector<SpaceIssue> issues_; ///< 限长诊断数组。
        bool hasErrors_{}; ///< 包括被截断条目的错误状态。
        bool truncated_{}; ///< 是否有诊断未保留。
    };
    /** @brief 组合拥有映射并共享只读Mesh的未约束标量空间。
     * @details 不删除Dirichlet节点，不读取物性、不计算Jacobian，不合并周期节点。
     * 调用者保证共形无悬挂网格；构建通过不等于几何、约束或可解性检查通过。
     * 构建后只读查询可并发；底层Mesh不得通过其他别名替换、移动或修改。
     * 复制深复制映射；移动后来源只允许析构或赋值。所有span受当前对象寿命限制。
     * @code auto dofs=result.space_->cellDofs(0); @endcode */
    class FiniteElementSpace {
    public:
        /** @brief 获取维数；无参数无额外前提。@return 2或3。@code auto d=s.dimension(); @endcode */
        [[nodiscard]] unsigned dimension() const noexcept { return dimension_; }
        /** @brief 获取单元数；无参数无额外前提。@return 活动单元数。@code auto n=s.cellCount(); @endcode */
        [[nodiscard]] std::size_t cellCount() const noexcept { return cells_.size(); }
        /** @brief 获取DOF数；无参数无额外前提。@return 全局DOF数。@code auto n=s.dofCount(); @endcode */
        [[nodiscard]] std::size_t dofCount() const noexcept { return dofInfo_.size(); }
        /** @brief 借用Mesh；无参数无额外前提。@return 对象寿命内的只读引用。
         * @code auto& m=s.mesh(); @endcode */
        [[nodiscard]] const mesh::Mesh& mesh() const noexcept { return *mesh_; }
        /** @brief 借用活动顺序；无参数无额外前提。@return 对象寿命内的只读span。
         * @code auto c=s.cells(); @endcode */
        [[nodiscard]] std::span<const mesh::CellRef> cells() const noexcept { return cells_; }
        /** @brief 获取一行局部到全局映射。@param cell 空间内行号，非单元tag。
         * @pre cell小于cellCount。@return 只读span；越界抛out_of_range。
         * @code auto row=s.cellDofs(0); @endcode */
        [[nodiscard]] std::span<const DofIndex> cellDofs(std::size_t cell) const;
        /** @brief 二分定位活动单元。@param cell Mesh中的单元位置。@pre 无。
         * @return 可选空间行号；非活动单元为空，不抛输入异常。@code auto e=s.cellPosition({0,0}); @endcode */
        [[nodiscard]] std::optional<std::size_t> cellPosition(mesh::CellRef cell) const noexcept;
        /** @brief 查询单元有限元键。@param cell 空间内行号。@pre cell有效。
         * @return 键值；越界抛out_of_range。@code auto k=s.elementKey(0); @endcode */
        [[nodiscard]] ElementKey elementKey(std::size_t cell) const { return elementKeys_.at(cell); }
        /** @brief 借用DOF来源。@param dof 全局编号。@pre dof有效。
         * @return 只读引用；越界抛out_of_range。@code auto& info=s.dofInfo(0); @endcode */
        [[nodiscard]] const DofInfo& dofInfo(DofIndex dof) const;
        /** @brief 查询活动顶点的DOF。@param node Mesh节点位置。@pre node在Mesh内。
         * @return 可选编号；未参与节点为空，节点越界抛out_of_range。
         * @code auto i=s.nodeDof(0); @endcode */
        [[nodiscard]] std::optional<DofIndex> nodeDof(mesh::Index node) const;

    private:
        /** @brief 仅构建器可创建候选。无参数；前提是发布前完成所有字段。
         * @note 无返回值，无分配。@code FiniteElementSpace candidate; @endcode */
        FiniteElementSpace() = default;
        friend SpaceBuildResult buildFiniteElementSpace(std::shared_ptr<const mesh::Mesh>,
                                                        std::span<const mesh::CellRef>, std::span<const ElementKey>,
                                                        SpaceLimits);
        std::shared_ptr<const mesh::Mesh> mesh_; ///< 网格共享只读所有权。
        unsigned dimension_{}; ///< 活动域维数。
        std::vector<mesh::CellRef> cells_; ///< 调用者的活动单元顺序。
        std::vector<ElementKey> elementKeys_; ///< 与cells等长的有限元键。
        std::vector<std::size_t> cellOffsets_; ///< 局部DOF行边界，长度E+1。
        std::vector<DofIndex> cellDofs_; ///< 连续局部到全局映射。
        std::vector<DofInfo> dofInfo_; ///< 按活动节点Index升序的来源。
        std::vector<std::pair<mesh::CellRef, std::size_t>> cellLookup_; ///< 按CellRef排序的空间行查询。
    };
    /** @brief 将代码转换为稳定字符串；无动态分配。
     * @param code 诊断代码。@pre 无。@return 静态字符串，未知枚举返回SPACE_UNKNOWN。
     * @code auto text=spaceIssueName(issue.code_); @endcode */
    [[nodiscard]] const char* spaceIssueName(SpaceIssueCode code) noexcept;
    /** @brief 构建连续P1空间；不依赖Case/JSON/CUDA。
     * @param mesh 共享只读网格。@param active 唯一有效、同维的活动单元。
     * @param elements 每活动单元的有限元键。@param limits 正资源预算。
     * @pre 调用者保证几何共形无悬挂，且不通过别名修改底层Mesh；输入span只在调用期借用。
     * @return 完整空间或限长诊断；非法输入和超限不发布部分空间。
     * @throws std::invalid_argument 非法限额；标准资源异常传播。无设备操作。
     * @code auto r=buildFiniteElementSpace(mesh,active,keys); @endcode */
    [[nodiscard]] SpaceBuildResult buildFiniteElementSpace(std::shared_ptr<const mesh::Mesh> mesh,
                                                           std::span<const mesh::CellRef> active,
                                                           std::span<const ElementKey> elements,
                                                           SpaceLimits limits = {});
    /** @brief 按母单元侧参数顺序提取全局DOF，不排序、不判断外边界。
     * @param space 有效空间。@param use 活动母单元及其局部侧号。@pre 母单元属于空间，侧号有效。
     * @return 独立拥有的侧DOF数组；非法引用抛out_of_range，分配异常传播。
     * @code auto ids=facetDofs(space,{{0,0},2}); @endcode */
    [[nodiscard]] std::vector<DofIndex> facetDofs(const FiniteElementSpace& space, mesh::FacetUse use);
} // namespace thermofem::fem
