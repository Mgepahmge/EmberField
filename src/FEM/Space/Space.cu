/** @file Space.cu
 * @brief 实现按活动顶点身份确定性编号的连续P1空间，不进行物理或几何处理。
 */
#include <FEM/Space/Space.cuh>
#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>

namespace thermofem::fem {
    const char* spaceIssueName(SpaceIssueCode code) noexcept {
        switch (code) {
        case SpaceIssueCode::NullMesh:
            return "SPACE_NULL_MESH";
        case SpaceIssueCode::EmptyActive:
            return "SPACE_EMPTY_ACTIVE";
        case SpaceIssueCode::LengthMismatch:
            return "SPACE_LENGTH_MISMATCH";
        case SpaceIssueCode::CellOutOfRange:
            return "SPACE_CELL_OUT_OF_RANGE";
        case SpaceIssueCode::DuplicateCell:
            return "SPACE_DUPLICATE_CELL";
        case SpaceIssueCode::DuplicateConnectivity:
            return "SPACE_DUPLICATE_CONNECTIVITY";
        case SpaceIssueCode::RepeatedNode:
            return "SPACE_REPEATED_NODE";
        case SpaceIssueCode::DimensionMismatch:
            return "SPACE_DIMENSION_MISMATCH";
        case SpaceIssueCode::UnsupportedElement:
            return "SPACE_UNSUPPORTED_ELEMENT";
        case SpaceIssueCode::IncompatibleGeometry:
            return "SPACE_INCOMPATIBLE_GEOMETRY";
        case SpaceIssueCode::ResourceLimit:
            return "SPACE_RESOURCE_LIMIT";
        case SpaceIssueCode::IndexOverflow:
            return "SPACE_INDEX_OVERFLOW";
        case SpaceIssueCode::UnappliedRelations:
            return "SPACE_UNAPPLIED_RELATIONS";
        }
        return "SPACE_UNKNOWN";
    }
    namespace {
        /** @brief 限长记录诊断，同时保留截断错误状态。
         * @param result 候选结果。@param limit 保存上限。@param code 稳定代码。
         * @param message 解释。@param cell 相关单元。@param localDof 可选局部位置。@param error 错误或警告。
         * @pre limit为正。@note 无返回值；分配异常传播。@code addIssue(r,100,code,"错误"); @endcode */
        void addIssue(SpaceBuildResult& result, std::size_t limit, SpaceIssueCode code, std::string message,
                      std::optional<mesh::CellRef> cell = {}, std::optional<std::size_t> localDof = {},
                      bool error = true) {
            result.hasErrors_ = result.hasErrors_ || error;
            if (result.issues_.size() < limit) {
                result.issues_.push_back({code, error, cell, localDof, std::move(message)});
            }
            else {
                result.truncated_ = true;
            }
        }
    } // namespace

    SpaceBuildResult buildFiniteElementSpace(std::shared_ptr<const mesh::Mesh> mesh,
                                             std::span<const mesh::CellRef> active,
                                             std::span<const ElementKey> elements, SpaceLimits limits) {
        if (!limits.maxCells_ || !limits.maxLocalDofs_ || !limits.maxGlobalDofs_ || !limits.maxDiagnostics_) {
            throw std::invalid_argument("空间构建限额必须为正");
        }
        SpaceBuildResult result;
        if (!mesh) {
            addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::NullMesh, "网格所有权为空");
        }
        if (active.empty()) {
            addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::EmptyActive, "活动单元集合为空");
        }
        if (active.size() != elements.size()) {
            addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::LengthMismatch, "活动单元与有限元键数量不一致");
        }
        if (active.size() > limits.maxCells_) {
            addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::ResourceLimit, "活动单元数超过限额");
        }
        if (active.size() == std::numeric_limits<std::size_t>::max()) {
            addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::IndexOverflow, "单元行边界长度溢出");
        }
        if (result.hasErrors_) {
            return result;
        }
        FiniteElementSpace candidate;
        candidate.mesh_ = mesh;
        candidate.cellOffsets_.push_back(0);
        std::set<mesh::CellRef> seenCells;
        std::set<std::pair<mesh::CellFamily, std::vector<mesh::Index>>> seenConnectivity;
        std::vector<mesh::Index> vertices;
        const auto& data = mesh->data();
        if (!data.periodic_.empty() || data.partitionCount_ || !data.partitionedEntities_.empty() ||
            !data.ghostEntities_.empty() || !data.ghostElements_.empty() || !data.elementPartitions_.empty()) {
            addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::UnappliedRelations,
                     "仅构造给定连接的局部未约束空间，不合并周期/幽灵自由度或建立分布式所有权", {}, {}, false);
        }
        for (std::size_t e = 0; e < active.size(); ++e) {
            const auto cell = active[e];
            if (cell.block_ >= data.elements_.size() || cell.element_ >= data.elements_[cell.block_].tags_.size()) {
                addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::CellOutOfRange, "活动单元引用越界", cell);
                continue;
            }
            if (!seenCells.insert(cell).second) {
                addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::DuplicateCell, "活动单元重复", cell);
                continue;
            }
            const auto key = elements[e];
            if (key.basisFamily_ != BasisFamily::ContinuousLagrange || key.order_ != 1 ||
                (key.family_ != mesh::CellFamily::Triangle && key.family_ != mesh::CellFamily::Tetrahedron)) {
                addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::UnsupportedElement, "未支持指定参考有限元",
                         cell);
                continue;
            }
            const auto& ref = referenceElement(key);
            const auto& geometry = data.elements_[cell.block_].type_;
            if (geometry.family_ != key.family_ || geometry.dimension_ != static_cast<int>(ref.dimension()) ||
                geometry.order_ != 1 || geometry.nodeCount_ != ref.dofCount() ||
                geometry.referenceLayout_ != "complete") {
                addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::IncompatibleGeometry,
                         "几何布局与首期完整一阶单纯形不兼容，不能按节点数猜测或降阶", cell);
                continue;
            }
            if (!candidate.dimension_) {
                candidate.dimension_ = ref.dimension();
            }
            else if (candidate.dimension_ != ref.dimension()) {
                addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::DimensionMismatch, "单个空间不能混合活动维数",
                         cell);
                continue;
            }
            const auto nodes = mesh->connectivity(cell.block_, cell.element_);
            std::vector<mesh::Index> sorted(nodes.begin(), nodes.end());
            std::sort(sorted.begin(), sorted.end());
            const auto repeated = std::adjacent_find(sorted.begin(), sorted.end());
            if (repeated != sorted.end()) {
                const auto local =
                    static_cast<std::size_t>(std::find(nodes.begin(), nodes.end(), *repeated) - nodes.begin());
                addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::RepeatedNode, "单元内节点身份重复", cell,
                         local);
                continue;
            }
            if (!seenConnectivity.emplace(key.family_, std::move(sorted)).second) {
                addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::DuplicateConnectivity,
                         "不同活动单元具有相同完整连接节点集合", cell);
                continue;
            }
            if (vertices.size() > std::numeric_limits<std::size_t>::max() - nodes.size()) {
                addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::IndexOverflow, "局部DOF总数溢出", cell);
                break;
            }
            if (nodes.size() > limits.maxLocalDofs_ - vertices.size()) {
                addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::ResourceLimit, "局部DOF总数超过限额", cell);
                break;
            }
            vertices.insert(vertices.end(), nodes.begin(), nodes.end());
            candidate.cellOffsets_.push_back(vertices.size());
        }
        if (result.hasErrors_) {
            return result;
        }
        auto unique = vertices;
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        if (unique.size() > limits.maxGlobalDofs_) {
            addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::ResourceLimit, "全局DOF数超过限额");
            return result;
        }
        if (std::cmp_greater(unique.size(), std::numeric_limits<DofIndex>::max())) {
            addIssue(result, limits.maxDiagnostics_, SpaceIssueCode::IndexOverflow, "全局DOF计数不能由DofIndex表示");
            return result;
        }
        candidate.dofInfo_.reserve(unique.size());
        for (auto node : unique) {
            candidate.dofInfo_.push_back({DofKind::Vertex, node});
        }
        candidate.cellDofs_.reserve(vertices.size());
        for (auto node : vertices) {
            candidate.cellDofs_.push_back(
                static_cast<DofIndex>(std::lower_bound(unique.begin(), unique.end(), node) - unique.begin()));
        }
        candidate.cells_.assign(active.begin(), active.end());
        candidate.elementKeys_.assign(elements.begin(), elements.end());
        candidate.cellLookup_.reserve(active.size());
        for (std::size_t e = 0; e < active.size(); ++e) {
            candidate.cellLookup_.emplace_back(active[e], e);
        }
        std::sort(candidate.cellLookup_.begin(), candidate.cellLookup_.end());
        result.space_ = std::shared_ptr<const FiniteElementSpace>(new FiniteElementSpace(std::move(candidate)));
        return result;
    }

    std::span<const DofIndex> FiniteElementSpace::cellDofs(std::size_t cell) const {
        if (cell >= cells_.size()) {
            throw std::out_of_range("空间单元行号越界");
        }
        return std::span<const DofIndex>(cellDofs_).subspan(cellOffsets_[cell],
                                                            cellOffsets_[cell + 1] - cellOffsets_[cell]);
    }
    std::optional<std::size_t> FiniteElementSpace::cellPosition(mesh::CellRef cell) const noexcept {
        const auto found = std::lower_bound(
            cellLookup_.begin(), cellLookup_.end(), cell,
            [](const std::pair<mesh::CellRef, std::size_t>& item, mesh::CellRef key) { return item.first < key; });
        if (found == cellLookup_.end() || found->first != cell) {
            return {};
        }
        return found->second;
    }
    const DofInfo& FiniteElementSpace::dofInfo(DofIndex dof) const {
        if (std::cmp_greater_equal(dof, dofInfo_.size())) {
            throw std::out_of_range("全局DOF越界");
        }
        return dofInfo_[static_cast<std::size_t>(dof)];
    }
    std::optional<DofIndex> FiniteElementSpace::nodeDof(mesh::Index node) const {
        if (node >= mesh_->data().nodes_.size()) {
            throw std::out_of_range("Mesh节点位置越界");
        }
        const auto found = std::lower_bound(dofInfo_.begin(), dofInfo_.end(), node,
                                            [](const DofInfo& info, mesh::Index key) { return *info.node_ < key; });
        if (found == dofInfo_.end() || found->node_ != node) {
            return {};
        }
        return static_cast<DofIndex>(found - dofInfo_.begin());
    }
    std::vector<DofIndex> facetDofs(const FiniteElementSpace& space, mesh::FacetUse use) {
        const auto e = space.cellPosition(use.cell_);
        if (!e) {
            throw std::out_of_range("侧的母单元不属于当前空间");
        }
        const auto local = referenceElement(space.elementKey(*e)).facetLocalDofs(use.localSide_);
        const auto row = space.cellDofs(*e);
        std::vector<DofIndex> result;
        result.reserve(local.size());
        for (auto a : local) {
            result.push_back(row[a]);
        }
        return result;
    }
} // namespace thermofem::fem
