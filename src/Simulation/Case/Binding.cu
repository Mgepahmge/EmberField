/** @file Binding.cu
 * @brief 实现算例到活动域的事务绑定，明确区分身份合法与后端可执行。
 */
#include <Simulation/Case/Binding.cuh>
#include <Simulation/Case/Region.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace emberfield::problem {
    std::array<double, 3> BoundCase::position(mesh::Index node) const {
        auto p = data_.mesh_->data().nodes_.at(node).position_;
        for (auto& value : p) {
            value *= data_.definition_.mesh_.coordinateScale_;
        }
        return p;
    }
    namespace {
        /** @brief 生成独立绑定诊断。@param report 结果集合。@param code 错误码。
         * @param path 字段路径。@param message 消息。@param id 定义id。@param tag 单元标签。
         * @pre 无。@note 无返回值。@code issue(report,"MATERIAL_GAP","/","缺失"); @endcode */
        void issue(Diagnostics& report, std::string code, std::string path, std::string message, std::string id = {},
                   std::optional<mesh::Tag> tag = {}) {
            report.add({Severity::Error, std::move(code), std::move(path), std::move(id), tag, std::move(message),
                        "检查网格分组及算例定义"});
        }
        /** @brief 判断首个计算后端计划支持的域布局；不表示已存在组装器。
         * @param t 参考类型。@param dimension 物理维数。@pre 无。
         * @return 是否为完整一阶Triangle3/Tet4。@code auto ok=firstOrderDomain(type,2); @endcode */
        bool firstOrderDomain(const mesh::ElementType& t, int dimension) {
            return t.order_ == 1 && t.referenceLayout_ == "complete" && t.dimension_ == dimension &&
                ((dimension == 2 && t.family_ == mesh::CellFamily::Triangle && t.nodeCount_ == 3) ||
                 (dimension == 3 && t.family_ == mesh::CellFamily::Tetrahedron && t.nodeCount_ == 4));
        }
        /** @brief 判断可靠的显式低维线性布局。@param t 参考类型。@pre 无。
         * @return 是否可按完整节点身份匹配线性侧。@code auto ok=linearBoundary(type); @endcode */
        bool linearBoundary(const mesh::ElementType& t) {
            return t.order_ == 1 && t.referenceLayout_ == "complete" &&
                ((t.dimension_ == 1 && t.family_ == mesh::CellFamily::Segment && t.nodeCount_ == 2) ||
                 (t.dimension_ == 2 && t.family_ == mesh::CellFamily::Triangle && t.nodeCount_ == 3) ||
                 (t.dimension_ == 2 && t.family_ == mesh::CellFamily::Quadrilateral && t.nodeCount_ == 4));
        }
    } // namespace
    BindResult bindCase(const CaseDefinition& definition, std::shared_ptr<const mesh::Mesh> mesh, CaseLimits limits) {
        auto diagnostics = validateCase(definition, limits);
        if (!mesh) {
            issue(diagnostics, "NULL_MESH", "/mesh", "网格所有权为空");
        }
        if (diagnostics.hasErrors()) {
            return {{}, std::move(diagnostics)};
        }
        const auto& md = mesh->data();
        // 分区完整性不能由局部成员表证明，首期保守拒绝多分区/幽灵输入。
        bool partitioned = md.partitionCount_ > 1 || !md.ghostElements_.empty() || !md.ghostEntities_.empty();
        for (const auto& e : md.partitionedEntities_) {
            for (const auto p : e.partitions_) {
                partitioned = partitioned || p != 1;
            }
        }
        for (const auto& e : md.elementPartitions_) {
            partitioned = partitioned || !e.ghostPartitions_.empty() ||
                std::any_of(e.partitions_.begin(), e.partitions_.end(), [](int p) { return p != 1; });
        }
        if (partitioned) {
            issue(diagnostics, "UNSUPPORTED_PARTITION", "/mesh",
                  "首期无法证明多分区/幽灵输入构成完整唯一拥有的全域；请提供未分区完整网格");
            return {{}, std::move(diagnostics)};
        }
        PhysicalMembership membership;
        try {
            membership = buildPhysicalMembership(*mesh, limits.maxValues_);
        }
        catch (const std::length_error& e) {
            issue(diagnostics, "RESOURCE_LIMIT", "/mesh", e.what());
            return {{}, std::move(diagnostics)};
        }
        BindingData candidate;
        candidate.mesh_ = std::move(mesh);
        candidate.definition_ = definition;
        candidate.capability_.diagnostics_ = Diagnostics(limits.maxDiagnostics_);
        std::size_t selectedCount = 0;
        auto resolve = [&](const RegionSelector& region, const std::string& path, const std::string& id) {
            auto result = resolveRegion(region, membership, limits);
            for (auto value : result.diagnostics_.entries()) {
                value.path_ = path + value.path_;
                value.objectId_ = id;
                diagnostics.add(std::move(value));
            }
            if (result.cells_.size() > limits.maxValues_ - selectedCount) {
                issue(diagnostics, "RESOURCE_LIMIT", path, "全部绑定选择成员超过限额", id);
                result.cells_.clear();
            }
            else {
                selectedCount += result.cells_.size();
            }
            return result.cells_;
        };
        candidate.active_ = resolve(definition.analysis_.domain_, "/analysis/domain", {});
        if (diagnostics.hasErrors()) {
            return {{}, std::move(diagnostics)};
        }
        std::map<mesh::CellRef, std::size_t> activeIndices;
        for (std::size_t i = 0; i < candidate.active_.size(); ++i) {
            activeIndices.emplace(candidate.active_[i], i);
        }
        const auto noMaterial = std::numeric_limits<std::size_t>::max();
        candidate.materialIndices_.assign(candidate.active_.size(), noMaterial);
        std::map<std::string, std::size_t> materialIds;
        for (std::size_t i = 0; i < definition.materials_.size(); ++i) {
            materialIds.emplace(definition.materials_[i].id_, i);
        }
        std::set<std::size_t> usedMaterials;
        for (std::size_t i = 0; i < definition.materialAssignments_.size(); ++i) {
            const auto& assignment = definition.materialAssignments_[i];
            const auto path = "/materialAssignments/" + std::to_string(i);
            for (const auto cell : resolve(assignment.region_, path + "/region", assignment.id_)) {
                const auto it = activeIndices.find(cell);
                const auto tag = md.elements_[cell.block_].tags_[cell.element_];
                if (it == activeIndices.end()) {
                    issue(diagnostics, "OUTSIDE_DOMAIN", path, "材料区域包含域外单元", assignment.id_, tag);
                    continue;
                }
                auto& material = candidate.materialIndices_[it->second];
                if (material != noMaterial) {
                    issue(diagnostics, "MATERIAL_OVERLAP", path, "活动单元被重复分配材料", assignment.id_, tag);
                }
                else {
                    material = materialIds.at(assignment.material_);
                    usedMaterials.insert(material);
                }
            }
        }
        for (std::size_t i = 0; i < candidate.active_.size(); ++i) {
            if (candidate.materialIndices_[i] == noMaterial) {
                const auto cell = candidate.active_[i];
                issue(diagnostics, "MATERIAL_GAP", "/materialAssignments", "活动单元缺少材料", {},
                      md.elements_[cell.block_].tags_[cell.element_]);
            }
        }
        for (std::size_t i = 0; i < definition.materials_.size(); ++i) {
            if (!usedMaterials.contains(i)) {
                diagnostics.add({Severity::Warning,
                                 "UNUSED_MATERIAL",
                                 "/materials/" + std::to_string(i),
                                 definition.materials_[i].id_,
                                 {},
                                 "材料未被活动域使用",
                                 "删除无用材料或检查区域分配"});
            }
        }
        auto domainSelection = [&](const RegionSelector& region, const std::string& path, const std::string& id) {
            auto cells = resolve(region, path, id);
            for (const auto cell : cells) {
                if (!activeIndices.contains(cell)) {
                    issue(diagnostics, "OUTSIDE_DOMAIN", path, "选中活动域外单元", id,
                          md.elements_[cell.block_].tags_[cell.element_]);
                }
            }
            return cells;
        };
        for (std::size_t i = 0; i < definition.sources_.size(); ++i) {
            const auto& source = definition.sources_[i];
            candidate.sources_.push_back(
                {i, domainSelection(source.region_, "/sources/" + std::to_string(i) + "/region", source.id_), {}});
        }
        if (definition.initialState_) {
            std::set<mesh::CellRef> seen;
            for (std::size_t i = 0; i < definition.initialState_->overrides_.size(); ++i) {
                const auto& o = definition.initialState_->overrides_[i];
                const auto path = "/initialState/overrides/" + std::to_string(i);
                auto cells = domainSelection(o.region_, path + "/region", o.id_);
                for (const auto cell : cells) {
                    if (!seen.insert(cell).second) {
                        issue(diagnostics, "INITIAL_OVERLAP", path, "初始温度覆盖区域重叠", o.id_);
                    }
                }
                candidate.initialOverrides_.push_back({i, std::move(cells), {}});
            }
        }
        const int dimension = definition.analysis_.space_ == SpaceModel::Solid3d ? 3 : 2;
        std::set<mesh::Index> activeNodes;
        std::set<std::size_t> unsupportedBlocks;
        for (const auto cell : candidate.active_) {
            if (!firstOrderDomain(md.elements_[cell.block_].type_, dimension)) {
                unsupportedBlocks.insert(cell.block_);
            }
            const auto nodes = candidate.mesh_->connectivity(cell.block_, cell.element_);
            activeNodes.insert(nodes.begin(), nodes.end());
            if (std::set<mesh::Index>(nodes.begin(), nodes.end()).size() != nodes.size()) {
                issue(diagnostics, "DEGENERATE_TOPOLOGY", "/mesh", "单元含重复节点身份", {},
                      md.elements_[cell.block_].tags_[cell.element_]);
            }
        }
        for (const auto node : activeNodes) {
            auto position = md.nodes_[node].position_;
            for (auto& value : position) {
                value *= definition.mesh_.coordinateScale_;
            }
            if (!std::all_of(position.begin(), position.end(), [](double x) { return std::isfinite(x); })) {
                issue(diagnostics, "INVALID_GEOMETRY", "/mesh/coordinateScale", "米制坐标溢出", {},
                      md.nodes_[node].tag_);
            }
            if (dimension == 2 && std::abs(position[2]) > definition.analysis_.geometryTolerance_) {
                issue(diagnostics, "NONPLANAR_DOMAIN", "/analysis/space", "平面二维活动节点不位于全局XY平面", {},
                      md.nodes_[node].tag_);
            }
        }
        for (const auto block : unsupportedBlocks) {
            issue(candidate.capability_.diagnostics_, "UNSUPPORTED_ELEMENT", "/mesh",
                  "首个组装后端只计划支持完整一阶Triangle3/Tet4，当前块为" + std::to_string(block));
        }
        if (definition.discretization_.order_ != 1) {
            issue(candidate.capability_.diagnostics_, "UNSUPPORTED_DISCRETIZATION", "/discretization/order",
                  "首个计算后端只计划支持一阶温度空间");
        }
        if (diagnostics.hasErrors()) {
            return {{}, std::move(diagnostics)};
        }
        try {
            candidate.topology_ = mesh::buildFacetTopology(*candidate.mesh_, candidate.active_, limits.maxValues_);
        }
        catch (const std::length_error& e) {
            issue(diagnostics, "RESOURCE_LIMIT", "/mesh", e.what());
            return {{}, std::move(diagnostics)};
        }
        bool complete = candidate.topology_.unsupported_.empty();
        std::map<std::vector<mesh::Index>, std::size_t> facetIndices;
        for (std::size_t i = 0; i < candidate.topology_.facets_.size(); ++i) {
            const auto& facet = candidate.topology_.facets_[i];
            facetIndices.emplace(facet.nodes_, i);
            if (facet.uses_.size() > 2) {
                issue(diagnostics, "NONMANIFOLD_DOMAIN", "/mesh", "活动域某侧连接超过两个单元");
            }
        }
        std::map<std::size_t, std::vector<std::size_t>> facetConditions;
        for (std::size_t i = 0; i < definition.boundaries_.size(); ++i) {
            const auto& boundary = definition.boundaries_[i];
            const auto path = "/boundaries/" + std::to_string(i);
            BoundSelection selection{i, resolve(boundary.region_, path + "/region", boundary.id_), {}};
            std::set<std::size_t> unique;
            for (const auto cell : selection.cells_) {
                const auto& type = md.elements_[cell.block_].type_;
                if (!linearBoundary(type)) {
                    complete = false;
                    issue(candidate.capability_.diagnostics_, "UNSUPPORTED_BOUNDARY_LAYOUT", path,
                          "边界布局无法由首期线性侧目录可靠匹配", boundary.id_);
                    continue;
                }
                if (!candidate.topology_.unsupported_.empty()) {
                    continue;
                }
                const auto nodes = candidate.mesh_->connectivity(cell.block_, cell.element_);
                std::vector<mesh::Index> key(nodes.begin(), nodes.end());
                std::sort(key.begin(), key.end());
                const auto it = facetIndices.find(key);
                if (it == facetIndices.end() || candidate.topology_.facets_[it->second].uses_.size() != 1) {
                    issue(diagnostics, "BOUNDARY_NOT_EXTERIOR", path, "选择的低维单元不是活动域外侧", boundary.id_,
                          md.elements_[cell.block_].tags_[cell.element_]);
                    continue;
                }
                unique.insert(it->second);
            }
            selection.facets_.assign(unique.begin(), unique.end());
            for (const auto facet : selection.facets_) {
                auto& existing = facetConditions[facet];
                for (const auto previous : existing) {
                    const auto& a = definition.boundaries_[previous].value_;
                    const auto& b = boundary.value_;
                    const auto* ta = std::get_if<TemperatureBoundary>(&a);
                    const auto* tb = std::get_if<TemperatureBoundary>(&b);
                    if (ta && tb) {
                        if (evaluate(ta->temperature_) != evaluate(tb->temperature_)) {
                            issue(diagnostics, "TEMPERATURE_CONFLICT", path, "同侧温度条件不一致", boundary.id_);
                        }
                    }
                    else if (ta || tb || definition.naturalOverlap_ == NaturalOverlap::Error) {
                        issue(diagnostics, "BOUNDARY_OVERLAP", path, "同侧条件不允许叠加", boundary.id_);
                    }
                }
                existing.push_back(i);
            }
            candidate.boundaries_.push_back(std::move(selection));
        }
        if (complete) {
            for (std::size_t i = 0; i < candidate.topology_.facets_.size(); ++i) {
                if (candidate.topology_.facets_[i].uses_.size() == 1 && !facetConditions.contains(i)) {
                    candidate.insulatedFacets_.push_back(i);
                }
            }
            if (!candidate.insulatedFacets_.empty()) {
                diagnostics.add({Severity::Warning,
                                 "DEFAULT_ADIABATIC",
                                 "/boundaryPolicy",
                                 {},
                                 {},
                                 "未覆盖外侧按绝热处理，共" + std::to_string(candidate.insulatedFacets_.size()) + "侧",
                                 "核对未覆盖外侧是否符合物理问题"});
            }
        }
        else {
            candidate.capability_.pending_.push_back(
                {PendingKind::FacetTopology, "布局侧关联未完整，不能认证外边界覆盖"});
        }
        candidate.capability_.pending_.push_back({PendingKind::Geometry, "仍需Jacobian、积分点质量与外法向检查"});
        candidate.capability_.pending_.push_back(
            {PendingKind::DegreesOfFreedom, "仍需温度约束交汇、初温连续性及t0边界相容检查"});
        candidate.capability_.pending_.push_back(
            {PendingKind::Solvability, "仍需每连通分量零空间、锚定与求解器数学前提检查"});
        candidate.capability_.pending_.push_back(
            {PendingKind::Assembly, "尚未实现有限元空间、积分、组装与时间推进后端"});
        candidate.capability_.pending_.push_back({PendingKind::ResultOutput, "尚未实现VTU温度结果输出组件"});
        if (diagnostics.hasErrors()) {
            return {{}, std::move(diagnostics)};
        }
        std::shared_ptr<const BoundCase> bound(new BoundCase(std::move(candidate)));
        return {std::move(bound), std::move(diagnostics)};
    }
} // namespace emberfield::problem
