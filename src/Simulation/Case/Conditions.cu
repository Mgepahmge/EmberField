/** @file Conditions.cu
 * @brief 实现共享温度冲突、物理初态及组装前条件关联。
 */
#include <Simulation/Case/Conditions.cuh>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include "ConditionSupport.cuh"
namespace emberfield::problem {
    void validatePreparationOptions(const PreparationOptions& o, unsigned d) {
        fem::GeometryOptions g;
        g.frame_ = d == 2 ? fem::GeometryFrame::PlanarXY : fem::GeometryFrame::SolidXYZ;
        g.detTolerance_ = o.detTolerance_;
        g.maxCondition_ = o.maxCondition_;
        g.normalTolerance_ = o.normalTolerance_;
        fem::validateGeometryOptions(g, d);
        if (!o.maxCells_ || !o.maxDofs_ || !o.maxFacetUses_ || !o.maxAssociations_ || !o.maxDiagnostics_) {
            throw std::invalid_argument("所有准备计数限额必须为正");
        }
    }
    std::optional<DirichletConstraints> buildDirichletConstraints(const BoundCase& b, const fem::FiniteElementSpace& s,
                                                                  std::size_t limit, Diagnostics& diagnostics) {
        if (!limit) {
            throw std::invalid_argument("温度关联限额必须为正");
        }
        if (!detail::context(b, s, diagnostics)) {
            return {};
        }
        const auto& data = b.data();
        struct Proposal {
            fem::DofIndex dof_; ///< 全局编号。
            double value_; ///< 规范化K值。
            std::size_t definition_; ///< 来源定义。
            mesh::FacetUse use_; ///< 来源侧。
        };
        std::size_t visits = 0;
        // 展开前预算校验，不能先创建无界提案数组。
        for (const auto& selection : data.boundaries_) {
            if (!std::holds_alternative<TemperatureBoundary>(
                    data.definition_.boundaries_.at(selection.definitionIndex_).value_)) {
                continue;
            }
            for (auto facet : selection.facets_) {
                const auto& f = data.topology_.facets_.at(facet);
                if (f.uses_.size() != 1) {
                    detail::issue(diagnostics, "BOUNDARY_NOT_EXTERIOR", "温度边界不是唯一外侧");
                    return {};
                }
                const auto row = s.cellPosition(f.uses_[0].cell_);
                if (!row) {
                    detail::issue(diagnostics, "BOUNDARY_INACTIVE", "温度边界母单元未激活");
                    return {};
                }
                const auto count =
                    fem::referenceElement(s.elementKey(*row)).facetLocalDofs(f.uses_[0].localSide_).size();
                if (!detail::consume(visits, count, limit, diagnostics)) {
                    return {};
                }
            }
        }
        std::vector<Proposal> proposals;
        proposals.reserve(visits);
        std::set<std::pair<std::size_t, std::size_t>> unique;
        for (const auto& selection : data.boundaries_) {
            const auto& boundary = data.definition_.boundaries_.at(selection.definitionIndex_);
            const auto* prescribed = std::get_if<TemperatureBoundary>(&boundary.value_);
            if (!prescribed) {
                continue;
            }
            const double value = detail::temperature(prescribed->temperature_);
            for (auto facet : selection.facets_) {
                if (!unique.emplace(selection.definitionIndex_, facet).second) {
                    continue;
                }
                const auto use = data.topology_.facets_.at(facet).uses_.at(0);
                for (auto dof : fem::facetDofs(s, use)) {
                    proposals.push_back({dof, value, selection.definitionIndex_, use});
                }
            }
        }
        std::sort(proposals.begin(), proposals.end(), [](const Proposal& a, const Proposal& b) {
            return std::tie(a.dof_, a.definition_, a.use_.cell_, a.use_.localSide_) <
                std::tie(b.dof_, b.definition_, b.use_.cell_, b.use_.localSide_);
        });
        DirichletConstraints result;
        bool failed = false;
        for (std::size_t begin = 0; begin < proposals.size();) {
            auto end = begin + 1;
            while (end < proposals.size() && proposals[end].dof_ == proposals[begin].dof_) {
                ++end;
            }
            const auto& first = proposals[begin];
            result.dofs_.push_back(first.dof_);
            result.temperatures_.push_back(first.value_);
            std::optional<std::size_t> last;
            for (auto i = begin; i < end; ++i) {
                const auto& p = proposals[i];
                if (p.value_ != first.value_) {
                    failed = true;
                    const auto& a = data.definition_.boundaries_[first.definition_].id_;
                    const auto& id = data.definition_.boundaries_[p.definition_].id_;
                    std::ostringstream text;
                    text.precision(17);
                    text << "DOF " << p.dof_ << " 的规定温度冲突：" << a << "=" << first.value_ << " 与 " << id << "="
                         << p.value_ << " K；局部侧 " << first.use_.localSide_ << " / " << p.use_.localSide_;
                    detail::issue(diagnostics, "DIRICHLET_CONFLICT", text.str(), id, detail::dofTag(s, p.dof_));
                }
                if (!last || *last != p.definition_) {
                    result.sourceIndices_.push_back(p.definition_);
                    last = p.definition_;
                }
            }
            result.sourceOffsets_.push_back(result.sourceIndices_.size());
            begin = end;
        }
        if (failed) {
            return {};
        }
        return result;
    }
    std::optional<std::vector<double>> buildInitialTemperature(const BoundCase& b, const fem::FiniteElementSpace& s,
                                                               const DirichletConstraints& constraints,
                                                               std::size_t limit, Diagnostics& diagnostics) {
        if (!limit) {
            throw std::invalid_argument("初温关联上限必须为正");
        }
        if (!detail::context(b, s, diagnostics)) {
            return {};
        }
        const auto& data = b.data();
        const auto& def = data.definition_;
        if (def.analysis_.type_ != AnalysisType::Transient || !def.initialState_) {
            detail::issue(diagnostics, "INITIAL_STATE_REQUIRED", "初温插值仅适用于有物理初态的瞬态算例");
            return {};
        }
        std::size_t visits = 0;
        for (std::size_t row = 0; row < s.cellCount(); ++row) {
            if (!detail::consume(visits, s.cellDofs(row).size(), limit, diagnostics)) {
                return {};
            }
        }
        std::vector<std::optional<std::size_t>> override(s.cellCount());
        for (const auto& selection : data.initialOverrides_) {
            for (auto cell : selection.cells_) {
                const auto row = s.cellPosition(cell);
                if (!row || override[*row]) {
                    detail::issue(diagnostics, "INITIAL_OVERRIDE_CONFLICT", "初温覆盖重叠或引用非活动单元");
                    return {};
                }
                override[*row] = selection.definitionIndex_;
            }
        }
        std::vector<double> temperature(s.dofCount());
        std::vector<std::optional<std::size_t>> firstCell(s.dofCount());
        bool failed = false;
        const auto source = [&](std::size_t row) {
            return override[row] ? def.initialState_->overrides_.at(*override[row]).id_ : std::string("default");
        };
        for (std::size_t row = 0; row < s.cellCount(); ++row) {
            const auto& field = override[row] ? def.initialState_->overrides_.at(*override[row]).temperature_
                                              : def.initialState_->defaultTemperature_;
            const double value = detail::temperature(field);
            for (auto dof : s.cellDofs(row)) {
                if (dof >= s.dofCount() || dof > std::numeric_limits<std::size_t>::max()) {
                    throw std::logic_error("空间DOF超出宿主索引范围");
                }
                const auto index = static_cast<std::size_t>(dof);
                if (!firstCell[index]) {
                    firstCell[index] = row;
                    temperature[index] = value;
                }
                else if (temperature[index] != value) {
                    failed = true;
                    std::ostringstream text;
                    text.precision(17);
                    const auto first = *firstCell[index];
                    text << "DOF " << dof << " 初温冲突：cellRow " << first << "(" << source(first)
                         << ")=" << temperature[index] << " 与 " << row << "(" << source(row) << ")=" << value << " K";
                    detail::issue(diagnostics, "INITIAL_DOF_CONFLICT", text.str(), source(row), detail::dofTag(s, dof));
                }
            }
        }
        for (std::size_t i = 0; i < temperature.size(); ++i) {
            if (!firstCell[i]) {
                failed = true;
                detail::issue(diagnostics, "INITIAL_DOF_GAP", "初态未覆盖某个空间DOF", {}, detail::dofTag(s, i));
            }
        }
        for (std::size_t i = 0; i < constraints.dofs().size(); ++i) {
            const auto dof = constraints.dofs()[i];
            if (dof >= temperature.size() || dof > std::numeric_limits<std::size_t>::max()) {
                detail::issue(diagnostics, "PREPARATION_CONTEXT_MISMATCH", "初温约束的DOF不属于该空间");
                return {};
            }
            if (temperature[static_cast<std::size_t>(dof)] != constraints.temperatures()[i]) {
                failed = true;
                const auto sourceIndex = constraints.sourceIndices().at(constraints.sourceOffsets().at(i));
                const auto id = def.boundaries_.at(sourceIndex).id_;
                std::ostringstream text;
                text.precision(17);
                text << "DOF " << dof << " 的初温 " << temperature[static_cast<std::size_t>(dof)] << " K 与边界 " << id
                     << " 的t0温度 " << constraints.temperatures()[i] << " K 不一致";
                detail::issue(diagnostics, "INITIAL_DIRICHLET_CONFLICT", text.str(), id, detail::dofTag(s, dof));
            }
        }
        if (failed) {
            return {};
        }
        return temperature;
    }
    ConditionPreparationResult prepareConditions(std::shared_ptr<const BoundCase> b,
                                                 std::shared_ptr<const fem::FiniteElementSpace> s,
                                                 const PreparationOptions& o) {
        validatePreparationOptions(o, s ? s->dimension() : 2);
        ConditionPreparationResult result(o.maxDiagnostics_);
        auto& diagnostics = result.diagnostics_;
        if (!b || !s) {
            detail::issue(diagnostics, "PREPARATION_NULL_INPUT", "绑定或空间为空");
            return result;
        }
        if (!detail::context(*b, *s, diagnostics)) {
            return result;
        }
        if (s->cellCount() > o.maxCells_ || s->dofCount() > o.maxDofs_) {
            detail::issue(diagnostics, "PREPARATION_LIMIT", "活动单元或全局DOF超过限额");
            return result;
        }
        std::size_t uses = 0;
        for (std::size_t row = 0; row < s->cellCount(); ++row) {
            if (!detail::consume(uses, fem::referenceElement(s->elementKey(row)).facetCount(), o.maxFacetUses_,
                                 diagnostics)) {
                return result;
            }
        }
        const auto& data = b->data();
        const auto& def = data.definition_;
        // 独立拓扑与绑定拓扑逐项比较，避免把两个同长度侧表的索引互换使用。
        const auto topology = mesh::buildFacetTopology(s->mesh(), s->cells(), o.maxFacetUses_);
        if (!topology.unsupported_.empty() || topology.facets_.size() != data.topology_.facets_.size()) {
            detail::issue(diagnostics, "PREPARATION_TOPOLOGY_MISMATCH", "绑定与空间活动侧拓扑不一致");
            return result;
        }
        for (std::size_t i = 0; i < topology.facets_.size(); ++i) {
            const auto& a = topology.facets_[i];
            const auto& c = data.topology_.facets_[i];
            bool equal = a.nodes_ == c.nodes_ && a.uses_.size() == c.uses_.size();
            for (std::size_t j = 0; equal && j < a.uses_.size(); ++j) {
                equal = a.uses_[j].cell_ == c.uses_[j].cell_ && a.uses_[j].localSide_ == c.uses_[j].localSide_;
            }
            if (!equal || a.uses_.empty() || a.uses_.size() > 2) {
                detail::issue(diagnostics, "PREPARATION_TOPOLOGY_MISMATCH", "侧身份、母侧或流形性不一致");
                return result;
            }
        }
        std::size_t visits = 0;
        if (!detail::consume(visits, s->cellCount(), o.maxAssociations_, diagnostics)) {
            return result;
        }
        for (const auto& selection : data.sources_) {
            if (!detail::consume(visits, selection.cells_.size(), o.maxAssociations_, diagnostics)) {
                return result;
            }
        }
        for (const auto& selection : data.boundaries_) {
            for (auto i : selection.facets_) {
                const auto& f = topology.facets_.at(i);
                if (f.uses_.size() != 1) {
                    detail::issue(diagnostics, "BOUNDARY_NOT_EXTERIOR", "物理边界非活动域唯一外侧");
                    return result;
                }
                if (!detail::consume(visits, f.nodes_.size(), o.maxAssociations_, diagnostics)) {
                    return result;
                }
            }
        }
        if (def.analysis_.type_ == AnalysisType::Transient) {
            for (std::size_t row = 0; row < s->cellCount(); ++row) {
                if (!detail::consume(visits, s->cellDofs(row).size(), o.maxAssociations_, diagnostics)) {
                    return result;
                }
            }
        }
        auto p = std::shared_ptr<PreparedConditions>(new PreparedConditions);
        p->bound_ = std::move(b);
        p->space_ = std::move(s);
        p->materialIndices_ = data.materialIndices_;
        if (p->materialIndices_.size() != p->space_->cellCount()) {
            throw std::logic_error("绑定材料行数不一致");
        }
        for (auto i : p->materialIndices_) {
            if (i >= def.materials_.size()) {
                throw std::logic_error("绑定材料索引越界");
            }
        }
        std::vector<std::vector<std::size_t>> sources(p->space_->cellCount());
        for (const auto& selection : data.sources_) {
            for (auto cell : selection.cells_) {
                const auto row = p->space_->cellPosition(cell);
                if (!row) {
                    detail::issue(diagnostics, "SOURCE_INACTIVE", "源引用非活动单元");
                    return result;
                }
                sources[*row].push_back(selection.definitionIndex_);
            }
        }
        p->sourceOffsets_.push_back(0);
        for (auto& row : sources) {
            std::sort(row.begin(), row.end());
            row.erase(std::unique(row.begin(), row.end()), row.end());
            for (auto index : row) {
                if (index >= def.sources_.size()) {
                    throw std::logic_error("源定义索引越界");
                }
            }
            p->sourceIndices_.insert(p->sourceIndices_.end(), row.begin(), row.end());
            p->sourceOffsets_.push_back(p->sourceIndices_.size());
        }
        std::vector<bool> prescribed(topology.facets_.size(), false);
        std::vector<std::size_t> naturalCount(topology.facets_.size(), 0);
        std::set<std::pair<std::size_t, std::size_t>> unique;
        for (const auto& selection : data.boundaries_) {
            const auto& boundary = def.boundaries_.at(selection.definitionIndex_);
            const bool temperature = std::holds_alternative<TemperatureBoundary>(boundary.value_);
            for (auto index : selection.facets_) {
                if (!unique.emplace(index, selection.definitionIndex_).second) {
                    continue;
                }
                if (temperature) {
                    prescribed[index] = true;
                    continue;
                }
                ++naturalCount[index];
                NaturalBoundaryTerm term;
                term.definitionIndex_ = selection.definitionIndex_;
                term.facetIndex_ = index;
                term.use_ = topology.facets_[index].uses_[0];
                const auto row = p->space_->cellPosition(term.use_.cell_);
                const auto local =
                    fem::referenceElement(p->space_->elementKey(*row)).facetLocalDofs(term.use_.localSide_);
                term.parentLocalDofs_.assign(local.begin(), local.end());
                term.globalDofs_ = fem::facetDofs(*p->space_, term.use_);
                p->naturalTerms_.push_back(std::move(term));
            }
        }
        bool failed = false;
        for (std::size_t i = 0; i < prescribed.size(); ++i) {
            if ((prescribed[i] && naturalCount[i]) ||
                (naturalCount[i] > 1 && def.naturalOverlap_ == NaturalOverlap::Error)) {
                failed = true;
                detail::issue(diagnostics, "BOUNDARY_OVERLAP", "同侧温度与自然条件冲突，或自然条件未允许叠加");
            }
        }
        std::sort(p->naturalTerms_.begin(), p->naturalTerms_.end(),
                  [](const NaturalBoundaryTerm& a, const NaturalBoundaryTerm& c) {
                      return std::pair{a.facetIndex(), a.definitionIndex()} <
                          std::pair{c.facetIndex(), c.definitionIndex()};
                  });
        auto constraints = buildDirichletConstraints(*p->bound_, *p->space_, o.maxAssociations_, diagnostics);
        if (!constraints) {
            failed = true;
        }
        else {
            p->constraints_ = std::move(*constraints);
            if (def.analysis_.type_ == AnalysisType::Transient) {
                auto initial =
                    buildInitialTemperature(*p->bound_, *p->space_, p->constraints_, o.maxAssociations_, diagnostics);
                if (!initial) {
                    failed = true;
                }
                else {
                    p->initialTemperature_ = std::move(*initial);
                }
            }
        }
        if (!failed && !diagnostics.hasErrors()) {
            result.conditions_ = std::move(p);
        }
        return result;
    }
} // namespace emberfield::problem
