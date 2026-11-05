/** @file Preparation.cu
 * @brief 实现事务性阶段准备及常量配置到局部数值运算的适配。
 */
#include <NVTXProfiler.cuh>
#include <Simulation/Case/Preparation.cuh>
#include <Simulation/Case/Temporal.cuh>
#include <cmath>
#include "../Thermal/LocalResultAccess.cuh"
#include "ConditionSupport.cuh"
#include "ThermalSampling.cuh"
namespace emberfield::problem {
    PreparationResult prepareThermalCase(std::shared_ptr<const BoundCase> b,
                                         std::shared_ptr<const fem::FiniteElementSpace> s,
                                         const PreparationOptions& o) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::problem::prepareThermalCase", emberfield::NvtxColor::Yellow);
        validatePreparationOptions(o, s ? s->dimension() : 2);
        PreparationResult result(o.maxDiagnostics_);
        if (!b || !s) {
            detail::issue(result.diagnostics_, "PREPARATION_NULL_INPUT", "绑定或空间为空");
            return result;
        }
        if (!detail::context(*b, *s, result.diagnostics_)) {
            return result;
        }
        if (s->cellCount() > o.maxCells_ || s->dofCount() > o.maxDofs_) {
            detail::issue(result.diagnostics_, "PREPARATION_LIMIT", "空间超过单元或全局DOF预算");
            return result;
        }
        const auto& def = b->data().definition_;
        fem::GeometryOptions geometryOptions;
        geometryOptions.frame_ =
            def.analysis_.space_ == SpaceModel::Planar2d ? fem::GeometryFrame::PlanarXY : fem::GeometryFrame::SolidXYZ;
        geometryOptions.coordinateScale_ = def.mesh_.coordinateScale_;
        geometryOptions.planarTolerance_ = def.analysis_.geometryTolerance_;
        geometryOptions.detTolerance_ = o.detTolerance_;
        geometryOptions.maxCondition_ = o.maxCondition_;
        geometryOptions.normalTolerance_ = o.normalTolerance_;
        std::shared_ptr<const fem::PreparedGeometry> geometry;
        try {
            geometry = fem::buildPreparedGeometry(s, geometryOptions, o.maxCells_, o.maxFacetUses_);
            result.stages_[0] = StageStatus::Passed;
        }
        catch (const fem::EvaluationError& e) {
            result.stages_[0] = StageStatus::Failed;
            std::optional<std::uint64_t> tag;
            if (e.cell()) {
                tag = s->mesh().data().elements_.at(e.cell()->block_).tags_.at(e.cell()->element_);
            }
            detail::issue(result.diagnostics_, fem::evaluationCodeName(e.code()), e.what(), {}, tag);
        }
        // 当前元型已由context限定，固定二次目录和参考元仍单独确认，不冒充全域数值积分。
        for (std::size_t row = 0; row < s->cellCount(); ++row) {
            const auto& rule = fem::quadratureRule(s->elementKey(row).family_, 2);
            const auto& side =
                fem::quadratureRule(s->dimension() == 2 ? mesh::CellFamily::Segment : mesh::CellFamily::Triangle, 2);
            if (rule.exactDegree() < 2 || side.exactDegree() < 2) {
                throw std::logic_error("固定二次规则目录失效");
            }
        }
        result.stages_[1] = StageStatus::Passed;
        auto conditions = prepareConditions(b, s, o);
        result.diagnostics_.append(conditions.diagnostics());
        result.stages_[2] = conditions.ok() ? StageStatus::Passed : StageStatus::Failed;
        if (geometry && conditions.ok() && !result.diagnostics_.hasErrors()) {
            auto p = std::shared_ptr<PreparedThermalCase>(new PreparedThermalCase);
            p->geometry_ = std::move(geometry);
            p->conditions_ = conditions.conditions();
            result.prepared_ = std::move(p);
        }
        return result;
    }
    ThermalCellTerms evaluateCellTerms(const PreparedThermalCase& p, std::size_t row,
                                       const EvaluatedThermalState* state) {
        if (state && (!state->plan() || state->plan()->assemblyPlan()->prepared().get() != &p)) {
            throw std::invalid_argument("CPU体积分快照来源错配");
        }
        const auto t = fem::evaluateCellIntegration(p.geometry(), row, 2);
        const auto& def = p.bound()->data().definition_;
        const auto& c = *p.conditions();
        const auto& material = def.materials_.at(c.materialIndices().at(row));
        const auto q = t.pointCount();
        const double scale = detail::thermalSectionScale(p);
        std::vector<double> samples(q, evaluate(material.conductivity_));
        auto k = thermal::integrateConductivity(t, samples, scale);
        std::optional<thermal::LocalMatrix> m;
        if (def.analysis_.type_ == AnalysisType::Transient) {
            samples.assign(q, detail::thermalCapacity(material, p.space()->cells()[row]));
            m = thermal::integrateCapacity(t, samples, scale);
        }
        std::vector<double> load(t.dofCount(), 0);
        for (auto j = c.sourceOffsets().at(row); j < c.sourceOffsets().at(row + 1); ++j) {
            const auto& source = def.sources_.at(c.sourceIndices().at(j));
            samples.assign(q, state ? state->sources().at(c.sourceIndices().at(j)) : evaluate(source.value_));
            const auto term = thermal::integrateVolumeSource(t, samples, scale);
            for (std::size_t a = 0; a < load.size(); ++a) {
                load[a] += term[a];
                if (!std::isfinite(load[a])) {
                    throw fem::EvaluationError(fem::EvaluationCode::NumericalOverflow, fem::EvaluationStage::Element,
                                               "体源叠加溢出：" + source.id_, p.space()->cells()[row], {}, {}, a);
                }
            }
        }
        return ThermalCellTerms(row, std::move(k), std::move(m),
                                thermal::detail::LocalResultAccess::vector(t.dofCount(), std::move(load)));
    }
    std::variant<thermal::LocalVector, thermal::ConvectionContribution>
    evaluateBoundaryTerm(const PreparedThermalCase& p, std::size_t index, const EvaluatedThermalState* state) {
        if (state && (!state->plan() || state->plan()->assemblyPlan()->prepared().get() != &p)) {
            throw std::invalid_argument("CPU自然积分快照来源错配");
        }
        const auto& term = p.conditions()->naturalTerms().at(index);
        const auto t = fem::evaluateFacetIntegration(p.geometry(), term.facetIndex(), 2);
        if (t.use().cell_ != term.use().cell_ || t.use().localSide_ != term.use().localSide_ ||
            t.parentLocalDofs() != term.parentLocalDofs()) {
            throw std::logic_error("侧积分与条件关联顺序不一致");
        }
        const auto& value = p.bound()->data().definition_.boundaries_.at(term.definitionIndex()).value_;
        const double scale = detail::thermalSectionScale(p);
        if (const auto* flux = std::get_if<FluxBoundary>(&value)) {
            return thermal::integrateOutwardFlux(
                t,
                std::vector<double>(t.pointCount(),
                                    state ? state->primary().at(term.definitionIndex()) : evaluate(flux->flux_)),
                scale);
        }
        if (const auto* convection = std::get_if<ConvectionBoundary>(&value)) {
            return thermal::integrateConvection(t,
                                                std::vector<double>(t.pointCount(),
                                                                    state ? state->primary().at(term.definitionIndex())
                                                                          : evaluate(convection->coefficient_)),
                                                std::vector<double>(t.pointCount(),
                                                                    state ? state->ambient().at(term.definitionIndex())
                                                                          : evaluate(convection->ambientTemperature_)),
                                                scale);
        }
        throw std::logic_error("自然侧项包含规定温度");
    }
} // namespace emberfield::problem
