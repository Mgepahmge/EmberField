/** @file Write.cu @brief 独立的类型化算例JSON输出适配。 */
#define PICOJSON_USE_INT64
#include <IO/Case/Write.cuh>
#include <Simulation/Case/Validation.cuh>
#include <picojson.h>
namespace emberfield::io {
    namespace {
        using J = picojson::value;
        using O = picojson::object;
        using A = picojson::array;
        using namespace problem;
        /** @brief 整数直接序列化，避免double。@param n 无符号整数。@pre int64可表示。@return JSON整数。@code auto
         * v=count(2); @endcode */
        J count(std::uint64_t n) {
            if (n > INT64_MAX) {
                throw std::length_error("JSON整数超出int64");
            }
            return J(static_cast<std::int64_t>(n));
        }
        /** @brief 标量场回显。@param s 常量场。@pre 有效。@return 场对象。@code auto j=scalar(s); @endcode */
        J scalar(const ScalarField& s) { return J(O{{"kind", J("constant")}, {"value", J(evaluate(s))}}); }
        /** @brief 选择器递归回显。@param r 有限深度树。@pre 有效。@return 选择器JSON。@code auto j=region(r); @endcode
         */
        J region(const RegionSelector& r) {
            O o;
            if (r.kind_ == RegionKind::PhysicalGroup) {
                o["kind"] = J("physicalGroup");
                o["dimension"] = J(static_cast<std::int64_t>(r.dimension_));
                if (r.tag_) {
                    o["tag"] = J(*r.tag_);
                }
                if (r.name_) {
                    o["name"] = J(*r.name_);
                }
            }
            else if (r.kind_ == RegionKind::Difference) {
                o["kind"] = J("difference");
                o["base"] = region(r.operands_[0]);
                o["remove"] = region(r.operands_[1]);
            }
            else {
                o["kind"] = J(r.kind_ == RegionKind::Union ? "union" : "intersection");
                A a;
                for (const auto& x : r.operands_) {
                    a.push_back(region(x));
                }
                o["items"] = J(a);
            }
            return J(o);
        }
    } // namespace
    std::string serializeCase(const problem::CaseDefinition& c) {
        if (validateCase(c).hasErrors()) {
            throw std::invalid_argument("拒绝序列化非法算例");
        }
        O root{{"schemaVersion", count(c.schemaVersion_)}, {"name", J(c.name_)}, {"description", J(c.description_)}};
        root["mesh"] = J(O{{"path", J(c.mesh_.path_.generic_string())},
                           {"format", J(c.mesh_.format_)},
                           {"coordinateScale", J(c.mesh_.coordinateScale_)}});
        O analysis{{"type", J(c.analysis_.type_ == AnalysisType::Steady ? "steady" : "transient")},
                   {"space", J(c.analysis_.space_ == SpaceModel::Planar2d ? "planar2d" : "solid3d")},
                   {"domain", region(c.analysis_.domain_)},
                   {"geometryTolerance", J(c.analysis_.geometryTolerance_)}};
        if (c.analysis_.thickness_) {
            analysis["thickness"] = J(*c.analysis_.thickness_);
        }
        root["analysis"] = J(analysis);
        A materials, assignments, sources, boundaries;
        for (const auto& m : c.materials_) {
            O o{{"id", J(m.id_)}, {"displayName", J(m.displayName_)}, {"conductivity", scalar(m.conductivity_)}};
            if (m.density_) {
                o["density"] = scalar(*m.density_);
            }
            if (m.specificHeat_) {
                o["specificHeat"] = scalar(*m.specificHeat_);
            }
            if (m.volumetricHeatCapacity_) {
                o["volumetricHeatCapacity"] = scalar(*m.volumetricHeatCapacity_);
            }
            materials.emplace_back(o);
        }
        for (const auto& a : c.materialAssignments_) {
            assignments.emplace_back(O{{"id", J(a.id_)}, {"region", region(a.region_)}, {"material", J(a.material_)}});
        }
        for (const auto& s : c.sources_) {
            sources.emplace_back(O{{"id", J(s.id_)}, {"region", region(s.region_)}, {"value", scalar(s.value_)}});
        }
        for (const auto& b : c.boundaries_) {
            O o{{"id", J(b.id_)}, {"region", region(b.region_)}};
            if (const auto* t = std::get_if<TemperatureBoundary>(&b.value_)) {
                o["kind"] = J("temperature");
                o["temperature"] = scalar(t->temperature_);
            }
            else if (const auto* f = std::get_if<FluxBoundary>(&b.value_)) {
                o["kind"] = J("flux");
                o["flux"] = scalar(f->flux_);
            }
            else {
                const auto& h = std::get<ConvectionBoundary>(b.value_);
                o["kind"] = J("convection");
                o["coefficient"] = scalar(h.coefficient_);
                o["ambientTemperature"] = scalar(h.ambientTemperature_);
            }
            boundaries.emplace_back(o);
        }
        root["materials"] = J(materials);
        root["materialAssignments"] = J(assignments);
        root["sources"] = J(sources);
        root["boundaries"] = J(boundaries);
        root["boundaryPolicy"] =
            J(O{{"naturalOverlap", J(c.naturalOverlap_ == NaturalOverlap::Add ? "add" : "error")}});
        if (c.initialState_) {
            A overrides;
            for (const auto& x : c.initialState_->overrides_) {
                overrides.emplace_back(
                    O{{"id", J(x.id_)}, {"region", region(x.region_)}, {"temperature", scalar(x.temperature_)}});
            }
            root["initialState"] =
                J(O{{"defaultTemperature", scalar(c.initialState_->defaultTemperature_)}, {"overrides", J(overrides)}});
        }
        if (c.time_) {
            const auto& t = *c.time_;
            root["time"] = J(O{{"method", J("backwardEuler")},
                               {"start", J(t.start_)},
                               {"end", J(t.end_)},
                               {"step", J(t.step_)},
                               {"maxSteps", count(t.maxSteps_)}});
        }
        root["discretization"] = J(O{{"family", J("lagrangeContinuous")},
                                     {"order", count(c.discretization_.order_)},
                                     {"quadrature", J("auto")}});
        const auto& l = c.linearSolver_;
        root["linearSolver"] =
            J(O{{"method", J(l.method_ == LinearMethod::Cg ? "cg" : "bicgstab")},
                {"preconditioner", J(O{{"kind", J("identity")}})},
                {"relativeTolerance", J(l.relativeTolerance_)},
                {"absoluteTolerance", J(l.absoluteTolerance_)},
                {"maxIterations", count(l.maxIterations_)},
                {"initialGuess", J(l.initialGuess_ == InitialGuess::Zero ? "zero" : "previousStep")}});
        const auto& o = c.output_;
        const auto& b = o.limits_;
        A fields, monitors;
        for (const auto& f : o.fields_) {
            fields.emplace_back(std::string(outputFieldName(f)));
        }
        for (const auto& m : o.monitors_) {
            O q{{"id", J(m.id_)}};
            if (const auto* n = std::get_if<NodeMonitor>(&m.value_)) {
                q["kind"] = J("node");
                q["nodeTag"] = J(std::to_string(n->nodeTag_));
            }
            else if (const auto* p = std::get_if<PointMonitor>(&m.value_)) {
                q["kind"] = J("point");
                A pos;
                for (double x : p->positionM_) {
                    pos.emplace_back(x);
                }
                q["positionM"] = J(pos);
                q["toleranceM"] = J(p->toleranceM_);
            }
            else {
                const auto& r = std::get<RegionMonitor>(m.value_);
                q["kind"] = J("region");
                q["region"] = region(r.region_);
                A s;
                for (const auto& f : r.statistics_) {
                    s.emplace_back(std::string(regionStatisticName(f)));
                }
                q["statistics"] = J(s);
            }
            monitors.emplace_back(q);
        }
        O output{{"directory", J(o.directory_.generic_string())},
                 {"format", J("vtu")},
                 {"encoding", J(o.encoding_ == OutputEncoding::Ascii ? "ascii" : "appendedRaw")},
                 {"fields", J(fields)},
                 {"overwrite", J(o.overwrite_)},
                 {"includeFinal", J(o.includeFinal_)},
                 {"monitors", J(monitors)},
                 {"energy",
                  J(O{{"enabled", J(o.energy_.enabled_)},
                      {"absoluteToleranceW", J(o.energy_.absoluteToleranceW_)},
                      {"relativeTolerance", J(o.energy_.relativeTolerance_)}})},
                 {"limits",
                  J(O{{"maxPoints", count(b.maxPoints_)},
                      {"maxCells", count(b.maxCells_)},
                      {"maxHostBytes", count(b.maxHostBytes_)},
                      {"maxDeviceBytes", count(b.maxDeviceBytes_)},
                      {"batchCells", count(b.batchCells_)},
                      {"maxFrames", count(b.maxFrames_)},
                      {"maxMonitors", count(b.maxMonitors_)},
                      {"maxOutputBytes", count(b.maxOutputBytes_)}})}};
        if (c.analysis_.type_ == AnalysisType::Transient) {
            output["includeInitial"] = J(o.includeInitial_.value_or(true));
            output["schedule"] = J(O{{"kind", J("everySteps")}, {"interval", count(o.everySteps_.value_or(1))}});
        }
        root["output"] = J(output);
        return J(root).serialize(true);
    }
} // namespace emberfield::io
