/** @file temporal_test.cpp @brief 时间函数端点、极端值与曲线全区间相容性测试。 */
#include <IO/Case/Case.cuh>
#include <IO/Case/Write.cuh>
#include <Simulation/Case/Postprocess.cuh>
#include <Simulation/Case/ScalarField.cuh>
#include <Simulation/Case/Temporal.cuh>
#include <Solver/Iterative/Cg.cuh>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include "../../../src/Simulation/Case/StepSystemCache.cuh"
#include "support/AssemblyFixture.cuh"
#include "support/CaseFixture.cuh"
#include "support/IntegrationFixture.cuh"
using namespace emberfield::problem;

TEST(TimeField, interpolationAndRightContinuousKnots) {
    ScalarField linear = TimeTableScalar{{{0, 300}, {2, 400}, {4, 200}}, TimeInterpolation::Linear};
    validateScalarField(linear);
    EXPECT_DOUBLE_EQ(evaluate(linear, 1), 350);
    EXPECT_DOUBLE_EQ(evaluate(linear, 2), 400);
    EXPECT_THROW((void)evaluate(linear), std::invalid_argument);
    EXPECT_THROW((void)evaluate(linear, -1), std::out_of_range);
    auto previous = std::get<TimeTableScalar>(linear);
    previous.interpolation_ = TimeInterpolation::Previous;
    previous.outside_ = TimeOutside::Hold;
    EXPECT_DOUBLE_EQ(evaluate(previous, std::nextafter(2.0, 0.0)), 300);
    EXPECT_DOUBLE_EQ(evaluate(previous, 2), 400);
    EXPECT_DOUBLE_EQ(evaluate(previous, -1), 300);
    EXPECT_DOUBLE_EQ(evaluate(previous, 9), 200);
}
TEST(TimeField, rejectsInvalidTablesAndFiniteTime) {
    EXPECT_THROW(validateScalarField(TimeTableScalar{}), std::invalid_argument);
    EXPECT_THROW(validateScalarField(TimeTableScalar{{{0, 1}, {0, 2}}}), std::invalid_argument);
    EXPECT_THROW(validateScalarField(TimeTableScalar{{{1, 1}, {0, 2}}}), std::invalid_argument);
    EXPECT_THROW((void)evaluate(ConstantScalar{3}, std::numeric_limits<double>::infinity()), std::invalid_argument);
    EXPECT_THROW(validateScalarField(ConstantScalar{std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
}

namespace {
    using namespace emberfield;
    using namespace assemblyfixture;
    /** @brief 构造时间字段算例。无参数无前提。@return 合法定义。@code auto c=timeCase(); @endcode */
    CaseDefinition timeCase() {
        auto c = casefixture::definition();
        c.mesh_.path_ = "/tmp/plate.msh";
        casefixture::makeTransient(c);
        c.time_ = TimeOptions{0, 2, .75, 100};
        std::get<TemperatureBoundary>(c.boundaries_[0].value_).temperature_ = TimeTableScalar{{{0, 300}, {2, 320}}};
        c.sources_ = {{"q", casefixture::group(2, 11),
                       TimeTableScalar{{{0, 10}, {.5, 20}, {2, 30}}, TimeInterpolation::Previous}}};
        c.boundaries_.push_back({"flux", casefixture::group(1, 21), FluxBoundary{TimeTableScalar{{{0, 1}, {2, 3}}}}});
        c.boundaries_.push_back(
            {"convection", casefixture::group(1, 22),
             ConvectionBoundary{TimeTableScalar{{{0, 2}, {1, 4}, {2, 4}}, TimeInterpolation::Previous},
                                TimeTableScalar{{{0, 300}, {2, 340}}}}});
        return c;
    }
    /** @brief 独立准备。@param c 合法算例。@pre 无。@return 准备上下文；失败异常。
     * @code auto p=prepareTime(c); @endcode */
    std::shared_ptr<const PreparedThermalCase> prepareTime(const CaseDefinition& c,
                                                           mesh::MeshData data = casefixture::meshData()) {
        auto b = casefixture::bind(c, std::move(data));
        if (!b.bound_) {
            throw std::runtime_error("时间算例绑定失败");
        }
        auto s = buildTemperatureSpace(*b.bound_);
        if (!s.space_) {
            throw std::runtime_error("时间算例空间失败");
        }
        auto p = prepareThermalCase(b.bound_, s.space_);
        if (!p.ok()) {
            throw std::runtime_error("时间算例准备失败");
        }
        return p.prepared();
    }
    /** @brief 替换测试定义中的物理字段。@param c 副本。@param time 时刻。@pre 表已验证。
     * @return 同刻常量参考。@code auto d=freeze(c,1); @endcode */
    CaseDefinition freeze(CaseDefinition c, double time) {
        for (auto& source : c.sources_) {
            source.value_ = ConstantScalar{evaluate(source.value_, time)};
        }
        for (auto& b : c.boundaries_) {
            if (auto* g = std::get_if<TemperatureBoundary>(&b.value_)) {
                g->temperature_ = ConstantScalar{evaluate(g->temperature_, time)};
            }
            else if (auto* q = std::get_if<FluxBoundary>(&b.value_)) {
                q->flux_ = ConstantScalar{evaluate(q->flux_, time)};
            }
            else {
                auto& h = std::get<ConvectionBoundary>(b.value_);
                h.coefficient_ = ConstantScalar{evaluate(h.coefficient_, time)};
                h.ambientTemperature_ = ConstantScalar{evaluate(h.ambientTemperature_, time)};
            }
        }
        // 参考模型只用于积分；保持独立初态与规定温度相容。
        c.initialState_->defaultTemperature_ = std::get<TemperatureBoundary>(c.boundaries_[0].value_).temperature_;
        return c;
    }
    TEST(TimeField, JsonRoundTripAndStrictPhysicalValidation) {
        const auto c = timeCase();
        std::istringstream input(io::serializeCase(c));
        auto r = io::readCase(input, "/tmp", "time.json");
        ASSERT_TRUE(r.definition_);
        EXPECT_EQ(std::get<TimeTableScalar>(r.definition_->sources_[0].value_).points_,
                  std::get<TimeTableScalar>(c.sources_[0].value_).points_);
        EXPECT_DOUBLE_EQ(evaluate(r.definition_->sources_[0].value_, .5), 20);
        auto invalid = c;
        invalid.analysis_.type_ = AnalysisType::Steady;
        EXPECT_TRUE(validateCase(invalid).hasErrors());
        invalid = c;
        invalid.materials_[0].conductivity_ = TimeTableScalar{{{0, 1}, {2, 2}}};
        EXPECT_TRUE(validateCase(invalid).hasErrors());
        invalid = c;
        std::get<TimeTableScalar>(invalid.sources_[0].value_).points_.back()[0] = 1;
        EXPECT_TRUE(casefixture::hasCode(validateCase(invalid), "TIME_TABLE_COVERAGE"));
        invalid = c;
        std::get<TimeTableScalar>(std::get<ConvectionBoundary>(invalid.boundaries_.back().value_).coefficient_)
            .points_[1][1] = -1;
        EXPECT_TRUE(validateCase(invalid).hasErrors());
        for (
            const std::string replacement :
            {R"({"kind":"timeTable","interpolation":"cubic","points":[{"time":0,"value":1},{"time":2,"value":2}]})",
             R"({"kind":"timeTable","interpolation":"linear","points":[{"time":0,"value":1},{"time":0,"value":2}]})",
             R"({"kind":"timeTable","interpolation":"previous","outside":"linear","points":[{"time":0,"value":1}]})"}) {
            auto text = io::serializeCase(c);
            const auto start = text.find("\"conductivity\"");
            const auto open = text.find('{', start), close = text.find('}', open);
            text.replace(open, close - open + 1, replacement);
            std::istringstream stream(text);
            EXPECT_FALSE(io::readCase(stream, "/tmp").definition_);
        }
    }
    TEST(TimeField, RejectsDirichletCurvesThatOnlyAgreeInitially) {
        auto c = timeCase();
        c.boundaries_.push_back({"top", casefixture::group(1, 23), TemperatureBoundary{ConstantScalar{300}}});
        auto bound = casefixture::bind(c);
        if (bound.bound_) {
            auto space = buildTemperatureSpace(*bound.bound_);
            auto p = prepareThermalCase(bound.bound_, space.space_);
            EXPECT_FALSE(p.ok());
            EXPECT_TRUE(casefixture::hasCode(p.diagnostics(), "DIRICHLET_CONFLICT"));
        }
        else {
            EXPECT_TRUE(bound.diagnostics_.hasErrors());
        }
    }
    TEST(TimeField, CursorPreservesNominalGridAndCountsInsertedEvents) {
        auto c = timeCase();
        auto plan = buildThermalEvolutionPlan(prepareTime(c));
        ThermalTimeCursor cursor(plan);
        std::vector<double> ends;
        while (auto step = cursor.peek()) {
            ends.push_back(step->end_);
            cursor.accept(*step);
        }
        EXPECT_EQ(ends, (std::vector<double>{.5, .75, 1, 1.5, 2}));
        EXPECT_EQ(cursor.plannedSteps(), ends.size());
        EXPECT_FALSE(cursor.peek());
        c.time_->maxSteps_ = 4;
        auto limited = buildThermalEvolutionPlan(prepareTime(c));
        EXPECT_THROW(ThermalTimeCursor{limited}, std::length_error);
    }
    TEST(TimeField, CursorHandlesNegativeTimeHoldTightBudgetsAndFloatingStagnation) {
        auto c = timeCase();
        c.boundaries_.clear();
        c.time_ = TimeOptions{-2, 0, .75, 5};
        c.sources_[0].value_ = TimeTableScalar{{{-1.5, 10}, {-1, 20}}, TimeInterpolation::Previous, TimeOutside::Hold};
        auto plan = buildThermalEvolutionPlan(prepareTime(c));
        ThermalTimeCursor cursor(plan);
        std::vector<double> ends;
        while (auto step = cursor.peek()) {
            ends.push_back(step->end_);
            cursor.accept(*step);
        }
        EXPECT_EQ(ends, (std::vector<double>{-1.5, -1.25, -1, -.5, 0}));
        c.sources_.clear();
        c.time_ = TimeOptions{1e16, 1e16 + 100, .1, 2000};
        EXPECT_TRUE(validateCase(c).hasErrors());
        EXPECT_FALSE(casefixture::bind(c).bound_);
        c.time_ = TimeOptions{0, .3, .1, 3};
        auto exact = buildThermalEvolutionPlan(prepareTime(c));
        EXPECT_EQ(ThermalTimeCursor(exact).plannedSteps(), 3);
    }
    TEST_F(GpuAssembly, TemporalSnapshotsMatchConstantizedFullAssemblyAndCacheDependencies) {
        const auto c = timeCase();
        auto plan = buildThermalEvolutionPlan(prepareTime(c));
        ThermalEvolutionContext context(plan);
        auto first = context.assemble(evaluateThermalState(plan, 0));
        const auto firstS = dense(first.model().spatial()), firstF = host(first.model().load().values());
        for (double time : {.5, .75, 1., 1.5, 2.}) {
            auto state = evaluateThermalState(plan, time);
            auto numeric = context.assemble(state);
            auto reference = assembleThermalModel(buildThermalAssemblyPlan(prepareTime(freeze(c, time))));
            near(dense(numeric.model().spatial()), dense(reference.spatial()));
            near(dense(*numeric.model().capacity()), dense(*reference.capacity()));
            near(host(numeric.model().load().values()), host(reference.load().values()));
            EXPECT_EQ(&*numeric.model().capacity(), &*first.model().capacity());
            EXPECT_EQ(numeric.matrixRevision(), time < 1 ? 1 : 2);
            EXPECT_EQ(host(numeric.model().spatial().rowOffsets()), host(first.model().spatial().rowOffsets()));
            EXPECT_EQ(host(numeric.model().spatial().columnIndices()), host(first.model().spatial().columnIndices()));
            if (time < 1) {
                EXPECT_EQ(&numeric.model().spatial(), &first.model().spatial());
            }
            auto again = context.assemble(state);
            EXPECT_EQ(&again.model().load(), &numeric.model().load());
        }
        near(dense(first.model().spatial()), firstS);
        near(host(first.model().load().values()), firstF);
        auto other = buildThermalEvolutionPlan(prepareTime(c));
        EXPECT_THROW((void)context.assemble(evaluateThermalState(other, 1)), std::invalid_argument);
        EXPECT_THROW((void)evaluateThermalState(plan, 3), std::invalid_argument);
        EXPECT_THROW((void)assembleThermalModel(plan->assemblyPlan()), std::invalid_argument);
    }
    TEST_F(GpuAssembly, TemporalDirichletRetainsOldConsistentMassHistory) {
        auto c = timeCase();
        auto plan = buildThermalEvolutionPlan(prepareTime(c));
        ThermalEvolutionContext context(plan);
        auto old = evaluateThermalState(plan, 0), current = evaluateThermalState(plan, .5);
        auto numeric = context.assemble(current);
        Vector<double> previous{
            std::span<const double>(*plan->assemblyPlan()->prepared()->conditions()->initialTemperature())};
        TimeStep step{1, 0, .5, .5};
        auto system = buildThermalStepSystem(numeric, previous, old, step);
        auto a = dense(numeric.model().spatial()), mass = dense(*numeric.model().capacity());
        auto rhs = host(numeric.model().load().values()), history = host(previous.values());
        const auto n = history.size();
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                a[i * n + j] += mass[i * n + j] / .5;
                rhs[i] += mass[i * n + j] * history[j] / .5;
            }
        }
        const auto& dofs = plan->assemblyPlan()->prepared()->conditions()->constraints().dofs();
        auto original = a;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t cIndex = 0; cIndex < dofs.size(); ++cIndex) {
                rhs[i] -= original[i * n + dofs[cIndex]] * current.constraints()[cIndex];
            }
        }
        for (std::size_t cIndex = 0; cIndex < dofs.size(); ++cIndex) {
            const auto d = dofs[cIndex];
            for (std::size_t i = 0; i < n; ++i) {
                a[d * n + i] = a[i * n + d] = 0;
            }
            a[d * n + d] = 1;
            rhs[d] = current.constraints()[cIndex];
        }
        near(dense(system.matrix()), a);
        near(host(system.rhs().values()), rhs);
        EXPECT_THROW((void)buildThermalStepSystem(numeric, previous, current, step), std::invalid_argument);
        auto projected = systems::projectTemperatureState(previous, dofs, current.constraints());
        EXPECT_THROW((void)buildThermalStepSystem(numeric, projected, old, step), std::invalid_argument);
        ThermalPostprocessPlan post(plan->assemblyPlan()->prepared());
        auto otherNumeric = context.assemble(evaluateThermalState(plan, .75));
        EXPECT_THROW((void)post.evaluate(numeric.model(), projected, &previous, .5, 1, .5, false, true,
                                         &otherNumeric.parameters(), &old),
                     std::invalid_argument);
    }
    TEST_F(GpuAssembly, TemporalTetrahedronMatchesIndependentCpuIntegrationAndNaturalPower) {
        auto data = integrationfixture::data(3);
        data.elements_[0].physicalTags_ = std::vector<std::int64_t>{11};
        mesh::ElementBlock side;
        side.type_ = {mesh::CellFamily::Triangle, 2, 3, 1, "complete"};
        side.tags_ = {201};
        side.offsets_ = {0, 3};
        side.nodes_ = {0, 1, 2};
        side.physicalTags_ = std::vector<std::int64_t>{21};
        data.elements_.push_back(side);
        auto c = timeCase();
        c.analysis_.space_ = SpaceModel::Solid3d;
        c.analysis_.thickness_.reset();
        c.analysis_.domain_ = casefixture::group(3, 11);
        c.materialAssignments_[0].region_ = casefixture::group(3, 11);
        c.sources_[0].region_ = casefixture::group(3, 11);
        c.boundaries_.clear();
        c.naturalOverlap_ = NaturalOverlap::Add;
        c.output_.energy_.enabled_ = true;
        c.boundaries_.push_back({"flux", casefixture::group(2, 21), FluxBoundary{TimeTableScalar{{{0, 1}, {2, 3}}}}});
        c.boundaries_.push_back(
            {"a", casefixture::group(2, 21),
             ConvectionBoundary{TimeTableScalar{{{0, 0}, {1, 4}, {2, 0}}}, TimeTableScalar{{{0, 300}, {2, 340}}}}});
        c.boundaries_.push_back(
            {"b", casefixture::group(2, 21), ConvectionBoundary{ConstantScalar{2}, ConstantScalar{280}}});
        auto prepared = prepareTime(c, data);
        auto plan = buildThermalEvolutionPlan(prepared);
        ThermalEvolutionContext context(plan);
        ThermalPostprocessPlan post(prepared);
        for (double time : {0., .5, 1., 2., .5}) {
            auto state = evaluateThermalState(plan, time);
            auto numeric = context.assemble(state);
            auto term = evaluateCellTerms(*prepared, 0, &state);
            auto s = term.conductivity().entries(), m = term.capacity()->entries(), f = term.load().entries();
            const auto cellDofs = prepared->space()->cellDofs(0);
            for (std::size_t row = 0; row < prepared->conditions()->naturalTerms().size(); ++row) {
                auto boundary = evaluateBoundaryTerm(*prepared, row, &state);
                const auto& dofs = prepared->conditions()->naturalTerms()[row].globalDofs();
                for (std::size_t a = 0; a < dofs.size(); ++a) {
                    const auto i = std::find(cellDofs.begin(), cellDofs.end(), dofs[a]) - cellDofs.begin();
                    if (auto* q = std::get_if<thermal::LocalVector>(&boundary)) {
                        f[i] += (*q)[a];
                    }
                    else {
                        const auto& h = std::get<thermal::ConvectionContribution>(boundary);
                        f[i] += h.load()[a];
                        for (std::size_t b = 0; b < dofs.size(); ++b) {
                            const auto j = std::find(cellDofs.begin(), cellDofs.end(), dofs[b]) - cellDofs.begin();
                            s[i * 4 + j] += h.matrix().entries()[a * dofs.size() + b];
                        }
                    }
                }
            }
            const auto actual = dense(numeric.model().spatial()), actualM = dense(*numeric.model().capacity());
            const auto actualF = host(numeric.model().load().values());
            for (std::size_t i = 0; i < 4; ++i) {
                EXPECT_NEAR(actualF[cellDofs[i]], f[i], 1e-10);
                for (std::size_t j = 0; j < 4; ++j) {
                    EXPECT_NEAR(actual[cellDofs[i] * 4 + cellDofs[j]], s[i * 4 + j], 1e-10);
                    EXPECT_NEAR(actualM[cellDofs[i] * 4 + cellDofs[j]], m[i * 4 + j], 1e-9);
                }
            }
            if (time > 0) {
                Vector<double> t{std::span<const double>(std::vector<double>(4, 310))};
                auto old = evaluateThermalState(plan, 0);
                auto energy = post.evaluate(numeric, t, &t, &old, time, 1).energy_;
                EXPECT_NEAR(*energy.prescribedFluxOutW_, .5 * state.primary()[0], 1e-10);
                EXPECT_NEAR(*energy.convectionOutW_,
                            .5 * (state.primary()[1] * (310 - state.ambient()[1]) + 2 * (310 - 280)), 1e-10);
            }
        }
    }
    TEST_F(GpuAssembly, TemporalUniformHeatingMatchesDiscreteAnalyticSolutionAndEnergy) {
        auto c = timeCase();
        c.boundaries_.clear();
        c.time_ = TimeOptions{0, 2, .75, 100};
        c.sources_[0].value_ = TimeTableScalar{{{0, 0}, {.5, 3900000}, {2, 15600000}}};
        c.output_.energy_.enabled_ = true;
        auto prepared = prepareTime(c);
        auto plan = buildThermalEvolutionPlan(prepared);
        ThermalEvolutionContext context(plan);
        ThermalTimeCursor cursor(plan);
        ThermalPostprocessPlan postprocess(prepared);
        auto old = evaluateThermalState(plan, 0);
        Vector<double> temperature{std::span<const double>(*prepared->conditions()->initialTemperature())};
        double expected = 300;
        while (auto step = cursor.peek()) {
            auto current = evaluateThermalState(plan, step->end_);
            auto numeric = context.assemble(current);
            auto system = buildThermalStepSystem(numeric, temperature, old, *step);
            CgOptions<double> options;
            options.relativeTolerance_ = 1e-12;
            Cg<double, assembly::Index> cg(options);
            cg.setup(system.matrix());
            auto next = temperature;
            ASSERT_EQ(cg.apply(system.rhs(), next).status_, SolveStatus::Converged);
            expected += 2 * step->end_ * step->duration_;
            for (double value : host(next.values())) {
                EXPECT_NEAR(value, expected, 1e-9);
            }
            const auto fields = postprocess.evaluate(numeric, next, &temperature, &old, step->duration_, step->index_);
            ASSERT_TRUE(fields.energy_.sourcePowerW_);
            EXPECT_NEAR(*fields.energy_.sourcePowerW_, current.sources()[0] * .01, 1e-6);
            EXPECT_NEAR(*fields.energy_.storagePowerW_, *fields.energy_.sourcePowerW_, 1e-6);
            EXPECT_THROW((void)postprocess.evaluate(numeric.model(), next, &temperature, step->duration_, step->index_,
                                                    step->end_),
                         std::invalid_argument);
            temperature = std::move(next);
            old = std::move(current);
            cursor.accept(*step);
        }
    }
    TEST_F(GpuAssembly, TemporalPrescribedTemperatureOnlyReusesModelAndAllConstrainedSystemIsIdentity) {
        auto c = timeCase();
        c.sources_.clear();
        c.boundaries_.clear();
        c.output_.energy_.enabled_ = true;
        for (int tag = 21; tag <= 24; ++tag) {
            c.boundaries_.push_back({"side" + std::to_string(tag), casefixture::group(1, tag),
                                     TemperatureBoundary{TimeTableScalar{{{0, 300}, {2, 320}}}}});
        }
        auto prepared = prepareTime(c);
        auto plan = buildThermalEvolutionPlan(prepared);
        ThermalEvolutionContext context(plan);
        ThermalPostprocessPlan post(prepared);
        auto initial = context.assemble(evaluateThermalState(plan, 0));
        auto current = context.assemble(evaluateThermalState(plan, 1));
        EXPECT_EQ(&initial.model().spatial(), &current.model().spatial());
        EXPECT_EQ(&initial.model().load(), &current.model().load());
        EXPECT_EQ(initial.matrixRevision(), current.matrixRevision());
        Vector<double> previous{std::span<const double>(std::vector<double>(4, 300))};
        auto system = buildThermalStepSystem(current, previous, initial.parameters(), TimeStep{1, 0, 1, 1});
        near(dense(system.matrix()), {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
        near(host(system.rhs().values()), {310, 310, 310, 310});
        Vector<double> next{std::span<const double>(std::vector<double>(4, 310))};
        auto report = post.evaluate(current, next, &previous, &initial.parameters(), 1, 1).energy_;
        EXPECT_NEAR(*report.storagePowerW_, 390000, 1e-6);
        EXPECT_NEAR(*report.dirichletOutW_, -390000, 1e-6);
        auto initialReport = post.evaluate(initial, previous, nullptr, nullptr, {}, 0, true).energy_;
        EXPECT_TRUE(initialReport.sourcePowerW_);
        EXPECT_FALSE(initialReport.storagePowerW_);
        EXPECT_FALSE(initialReport.dirichletOutW_);
    }
    TEST_F(GpuAssembly, TemporalPreviousSourceUsesNewEndpointAndLinearSourceConvergesFirstOrder) {
        auto c = timeCase();
        c.boundaries_.clear();
        c.time_ = TimeOptions{0, 2, 1, 100};
        c.sources_[0].value_ = TimeTableScalar{{{0, 0}, {1, 3900000}, {2, 0}}, TimeInterpolation::Previous};
        const auto solve = [&](const CaseDefinition& definition) {
            auto prepared = prepareTime(definition);
            auto plan = buildThermalEvolutionPlan(prepared);
            ThermalEvolutionContext context(plan);
            ThermalTimeCursor cursor(plan);
            auto old = evaluateThermalState(plan, 0);
            Vector<double> temperature{std::span<const double>(*prepared->conditions()->initialTemperature())};
            while (auto step = cursor.peek()) {
                auto state = evaluateThermalState(plan, step->end_);
                auto numeric = context.assemble(state);
                auto system = buildThermalStepSystem(numeric, temperature, old, *step);
                CgOptions<double> options;
                options.relativeTolerance_ = 1e-13;
                Cg<double, assembly::Index> solver(options);
                solver.setup(system.matrix());
                if (solver.apply(system.rhs(), temperature).status_ != SolveStatus::Converged) {
                    throw std::runtime_error("独立时间收敛测试求解失败");
                }
                old = std::move(state);
                cursor.accept(*step);
            }
            return host(temperature.values())[0];
        };
        EXPECT_NEAR(solve(c), 301, 1e-9);
        c.sources_[0].value_ = TimeTableScalar{{{0, 0}, {2, 15600000}}};
        const auto coarse = solve(c);
        c.time_->step_ = .5;
        const auto fine = solve(c);
        EXPECT_NEAR(coarse, 306, 1e-9);
        EXPECT_NEAR(fine, 305, 1e-9);
        EXPECT_NEAR((coarse - 304) / (fine - 304), 2, 1e-8);
    }
} // namespace
TEST(TimeField, extremeInterpolationAndWholeCurveEquality) {
    const double m = std::numeric_limits<double>::max();
    ScalarField f = TimeTableScalar{{{-m, -m}, {m, m}}};
    EXPECT_DOUBLE_EQ(evaluate(f, 0), 0);
    ScalarField a = TimeTableScalar{{{0, 300}, {2, 400}}};
    ScalarField b = TimeTableScalar{{{0, 300}, {1, 350}, {2, 400}}};
    EXPECT_TRUE(equivalentFields(a, b, 0, 2));
    ScalarField c = TimeTableScalar{{{0, 300}, {1, 300}, {2, 400}}};
    EXPECT_FALSE(equivalentFields(a, c, 0, 2));
}

TEST_F(GpuAssembly, CachedStepSystemTracksEventsHChangesAndSourceIdentity) {
    auto prepared = prepareTime(timeCase());
    EXPECT_EQ(&prepared->geometry()->topology(), &prepared->bound()->data().topology_);
    auto plan = buildThermalEvolutionPlan(prepared);
    ThermalEvolutionContext evolution(plan);
    emberfield::problem::detail::ThermalStepSystemCache cache;
    auto old = evaluateThermalState(plan, 0);
    Vector<double> previous{std::span<const double>(*prepared->conditions()->initialTemperature())};
    const auto& dofs = prepared->conditions()->constraints().dofs();
    previous = systems::projectTemperatureState(previous, dofs, old.constraints());
    // Same dt with changing h/load/g, then event shortened interval and a repeated dt.
    for (auto [start, end] : {std::pair{0., .5}, std::pair{.5, 1.}, std::pair{1., 1.25}, std::pair{1.25, 1.5}}) {
        auto state = evaluateThermalState(plan, end);
        auto numeric = evolution.assemble(state);
        TimeStep step{1, start, end, end - start};
        auto expected = buildThermalStepSystem(numeric, previous, old, step);
        auto actual = cache.build(numeric, previous, old, step);
        EXPECT_EQ(host(actual.matrix().values()), host(expected.matrix().values()));
        EXPECT_EQ(host(actual.rhs().values()), host(expected.rhs().values()));
        previous = systems::projectTemperatureState(previous, dofs, state.constraints());
        old = state;
    }
}
