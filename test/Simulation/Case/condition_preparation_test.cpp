/** @file condition_preparation_test.cpp
 * @brief 验证真实绑定条件的共享节点冲突、阶段报告和局部配置适配。
 */
#include <Simulation/Case/Preparation.cuh>
#include <cmath>
#include <gtest/gtest.h>
#include <numeric>
#include "support/CaseFixture.cuh"
namespace {
    using namespace emberfield;
    using namespace problem;
    using namespace casefixture;
    /** @brief 创建联合准备结果。@param c 定义。@param m 网格。@param o 策略。
     * @pre 定义静态合法并能绑定。@return 准备结果。@code auto r=prepare(definition()); @endcode */
    PreparationResult prepare(const CaseDefinition& c, mesh::MeshData m = meshData(), PreparationOptions o = {}) {
        auto bound = casefixture::bind(c, std::move(m));
        if (!bound.bound_) {
            throw std::logic_error("无效算例测试夹具绑定");
        }
        auto space = buildTemperatureSpace(*bound.bound_);
        if (!space.space_) {
            throw std::logic_error("无效算例测试夹具空间");
        }
        return prepareThermalCase(bound.bound_, space.space_, o);
    }
    TEST(Conditions, SteadyRetainsFullSpaceAndSourceMappings) {
        auto c = definition();
        c.sources_ = {{"one", group(2, 11), ConstantScalar{10}}, {"two", group(2, 11), ConstantScalar{20}}};
        auto r = prepare(c);
        ASSERT_TRUE(r.ok());
        const auto& p = *r.prepared();
        const auto& pc = *p.conditions();
        EXPECT_EQ(pc.sourceOffsets(), (std::vector<std::size_t>{0, 2, 4}));
        EXPECT_EQ(pc.sourceIndices(), (std::vector<std::size_t>{0, 1, 0, 1}));
        EXPECT_EQ(pc.constraints().dofs(), (std::vector<fem::DofIndex>{0, 3}));
        EXPECT_EQ(pc.space()->dofCount(), 4);
        EXPECT_FALSE(pc.initialTemperature());
        auto terms = evaluateCellTerms(p, 0);
        EXPECT_FALSE(terms.capacity());
        EXPECT_NEAR(std::accumulate(terms.load().entries().begin(), terms.load().entries().end(), 0.0), 0.15, 1e-14);
        EXPECT_EQ(p.bound()->data().capability_.pending_.size(), pc.bound()->data().capability_.pending_.size());
    }
    TEST(Conditions, CornerAgreementMergesAndRetainsBothSources) {
        auto c = definition();
        c.boundaries_.push_back({"bottom", group(1, 21), TemperatureBoundary{ConstantScalar{300}}});
        auto r = prepare(c);
        ASSERT_TRUE(r.ok());
        const auto& d = r.prepared()->conditions()->constraints();
        EXPECT_EQ(d.dofs().size(), 3);
        EXPECT_EQ(d.sourceOffsets()[1] - d.sourceOffsets()[0], 2);
        EXPECT_EQ(d.sourceIndices()[0], 0);
        EXPECT_EQ(d.sourceIndices()[1], 1);
    }
    TEST(Conditions, CornerConflictReportsBothIdsAndPersistentNodeTag) {
        auto c = definition();
        c.boundaries_.push_back({"bottom", group(1, 21), TemperatureBoundary{ConstantScalar{400}}});
        auto r = prepare(c);
        EXPECT_FALSE(r.ok());
        EXPECT_FALSE(r.prepared());
        EXPECT_TRUE(hasCode(r.diagnostics(), "DIRICHLET_CONFLICT"));
        const auto& d = r.diagnostics().entries().front();
        EXPECT_EQ(d.meshTag_, 5000000001ULL);
        EXPECT_NE(d.message_.find("left"), std::string::npos);
        EXPECT_NE(d.message_.find("bottom"), std::string::npos);
        EXPECT_EQ(r.stage(PreparationStage::Geometry), StageStatus::Passed);
        EXPECT_EQ(r.stage(PreparationStage::Conditions), StageStatus::Failed);
    }
    TEST(Conditions, ExactComparisonDoesNotSilentlyApplyTolerance) {
        auto c = definition();
        c.boundaries_.push_back(
            {"bottom", group(1, 21), TemperatureBoundary{ConstantScalar{std::nextafter(300.0, 301.0)}}});
        auto r = prepare(c);
        EXPECT_TRUE(hasCode(r.diagnostics(), "DIRICHLET_CONFLICT"));
    }
    TEST(Conditions, NaturalAtPrescribedCornerIsRetainedAndFluxIsNegative) {
        auto c = definition();
        c.boundaries_.push_back({"bottom", group(1, 21), FluxBoundary{ConstantScalar{10}}});
        auto r = prepare(c);
        ASSERT_TRUE(r.ok());
        const auto& terms = r.prepared()->conditions()->naturalTerms();
        ASSERT_EQ(terms.size(), 1);
        EXPECT_EQ(terms[0].globalDofs(), (std::vector<fem::DofIndex>{0, 1}));
        const auto value = evaluateBoundaryTerm(*r.prepared(), 0);
        ASSERT_TRUE(std::holds_alternative<thermal::LocalVector>(value));
        const auto& f = std::get<thermal::LocalVector>(value);
        EXPECT_NEAR(f[0] + f[1], -0.1, 1e-14);
    }
    TEST(Conditions, AdditiveConvectionKeepsProductsPaired) {
        auto c = definition();
        c.naturalOverlap_ = NaturalOverlap::Add;
        c.boundaries_.push_back({"a", group(1, 21), ConvectionBoundary{ConstantScalar{2}, ConstantScalar{300}}});
        c.boundaries_.push_back({"b", group(1, 21), ConvectionBoundary{ConstantScalar{3}, ConstantScalar{400}}});
        auto r = prepare(c);
        ASSERT_TRUE(r.ok());
        ASSERT_EQ(r.prepared()->conditions()->naturalTerms().size(), 2);
        double load = 0;
        for (std::size_t i = 0; i < 2; ++i) {
            const auto value = evaluateBoundaryTerm(*r.prepared(), i);
            const auto& f = std::get<thermal::ConvectionContribution>(value).load();
            load += std::accumulate(f.entries().begin(), f.entries().end(), 0.0);
        }
        EXPECT_NEAR(load, 18, 1e-12);
    }
    TEST(Conditions, TransientHasConsistentInitialStateAndCapacity) {
        auto c = definition();
        makeTransient(c);
        auto r = prepare(c);
        ASSERT_TRUE(r.ok());
        const auto& initial = r.prepared()->conditions()->initialTemperature();
        ASSERT_TRUE(initial);
        EXPECT_EQ(*initial, (std::vector<double>{300, 300, 300, 300}));
        auto t = evaluateCellTerms(*r.prepared(), 0);
        ASSERT_TRUE(t.capacity());
        EXPECT_NEAR(std::accumulate(t.capacity()->entries().begin(), t.capacity()->entries().end(), 0.0), 19500, 1e-8);
    }
    TEST(Conditions, InitialVsDirichletMismatchFailsWithoutChangingState) {
        auto c = definition();
        makeTransient(c);
        c.initialState_->defaultTemperature_ = ConstantScalar{301};
        auto r = prepare(c);
        EXPECT_FALSE(r.ok());
        EXPECT_TRUE(hasCode(r.diagnostics(), "INITIAL_DIRICHLET_CONFLICT"));
        EXPECT_DOUBLE_EQ(evaluate(c.initialState_->defaultTemperature_), 301);
    }
    TEST(Conditions, DisjointCellOverridesStillConflictAtSharedDofs) {
        auto m = meshData();
        auto second = m.elements_[0];
        second.tags_ = {300};
        second.offsets_ = {0, 3};
        second.nodes_ = {0, 2, 3};
        second.physicalTags_ = std::vector<std::int64_t>{11, 72};
        m.elements_[0].tags_ = {100};
        m.elements_[0].offsets_ = {0, 3};
        m.elements_[0].nodes_ = {0, 1, 2};
        m.elements_[0].physicalTags_ = std::vector<std::int64_t>{11, 71};
        m.elements_.push_back(second);
        auto c = definition();
        makeTransient(c);
        c.initialState_->overrides_.push_back({"hot", group(2, 71), ConstantScalar{400}});
        auto r = prepare(c, m);
        EXPECT_FALSE(r.ok());
        EXPECT_TRUE(hasCode(r.diagnostics(), "INITIAL_DOF_CONFLICT"));
    }
    TEST(Conditions, EmptyAndAllConstrainedSpacesAreBothLegal) {
        auto c = definition();
        c.boundaries_.clear();
        auto r = prepare(c);
        ASSERT_TRUE(r.ok());
        EXPECT_TRUE(r.prepared()->conditions()->constraints().dofs().empty());
        for (int tag : {21, 22, 23, 24}) {
            c.boundaries_.push_back({std::to_string(tag), group(1, tag), TemperatureBoundary{ConstantScalar{300}}});
        }
        r = prepare(c);
        ASSERT_TRUE(r.ok());
        EXPECT_EQ(r.prepared()->conditions()->constraints().dofs().size(), 4);
        EXPECT_FALSE(r.prepared()->bound()->data().capability_.executable());
    }
    TEST(Conditions, AnotherIdenticalMeshIsNotTheSameContext) {
        auto c = definition();
        auto a = casefixture::bind(c), b = casefixture::bind(c);
        auto s = buildTemperatureSpace(*b.bound_);
        auto r = prepareThermalCase(a.bound_, s.space_);
        EXPECT_FALSE(r.ok());
        EXPECT_TRUE(hasCode(r.diagnostics(), "PREPARATION_CONTEXT_MISMATCH"));
        EXPECT_EQ(r.stage(PreparationStage::Geometry), StageStatus::NotRun);
    }
    TEST(Conditions, ActiveRowReorderingIsRejectedBeforeUse) {
        auto c = definition();
        auto b = casefixture::bind(c);
        auto active = b.bound_->data().active_;
        std::reverse(active.begin(), active.end());
        std::vector<fem::ElementKey> keys(active.size());
        auto s = fem::buildFiniteElementSpace(b.bound_->data().mesh_, active, keys);
        ASSERT_TRUE(s.space_);
        auto r = prepareThermalCase(b.bound_, s.space_);
        EXPECT_TRUE(hasCode(r.diagnostics(), "PREPARATION_CONTEXT_MISMATCH"));
    }
    TEST(Preparation, GeometryFailureDoesNotSuppressIndependentConditionErrors) {
        auto c = definition();
        c.boundaries_.push_back({"bottom", group(1, 21), TemperatureBoundary{ConstantScalar{400}}});
        auto m = meshData();
        m.nodes_[2].position_ = {2, 0, 0};
        auto r = prepare(c, m);
        EXPECT_FALSE(r.ok());
        EXPECT_TRUE(hasCode(r.diagnostics(), "DEGENERATE_CELL"));
        EXPECT_TRUE(hasCode(r.diagnostics(), "DIRICHLET_CONFLICT"));
        EXPECT_EQ(r.stage(PreparationStage::Geometry), StageStatus::Failed);
        EXPECT_EQ(r.stage(PreparationStage::Conditions), StageStatus::Failed);
    }
    TEST(Preparation, DiagnosticsTruncationPreservesFailure) {
        auto c = definition();
        makeTransient(c);
        c.initialState_->defaultTemperature_ = ConstantScalar{301};
        PreparationOptions o;
        o.maxDiagnostics_ = 1;
        auto r = prepare(c, meshData(), o);
        EXPECT_FALSE(r.ok());
        EXPECT_TRUE(r.diagnostics().hasErrors());
        EXPECT_EQ(r.diagnostics().entries().size(), 1);
        EXPECT_GT(r.diagnostics().omitted(), 0);
    }
    TEST(Preparation, LimitsHavePredictableBoundariesAndBadOptionsThrow) {
        auto c = definition();
        PreparationOptions o;
        o.maxAssociations_ = 4; // E=2，温度侧2次。
        EXPECT_TRUE(prepare(c, meshData(), o).ok());
        o.maxAssociations_ = 3;
        auto r = prepare(c, meshData(), o);
        EXPECT_TRUE(hasCode(r.diagnostics(), "PREPARATION_LIMIT"));
        o.maxCells_ = 1;
        r = prepare(c, meshData(), o);
        EXPECT_TRUE(hasCode(r.diagnostics(), "PREPARATION_LIMIT"));
        o.maxDofs_ = 0;
        EXPECT_THROW((void)prepare(c, meshData(), o), std::invalid_argument);
        auto empty = prepareThermalCase({}, {});
        EXPECT_TRUE(hasCode(empty.diagnostics(), "PREPARATION_NULL_INPUT"));
    }
    TEST(Preparation, LocalRangeFailureIsNotPretendedToHaveBeenCheckedByPreparation) {
        auto c = definition();
        makeTransient(c);
        c.materials_[0].volumetricHeatCapacity_ = ConstantScalar{1e308};
        c.analysis_.thickness_ = 1e308;
        auto r = prepare(c);
        ASSERT_TRUE(r.ok());
        EXPECT_THROW((void)evaluateCellTerms(*r.prepared(), 0), fem::EvaluationError);
    }
    TEST(Preparation, ScaledCoordinatesProduceSameLocalContributions) {
        auto c = definition();
        auto a = prepare(c);
        ASSERT_TRUE(a.ok());
        c.mesh_.coordinateScale_ = 1e-3;
        auto m = meshData();
        for (auto& n : m.nodes_) {
            for (auto& x : n.position_) {
                x *= 1000;
            }
        }
        auto b = prepare(c, m);
        ASSERT_TRUE(b.ok());
        const auto x = evaluateCellTerms(*a.prepared(), 0), y = evaluateCellTerms(*b.prepared(), 0);
        for (std::size_t i = 0; i < x.conductivity().entries().size(); ++i) {
            EXPECT_NEAR(x.conductivity().entries()[i], y.conductivity().entries()[i], 1e-13);
        }
    }
    TEST(Preparation, ResultsOwnTheirContextAndBoundsAreChecked) {
        auto c = definition();
        auto r = prepare(c);
        ASSERT_TRUE(r.ok());
        EXPECT_EQ(r.prepared()->geometry()->space().get(), r.prepared()->conditions()->space().get());
        EXPECT_THROW((void)evaluateCellTerms(*r.prepared(), 99), std::out_of_range);
        EXPECT_THROW((void)evaluateBoundaryTerm(*r.prepared(), 99), std::out_of_range);
        EXPECT_THROW((void)r.stage(static_cast<PreparationStage>(99)), std::out_of_range);
    }
} // namespace
