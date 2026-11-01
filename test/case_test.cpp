/** @file case_test.cpp
 * @brief 验证独立算例语义、区域选择、侧拓扑与事务绑定。
 */
#include <Case/Region.cuh>
#include <gtest/gtest.h>
#include <limits>
#include "support/CaseFixture.cuh"
namespace {
    using namespace thermofem;
    using namespace problem;
    using namespace casefixture;
    TEST(CaseDefinition, SupportsProgrammaticSteadyAndTransientWithoutFiles) {
        auto c = definition();
        EXPECT_FALSE(validateCase(c).hasErrors());
        makeTransient(c);
        EXPECT_FALSE(validateCase(c).hasErrors());
        c.materials_[0].volumetricHeatCapacity_.reset();
        c.materials_[0].density_ = ConstantScalar{7800};
        c.materials_[0].specificHeat_ = ConstantScalar{500};
        EXPECT_FALSE(validateCase(c).hasErrors());
    }
    TEST(CaseDefinition, AggregatesErrorsAndRetainsFailureAfterTruncation) {
        auto c = definition();
        c.analysis_.thickness_ = -1;
        c.materials_[0].conductivity_ = ConstantScalar{-1};
        c.linearSolver_.maxIterations_ = 0;
        auto report = validateCase(c);
        EXPECT_GE(report.entries().size(), 3);
        CaseLimits limits;
        limits.maxDiagnostics_ = 1;
        auto shortReport = validateCase(c, limits);
        EXPECT_TRUE(shortReport.hasErrors());
        EXPECT_EQ(shortReport.entries().size(), 1);
        EXPECT_GE(shortReport.omitted(), 2);
        Diagnostics d(1);
        d.add({Severity::Warning, "WARNING"});
        d.add({Severity::Error, "ERROR"});
        EXPECT_TRUE(d.hasErrors());
        Diagnostics merged(1);
        merged.append(d);
        EXPECT_TRUE(merged.hasErrors());
        EXPECT_GE(merged.omitted(), 1);
    }
    TEST(CaseDefinition, RejectsNonfiniteAndPhysicalRanges) {
        for (const double value :
             {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            auto c = definition();
            c.materials_[0].conductivity_ = ConstantScalar{value};
            EXPECT_TRUE(validateCase(c).hasErrors());
            c = definition();
            c.mesh_.coordinateScale_ = value;
            EXPECT_TRUE(validateCase(c).hasErrors());
        }
        auto c = definition();
        c.boundaries_[0].value_ = TemperatureBoundary{ConstantScalar{-0.1}};
        EXPECT_TRUE(validateCase(c).hasErrors());
        c.boundaries_[0].value_ = ConvectionBoundary{ConstantScalar{-1}, ConstantScalar{300}};
        EXPECT_TRUE(validateCase(c).hasErrors());
    }
    TEST(CaseDefinition, RejectsCapacityAmbiguityOverflowAndModeMismatch) {
        auto c = definition();
        c.materials_[0].density_ = ConstantScalar{1};
        EXPECT_TRUE(validateCase(c).hasErrors());
        c.materials_[0].specificHeat_ = ConstantScalar{1};
        EXPECT_FALSE(validateCase(c).hasErrors());
        c.materials_[0].volumetricHeatCapacity_ = ConstantScalar{1};
        EXPECT_TRUE(validateCase(c).hasErrors());
        c = definition();
        makeTransient(c);
        c.materials_[0].volumetricHeatCapacity_.reset();
        EXPECT_TRUE(validateCase(c).hasErrors());
        c.materials_[0].density_ = ConstantScalar{1e300};
        c.materials_[0].specificHeat_ = ConstantScalar{1e300};
        EXPECT_TRUE(validateCase(c).hasErrors());
        c = definition();
        c.time_ = TimeOptions{0, 1, 0.1, 100};
        EXPECT_TRUE(validateCase(c).hasErrors());
    }
    TEST(CaseDefinition, RejectsIdsReferencesAndSelectorShape) {
        auto c = definition();
        c.materials_.push_back(c.materials_[0]);
        EXPECT_TRUE(hasCode(validateCase(c), "DUPLICATE_ID"));
        c = definition();
        c.materialAssignments_[0].material_ = "missing";
        EXPECT_TRUE(hasCode(validateCase(c), "UNKNOWN_MATERIAL"));
        c = definition();
        c.analysis_.domain_.name_ = "body";
        EXPECT_TRUE(hasCode(validateCase(c), "INVALID_SELECTOR"));
        c = definition();
        c.analysis_.domain_.dimension_ = 1;
        EXPECT_TRUE(validateCase(c).hasErrors());
    }
    TEST(CaseDefinition, ValidatesTimePrecisionBudgetsAndOutputModes) {
        auto c = definition();
        makeTransient(c);
        c.time_->maxSteps_ = 2;
        EXPECT_TRUE(hasCode(validateCase(c), "RESOURCE_LIMIT"));
        c.time_ = TimeOptions{1e20, 1e20 + 1e6, 1, 10000000};
        EXPECT_TRUE(validateCase(c).hasErrors());
        c.time_ = TimeOptions{0, 1, 0.3, 4};
        EXPECT_FALSE(validateCase(c).hasErrors());
        c.output_.everySteps_ = 0;
        EXPECT_TRUE(validateCase(c).hasErrors());
        c = definition();
        c.output_.includeInitial_ = false;
        EXPECT_TRUE(validateCase(c).hasErrors());
        c = definition();
        c.linearSolver_.initialGuess_ = InitialGuess::PreviousStep;
        EXPECT_TRUE(validateCase(c).hasErrors());
        c = definition();
        c.linearSolver_.absoluteTolerance_ = 0;
        c.linearSolver_.relativeTolerance_ = 0;
        EXPECT_TRUE(validateCase(c).hasErrors());
    }
    TEST(CaseRegion, PreservesBlockOverridesIncludingExplicitEmpty) {
        auto d = meshData();
        mesh::Entity entity;
        entity.key_ = {2, 1};
        entity.physicalTags_ = {66};
        d.entities_.push_back(entity);
        d.elements_[0].classification_ = entity.key_;
        d.elements_[0].physicalTags_.reset();
        auto index = buildPhysicalMembership(mesh::Mesh(d));
        EXPECT_EQ(resolveRegion(group(2, 66), index).cells_.size(), 2);
        d.elements_[0].physicalTags_ = std::vector<std::int64_t>{};
        index = buildPhysicalMembership(mesh::Mesh(d));
        EXPECT_TRUE(resolveRegion(group(2, 66), index).diagnostics_.hasErrors());
        d.elements_[0].physicalTags_ = std::vector<std::int64_t>{11};
        index = buildPhysicalMembership(mesh::Mesh(d));
        EXPECT_EQ(resolveRegion(group(2, 11), index).cells_.size(), 2);
        EXPECT_TRUE(resolveRegion(group(2, 66), index).diagnostics_.hasErrors());
    }
    TEST(CaseRegion, ResolvesNamesUnnamedLargeTagsAndSetAlgebra) {
        auto index = buildPhysicalMembership(mesh::Mesh(meshData()));
        auto named = group(2, 11);
        named.tag_.reset();
        named.name_ = "body";
        EXPECT_EQ(resolveRegion(named, index).cells_.size(), 2);
        EXPECT_EQ(resolveRegion(group(2, 9007199254740993LL), index).cells_.size(), 2);
        RegionSelector combined;
        combined.kind_ = RegionKind::Union;
        combined.operands_ = {group(1, 21), group(1, 22), group(1, 21)};
        EXPECT_EQ(resolveRegion(combined, index).cells_.size(), 2);
        combined.kind_ = RegionKind::Intersection;
        combined.operands_ = {group(2, 11), group(2, 9007199254740993LL)};
        EXPECT_EQ(resolveRegion(combined, index).cells_.size(), 2);
        combined.kind_ = RegionKind::Difference;
        EXPECT_TRUE(hasCode(resolveRegion(combined, index).diagnostics_, "EMPTY_SELECTION"));
    }
    TEST(CaseRegion, RejectsAmbiguousEmptyMixedDimensionAndDepth) {
        auto d = meshData();
        d.physicalGroups_.push_back({{2, 99}, "body"});
        auto index = buildPhysicalMembership(mesh::Mesh(d));
        auto r = group(2, 11);
        r.tag_.reset();
        r.name_ = "body";
        EXPECT_TRUE(hasCode(resolveRegion(r, index).diagnostics_, "AMBIGUOUS_GROUP"));
        EXPECT_TRUE(hasCode(resolveRegion(group(2, 99), index).diagnostics_, "EMPTY_SELECTION"));
        RegionSelector s;
        s.kind_ = RegionKind::Union;
        s.operands_ = {group(2, 11), group(1, 21)};
        EXPECT_TRUE(resolveRegion(s, index).diagnostics_.hasErrors());
        s.operands_ = {group(2, 11)};
        CaseLimits limits;
        limits.maxSelectorDepth_ = 1;
        EXPECT_TRUE(hasCode(resolveRegion(s, index, limits).diagnostics_, "RESOURCE_LIMIT"));
    }
    TEST(FacetTopology, DistinguishesExteriorAndInternalWithoutBoundaryRecords) {
        mesh::Mesh m(meshData());
        auto t = mesh::buildFacetTopology(m, std::vector<mesh::CellRef>{{0, 0}, {0, 1}});
        ASSERT_TRUE(t.unsupported_.empty());
        ASSERT_EQ(t.facets_.size(), 5);
        EXPECT_EQ(std::count_if(t.facets_.begin(), t.facets_.end(), [](const auto& f) { return f.uses_.size() == 1; }),
                  4);
        EXPECT_THROW(mesh::buildFacetTopology(m, std::vector<mesh::CellRef>{{0, 0}, {0, 0}}), std::invalid_argument);
        EXPECT_THROW(mesh::buildFacetTopology(m, std::vector<mesh::CellRef>{{0, 0}}, 1), std::length_error);
    }
    TEST(FacetTopology, SupportsMixedFirstOrderCellCatalogAndRefusesHighOrderDowngrade) {
        for (const auto family : {mesh::CellFamily::Tetrahedron, mesh::CellFamily::Hexahedron, mesh::CellFamily::Prism,
                                  mesh::CellFamily::Pyramid}) {
            mesh::MeshData d;
            const std::size_t count = family == mesh::CellFamily::Tetrahedron ? 4
                : family == mesh::CellFamily::Hexahedron                      ? 8
                : family == mesh::CellFamily::Prism                           ? 6
                                                                              : 5;
            for (std::size_t i = 0; i < count; ++i) {
                d.nodes_.push_back({i + 1, {static_cast<double>(i), 0, 0}, {}, {}});
            }
            mesh::ElementBlock block;
            block.type_ = {family, 3, count, 1, "complete"};
            block.tags_ = {1};
            block.offsets_ = {0, count};
            for (std::size_t i = 0; i < count; ++i) {
                block.nodes_.push_back(i);
            }
            d.elements_.push_back(block);
            auto result = mesh::buildFacetTopology(mesh::Mesh(d), std::vector<mesh::CellRef>{{0, 0}});
            EXPECT_TRUE(result.unsupported_.empty());
            EXPECT_GE(result.facets_.size(), 4);
            d.elements_[0].type_.order_ = 2;
            auto unsupported = mesh::buildFacetTopology(mesh::Mesh(d), std::vector<mesh::CellRef>{{0, 0}});
            EXPECT_EQ(unsupported.unsupported_.size(), 1);
        }
    }
    TEST(CaseBinding, OwnsSnapshotPreservesUnitsAndReportsPendingExecution) {
        auto c = definition();
        c.mesh_.coordinateScale_ = 0.001;
        auto r = bind(c);
        ASSERT_TRUE(r.bound_);
        EXPECT_FALSE(r.diagnostics_.hasErrors());
        c.materials_[0].id_ = "changed";
        EXPECT_EQ(r.bound_->data().definition_.materials_[0].id_, "solid");
        EXPECT_DOUBLE_EQ(r.bound_->position(2)[0], 0.001);
        EXPECT_DOUBLE_EQ(r.bound_->data().definition_.analysis_.thickness_.value(), 0.01);
        EXPECT_EQ(r.bound_->data().insulatedFacets_.size(), 3);
        EXPECT_FALSE(r.bound_->data().capability_.executable());
        EXPECT_GE(r.bound_->data().capability_.pending_.size(), 5);
        EXPECT_THROW((void)r.bound_->position(100), std::out_of_range);
    }
    TEST(CaseBinding, RejectsNullMeshMaterialGapsAndOverlapTransactionally) {
        auto c = definition();
        EXPECT_FALSE(bindCase(c, {}).bound_);
        c.materialAssignments_.push_back({"duplicate", group(2, 11), "solid"});
        auto r = bind(c);
        EXPECT_FALSE(r.bound_);
        EXPECT_TRUE(hasCode(r.diagnostics_, "MATERIAL_OVERLAP"));
        c = definition();
        auto d = meshData();
        d.elements_[0].physicalTags_ = std::vector<std::int64_t>{88};
        c.analysis_.domain_ = group(2, 88);
        r = bind(c, d);
        EXPECT_FALSE(r.bound_);
        EXPECT_TRUE(hasCode(r.diagnostics_, "MATERIAL_GAP"));
    }
    TEST(CaseBinding, RejectsOutsideDomainButAllowsIndependentOverlappingSources) {
        auto c = definition();
        c.sources_ = {{"a", group(2, 11), ConstantScalar{1}}, {"b", group(2, 11), ConstantScalar{-1}}};
        auto r = bind(c);
        ASSERT_TRUE(r.bound_);
        EXPECT_EQ(r.bound_->data().sources_.size(), 2);
        auto d = meshData();
        auto extra = d.elements_[0];
        extra.tags_ = {999};
        extra.offsets_ = {0, 3};
        extra.nodes_ = {0, 1, 3};
        extra.physicalTags_ = std::vector<std::int64_t>{88};
        d.elements_.push_back(extra);
        c.sources_[0].region_ = group(2, 88);
        r = bind(c, d);
        EXPECT_FALSE(r.bound_);
        EXPECT_TRUE(hasCode(r.diagnostics_, "OUTSIDE_DOMAIN"));
    }
    TEST(CaseBinding, RejectsInteriorOrOrphanBoundary) {
        auto c = definition();
        c.boundaries_[0].region_ = group(1, 25);
        auto r = bind(c);
        EXPECT_FALSE(r.bound_);
        EXPECT_TRUE(hasCode(r.diagnostics_, "BOUNDARY_NOT_EXTERIOR"));
        auto d = meshData();
        d.nodes_.push_back({1000, {2, 2, 0}, {}, {}});
        d.elements_[1].nodes_ = {0, 4};
        c.boundaries_[0].region_ = group(1, 21);
        r = bind(c, d);
        EXPECT_TRUE(hasCode(r.diagnostics_, "BOUNDARY_NOT_EXTERIOR"));
    }
    TEST(CaseBinding, NaturalOverlapRequiresExplicitAddAndTemperatureRemainsExclusive) {
        auto c = definition();
        c.boundaries_ = {{"a", group(1, 21), FluxBoundary{ConstantScalar{1}}},
                         {"b", group(1, 21), ConvectionBoundary{ConstantScalar{20}, ConstantScalar{300}}}};
        EXPECT_TRUE(hasCode(bind(c).diagnostics_, "BOUNDARY_OVERLAP"));
        c.naturalOverlap_ = NaturalOverlap::Add;
        auto r = bind(c);
        ASSERT_TRUE(r.bound_);
        EXPECT_EQ(r.bound_->data().boundaries_[0].facets_, r.bound_->data().boundaries_[1].facets_);
        c.boundaries_[0].value_ = TemperatureBoundary{ConstantScalar{300}};
        EXPECT_FALSE(bind(c).bound_);
    }
    TEST(CaseBinding, EqualTemperatureOnSameSideAllowedDifferentSidesMayMeetAtCorner) {
        auto c = definition();
        c.boundaries_.push_back({"same", group(1, 24), TemperatureBoundary{ConstantScalar{300}}});
        EXPECT_TRUE(bind(c).bound_);
        c.boundaries_[1].value_ = TemperatureBoundary{ConstantScalar{301}};
        EXPECT_TRUE(hasCode(bind(c).diagnostics_, "TEMPERATURE_CONFLICT"));
        c.boundaries_[1].region_ = group(1, 21);
        auto r = bind(c);
        ASSERT_TRUE(r.bound_);
        EXPECT_FALSE(r.bound_->data().capability_.executable());
    }
    TEST(CaseBinding, ChecksInitialOverlapAndLeavesDofCompatibilityExplicit) {
        auto c = definition();
        makeTransient(c);
        c.initialState_->overrides_ = {{"a", group(2, 11), ConstantScalar{300}},
                                       {"b", group(2, 11), ConstantScalar{300}}};
        EXPECT_TRUE(hasCode(bind(c).diagnostics_, "INITIAL_OVERLAP"));
        c.initialState_->overrides_.pop_back();
        auto r = bind(c);
        ASSERT_TRUE(r.bound_);
        EXPECT_FALSE(r.bound_->data().capability_.executable());
    }
    TEST(CaseBinding, ValidatesPlanarityOverflowAndPartitionOwnership) {
        auto d = meshData();
        d.nodes_[0].position_[2] = 0.01;
        EXPECT_TRUE(hasCode(bind(definition(), d).diagnostics_, "NONPLANAR_DOMAIN"));
        d = meshData();
        d.partitionCount_ = 2;
        EXPECT_TRUE(hasCode(bind(definition(), d).diagnostics_, "UNSUPPORTED_PARTITION"));
        d = meshData();
        d.partitionCount_ = 2;
        d.ghostElements_.push_back({100, 1, {2}});
        EXPECT_FALSE(bind(definition(), d).bound_);
        d = meshData();
        d.nodes_[0].position_[0] = 1e300;
        auto c = definition();
        c.mesh_.coordinateScale_ = 1e300;
        EXPECT_TRUE(hasCode(bind(c, d).diagnostics_, "INVALID_GEOMETRY"));
    }
    TEST(CaseBinding, UnsupportedElementNeverBecomesExecutableOrSilentlyDowngraded) {
        auto d = meshData();
        d.elements_[0].type_.referenceLayout_ = "custom-layout";
        auto r = bind(definition(), d);
        ASSERT_TRUE(r.bound_);
        EXPECT_TRUE(r.bound_->data().capability_.diagnostics_.hasErrors());
        EXPECT_FALSE(r.bound_->data().capability_.executable());
        EXPECT_TRUE(r.bound_->data().insulatedFacets_.empty());
    }
    TEST(CaseBinding, ReportsUnusedMaterialsAndBudgetErrors) {
        auto c = definition();
        auto unused = c.materials_[0];
        unused.id_ = "unused";
        c.materials_.push_back(unused);
        auto r = bind(c);
        ASSERT_TRUE(r.bound_);
        EXPECT_TRUE(hasCode(r.diagnostics_, "UNUSED_MATERIAL"));
        CaseLimits limits;
        limits.maxValues_ = 5;
        auto mesh = std::make_shared<const mesh::Mesh>(meshData());
        EXPECT_FALSE(bindCase(definition(), mesh, limits).bound_);
    }
} // namespace
