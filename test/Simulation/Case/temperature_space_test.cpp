/** @file temperature_space_test.cpp
 * @brief 验证算例与独立有限元空间的单向适配，保留未完成物理检查。
 */
#include <Simulation/Case/TemperatureSpace.cuh>
#include <gtest/gtest.h>
#include <stdexcept>
#include "support/CaseFixture.cuh"

namespace {
    using namespace emberfield;
    TEST(TemperatureSpace, SteadyAndTransientHaveSameUnconstrainedNumbering) {
        auto definition = casefixture::definition();
        auto steady = casefixture::bind(definition);
        ASSERT_TRUE(steady.bound_);
        auto first = problem::buildTemperatureSpace(*steady.bound_);
        ASSERT_TRUE(first.space_);
        EXPECT_EQ(first.space_->dofCount(), 4);
        EXPECT_FALSE(first.diagnostics_.hasErrors());
        casefixture::makeTransient(definition);
        definition.materials_[0].conductivity_ = problem::ConstantScalar{30};
        auto transient = casefixture::bind(definition);
        ASSERT_TRUE(transient.bound_);
        auto second = problem::buildTemperatureSpace(*transient.bound_);
        ASSERT_TRUE(second.space_);
        for (std::size_t n = 0; n < 4; ++n) {
            EXPECT_EQ(first.space_->nodeDof(n), second.space_->nodeDof(n));
        }
        EXPECT_FALSE(steady.bound_->data().capability_.executable());
        bool pendingDofs = false;
        for (const auto& p : steady.bound_->data().capability_.pending_) {
            pendingDofs |= p.kind_ == problem::PendingKind::DegreesOfFreedom;
        }
        EXPECT_TRUE(pendingDofs);
        steady.bound_.reset();
        transient.bound_.reset();
        EXPECT_EQ(first.space_->mesh().data().nodes_[0].tag_, 5000000001ULL);
    }
    TEST(TemperatureSpace, MaterialInterfacesShareNodesWithoutAffectingNumbering) {
        auto d = casefixture::meshData();
        auto c = casefixture::definition();
        auto second = d.elements_[0];
        second.tags_ = {300};
        second.offsets_ = {0, 3};
        second.nodes_ = {0, 2, 3};
        second.physicalTags_ = std::vector<std::int64_t>{12};
        d.elements_[0].tags_ = {100};
        d.elements_[0].offsets_ = {0, 3};
        d.elements_[0].nodes_ = {0, 1, 2};
        d.elements_.push_back(second);
        auto region = casefixture::group(2, 11);
        c.analysis_.domain_ = {};
        c.analysis_.domain_.kind_ = problem::RegionKind::Union;
        c.analysis_.domain_.operands_ = {region, casefixture::group(2, 12)};
        c.materials_.push_back(c.materials_[0]);
        c.materials_[1].id_ = "other";
        c.materialAssignments_.push_back({"second", casefixture::group(2, 12), "other"});
        auto bound = casefixture::bind(c, d);
        ASSERT_TRUE(bound.bound_);
        auto r = problem::buildTemperatureSpace(*bound.bound_);
        ASSERT_TRUE(r.space_);
        EXPECT_EQ(r.space_->dofCount(), 4);
        EXPECT_EQ(r.space_->cellDofs(0)[0], r.space_->cellDofs(1)[0]);
        EXPECT_EQ(r.space_->cellDofs(0)[2], r.space_->cellDofs(1)[1]);
    }
    TEST(TemperatureSpace, ThreeDimensionalCaseUsesTetrahedronP1) {
        mesh::MeshData d;
        d.nodes_ = {{1, {0, 0, 0}, {}, {}}, {2, {1, 0, 0}, {}, {}}, {3, {0, 1, 0}, {}, {}}, {4, {0, 0, 1}, {}, {}}};
        mesh::ElementBlock block;
        block.type_ = {mesh::CellFamily::Tetrahedron, 3, 4, 1, "complete"};
        block.tags_ = {10};
        block.offsets_ = {0, 4};
        block.nodes_ = {0, 1, 2, 3};
        block.physicalTags_ = std::vector<std::int64_t>{11};
        d.elements_.push_back(block);
        auto c = casefixture::definition();
        c.analysis_.space_ = problem::SpaceModel::Solid3d;
        c.analysis_.thickness_.reset();
        c.analysis_.domain_ = casefixture::group(3, 11);
        c.materialAssignments_[0].region_ = casefixture::group(3, 11);
        c.boundaries_.clear();
        auto bound = casefixture::bind(c, d);
        ASSERT_TRUE(bound.bound_);
        auto r = problem::buildTemperatureSpace(*bound.bound_);
        ASSERT_TRUE(r.space_);
        EXPECT_EQ(r.space_->dimension(), 3);
        EXPECT_EQ(r.space_->dofCount(), 4);
    }
    TEST(TemperatureSpace, RejectsUnsupportedOrderAndBudgetWithoutChangingBinding) {
        auto c = casefixture::definition();
        c.discretization_.order_ = 2;
        auto binding = casefixture::bind(c);
        ASSERT_TRUE(binding.bound_);
        auto r = problem::buildTemperatureSpace(*binding.bound_);
        EXPECT_FALSE(r.space_);
        EXPECT_TRUE(casefixture::hasCode(r.diagnostics_, "SPACE_UNSUPPORTED_ELEMENT"));
        EXPECT_EQ(binding.bound_->data().definition_.discretization_.order_, 2);
        binding = casefixture::bind(casefixture::definition());
        ASSERT_TRUE(binding.bound_);
        r = problem::buildTemperatureSpace(*binding.bound_, {1, 6, 4, 10});
        EXPECT_TRUE(casefixture::hasCode(r.diagnostics_, "SPACE_RESOURCE_LIMIT"));
        EXPECT_THROW(problem::buildTemperatureSpace(*binding.bound_, {0, 1, 1, 1}), std::invalid_argument);
    }
    TEST(TemperatureSpace, RejectsPeriodicPolicyAndRetainsTruncatedErrors) {
        auto d = casefixture::meshData();
        d.periodic_.push_back({{0, 10}, {0, 20}, {}, {{0, 1}}});
        auto bound = casefixture::bind(casefixture::definition(), d);
        ASSERT_TRUE(bound.bound_);
        auto r = problem::buildTemperatureSpace(*bound.bound_);
        EXPECT_FALSE(r.space_);
        EXPECT_TRUE(casefixture::hasCode(r.diagnostics_, "SPACE_PERIODIC_POLICY_REQUIRED"));
        d = casefixture::meshData();
        d.partitionCount_ = 1;
        auto c = casefixture::definition();
        c.discretization_.order_ = 2;
        bound = casefixture::bind(c, d);
        ASSERT_TRUE(bound.bound_);
        r = problem::buildTemperatureSpace(*bound.bound_, {2, 6, 4, 1});
        EXPECT_FALSE(r.space_);
        EXPECT_TRUE(r.diagnostics_.hasErrors());
        EXPECT_GT(r.diagnostics_.omitted(), 0);
    }
} // namespace
