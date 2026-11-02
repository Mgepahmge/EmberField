/** @file finite_element_space_test.cpp
 * @brief 验证连续P1编号、拓扑侧兼容性、诊断限额与只读生命周期。
 */
#include <FEM/Space/Space.cuh>
#include <algorithm>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <stdexcept>

namespace {
    using namespace emberfield;
    using namespace fem;
    /** @brief 创建用于独立编号测试的两个共享单纯形。
     * @param tetra 是否四面体。@pre 无。@return 构建数据。@code auto d=data(false); @endcode */
    mesh::MeshData data(bool tetra = false) {
        mesh::MeshData d;
        for (std::size_t n = 0; n < 7; ++n) {
            d.nodes_.push_back({10000000001ULL + 17 * n, {static_cast<double>(n), 0, 0}, {}, {}});
        }
        mesh::ElementBlock block;
        block.type_ = tetra ? mesh::ElementType{mesh::CellFamily::Tetrahedron, 3, 4, 1, "complete"}
                            : mesh::ElementType{mesh::CellFamily::Triangle, 2, 3, 1, "complete"};
        block.tags_ = {10, 90};
        block.offsets_ = tetra ? std::vector<std::size_t>{0, 4, 8} : std::vector<std::size_t>{0, 3, 6};
        block.nodes_ =
            tetra ? std::vector<std::size_t>{1, 4, 2, 0, 4, 1, 2, 5} : std::vector<std::size_t>{1, 4, 2, 4, 5, 2};
        d.elements_.push_back(block);
        return d;
    }
    /** @brief 对所有块中单元建立空间，使用指定预算。
     * @param d 构建数据。@param limits 预算。@pre 数据满足Mesh结构要求。
     * @return 空间结果。@code auto r=build(data()); @endcode */
    SpaceBuildResult build(mesh::MeshData d, SpaceLimits limits = {}) {
        std::vector<mesh::CellRef> active;
        std::vector<ElementKey> keys;
        for (std::size_t b = 0; b < d.elements_.size(); ++b) {
            for (std::size_t e = 0; e < d.elements_[b].tags_.size(); ++e) {
                active.push_back({b, e});
                keys.push_back({d.elements_[b].type_.family_});
            }
        }
        return buildFiniteElementSpace(std::make_shared<const mesh::Mesh>(std::move(d)), active, keys, limits);
    }
    /** @brief 查询诊断代码。@param r 结果。@param code 代码。@pre 无。
     * @return 是否存在。@code auto ok=has(r,SpaceIssueCode::RepeatedNode); @endcode */
    bool has(const SpaceBuildResult& r, SpaceIssueCode code) {
        return std::any_of(r.issues_.begin(), r.issues_.end(), [code](const SpaceIssue& i) { return i.code_ == code; });
    }
    /** @brief 将只读映射视图变为测试期望比较值。@param s 视图。@pre 有效。
     * @return 数组副本。@code auto v=copy(s.cellDofs(0)); @endcode */
    std::vector<DofIndex> copy(std::span<const DofIndex> s) { return {s.begin(), s.end()}; }

    TEST(FiniteElementSpace, TriangleMappingPreservesLocalOrderAndSharesEdge) {
        const auto r = build(data());
        ASSERT_TRUE(r.space_);
        EXPECT_FALSE(r.hasErrors_);
        const auto& s = *r.space_;
        EXPECT_EQ(s.dimension(), 2);
        EXPECT_EQ(s.cellCount(), 2);
        EXPECT_EQ(s.dofCount(), 4);
        EXPECT_EQ(copy(s.cellDofs(0)), (std::vector<DofIndex>{0, 2, 1}));
        EXPECT_EQ(copy(s.cellDofs(1)), (std::vector<DofIndex>{2, 3, 1}));
        const std::vector<std::size_t> nodes{1, 2, 4, 5};
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            EXPECT_EQ(s.dofInfo(i).node_, nodes[i]);
            EXPECT_EQ(s.nodeDof(nodes[i]), i);
        }
        EXPECT_FALSE(s.nodeDof(0));
        EXPECT_EQ(s.mesh().data().nodes_[1].tag_, 10000000018ULL);
        EXPECT_EQ(facetDofs(s, {{0, 0}, 2}), (std::vector<DofIndex>{1, 0}));
    }
    TEST(FiniteElementSpace, TetrahedraShareThreeFaceDofsWithOppositeLocalOrder) {
        const auto r = build(data(true));
        ASSERT_TRUE(r.space_);
        const auto& s = *r.space_;
        EXPECT_EQ(s.dimension(), 3);
        EXPECT_EQ(s.dofCount(), 5);
        EXPECT_EQ(copy(s.cellDofs(0)), (std::vector<DofIndex>{1, 3, 2, 0}));
        EXPECT_EQ(copy(s.cellDofs(1)), (std::vector<DofIndex>{3, 1, 2, 4}));
        auto face0 = facetDofs(s, {{0, 0}, 0});
        auto face1 = facetDofs(s, {{0, 1}, 0});
        EXPECT_NE(face0, face1);
        std::sort(face0.begin(), face0.end());
        std::sort(face1.begin(), face1.end());
        EXPECT_EQ(face0, face1);
        EXPECT_EQ(face0, (std::vector<DofIndex>{1, 2, 3}));
    }
    TEST(FiniteElementSpace, EveryTopologyFacetUsesSameReferenceSideConvention) {
        for (bool tetra : {false, true}) {
            const auto r = build(data(tetra));
            ASSERT_TRUE(r.space_);
            const auto& s = *r.space_;
            const auto topology = mesh::buildFacetTopology(s.mesh(), s.cells());
            ASSERT_TRUE(topology.unsupported_.empty());
            for (const auto& f : topology.facets_) {
                for (auto use : f.uses_) {
                    auto ids = facetDofs(s, use);
                    std::vector<mesh::Index> actual;
                    for (auto id : ids) {
                        actual.push_back(*s.dofInfo(id).node_);
                    }
                    std::sort(actual.begin(), actual.end());
                    EXPECT_EQ(actual, f.nodes_);
                }
            }
        }
    }
    TEST(FiniteElementSpace, NumberingIndependentOfActiveTraversalAndInputLifetime) {
        auto m = std::make_shared<const mesh::Mesh>(data());
        std::vector<mesh::CellRef> active{{0, 1}, {0, 0}};
        std::vector<ElementKey> keys(2);
        auto r = buildFiniteElementSpace(m, active, keys);
        ASSERT_TRUE(r.space_);
        m.reset();
        active.clear();
        keys.clear();
        EXPECT_EQ(copy(r.space_->cellDofs(0)), (std::vector<DofIndex>{2, 3, 1}));
        EXPECT_EQ(r.space_->cellPosition({0, 0}), 1);
        EXPECT_FALSE(r.space_->cellPosition({9, 9}));
        const auto other = build(data());
        for (std::size_t node = 0; node < 7; ++node) {
            EXPECT_EQ(r.space_->nodeDof(node), other.space_->nodeDof(node));
        }
    }
    TEST(FiniteElementSpace, ActiveSubsetDoesNotIncludeUnusedOrBoundaryNodes) {
        auto m = std::make_shared<const mesh::Mesh>(data());
        const std::vector<mesh::CellRef> active{{0, 1}};
        const std::vector<ElementKey> keys(1);
        auto r = buildFiniteElementSpace(m, active, keys);
        ASSERT_TRUE(r.space_);
        EXPECT_EQ(r.space_->dofCount(), 3);
        EXPECT_FALSE(r.space_->nodeDof(1));
        EXPECT_EQ(copy(r.space_->cellDofs(0)), (std::vector<DofIndex>{1, 2, 0}));
    }
    TEST(FiniteElementSpace, NodeStoragePermutationPreservesTagAssociation) {
        auto d = data();
        const auto old = build(d);
        std::reverse(d.nodes_.begin(), d.nodes_.end());
        for (auto& n : d.elements_[0].nodes_) {
            n = d.nodes_.size() - 1 - n;
        }
        const auto reordered = build(d);
        ASSERT_TRUE(old.space_);
        ASSERT_TRUE(reordered.space_);
        EXPECT_NE(copy(old.space_->cellDofs(0)), copy(reordered.space_->cellDofs(0)));
        for (std::size_t e = 0; e < 2; ++e) {
            for (std::size_t a = 0; a < 3; ++a) {
                const auto i = old.space_->cellDofs(e)[a];
                const auto j = reordered.space_->cellDofs(e)[a];
                EXPECT_EQ(old.space_->mesh().data().nodes_[*old.space_->dofInfo(i).node_].tag_,
                          reordered.space_->mesh().data().nodes_[*reordered.space_->dofInfo(j).node_].tag_);
            }
        }
    }
    TEST(FiniteElementSpace, SameCoordinatesDoNotMergeAndGeometryIsNotCertified) {
        auto d = data();
        for (auto& node : d.nodes_) {
            node.position_ = {0, 0, 0};
        }
        const auto r = build(d);
        ASSERT_TRUE(r.space_);
        EXPECT_EQ(r.space_->dofCount(), 4);
        EXPECT_NE(r.space_->nodeDof(1), r.space_->nodeDof(2));
    }
    TEST(FiniteElementSpace, CoefficientsGatherAndInterpolateAccordingToMapping) {
        const auto r = build(data());
        ASSERT_TRUE(r.space_);
        std::vector<double> temperatures{10, 20, 30, 40};
        const std::array<double, 3> point{0.2, 0.3, 0};
        auto table = evaluateBasis({}, std::span(&point, 1));
        double value = 0;
        auto row = r.space_->cellDofs(0);
        for (std::size_t a = 0; a < 3; ++a) {
            value += table.values_[a] * temperatures[row[a]];
        }
        EXPECT_NEAR(value, 0.5 * 10 + 0.2 * 30 + 0.3 * 20, 1e-14);
    }
    TEST(FiniteElementSpace, CopyMoveAndConcurrentQueriesUseNoMutableCache) {
        const auto r = build(data());
        ASSERT_TRUE(r.space_);
        auto copied = *r.space_;
        auto moved = std::move(copied);
        EXPECT_EQ(copy(moved.cellDofs(1)), copy(r.space_->cellDofs(1)));
        EXPECT_EQ(&moved.mesh(), &r.space_->mesh());
        auto work = [s = r.space_]() {
            for (int i = 0; i < 100; ++i) {
                if (s->nodeDof(4) != 2 || s->cellPosition({0, 1}) != 1) {
                    return false;
                }
            }
            return true;
        };
        auto a = std::async(std::launch::async, work);
        auto b = std::async(std::launch::async, work);
        EXPECT_TRUE(a.get());
        EXPECT_TRUE(b.get());
    }
    TEST(FiniteElementSpace, RejectsEmptyMismatchedAndOutOfRangeInputsSafely) {
        EXPECT_TRUE(has(buildFiniteElementSpace({}, {}, {}), SpaceIssueCode::NullMesh));
        auto m = std::make_shared<const mesh::Mesh>(data());
        EXPECT_TRUE(has(buildFiniteElementSpace(m, {}, {}), SpaceIssueCode::EmptyActive));
        const std::vector<mesh::CellRef> active{{0, 0}};
        EXPECT_TRUE(has(buildFiniteElementSpace(m, active, {}), SpaceIssueCode::LengthMismatch));
        const std::vector<ElementKey> keys(2);
        const std::vector<mesh::CellRef> invalid{{0, 99}, {99, 0}};
        auto r = buildFiniteElementSpace(m, invalid, keys);
        EXPECT_TRUE(r.hasErrors_);
        EXPECT_FALSE(r.space_);
        EXPECT_EQ(r.issues_.size(), 2);
        EXPECT_TRUE(has(r, SpaceIssueCode::CellOutOfRange));
        EXPECT_THROW(buildFiniteElementSpace(m, active, std::span(keys).first(1), {1, 1, 1, 0}), std::invalid_argument);
    }
    TEST(FiniteElementSpace, RejectsRepeatedCellsNodesAndCompleteConnectivity) {
        auto m = std::make_shared<const mesh::Mesh>(data());
        const std::vector<mesh::CellRef> repeated{{0, 0}, {0, 0}};
        const std::vector<ElementKey> keys(2);
        EXPECT_TRUE(has(buildFiniteElementSpace(m, repeated, keys), SpaceIssueCode::DuplicateCell));
        auto d = data();
        d.elements_[0].nodes_[2] = 1;
        auto r = build(d);
        EXPECT_TRUE(has(r, SpaceIssueCode::RepeatedNode));
        ASSERT_FALSE(r.issues_.empty());
        EXPECT_TRUE(r.issues_[0].localDof_);
        d = data();
        d.elements_[0].nodes_ = {1, 4, 2, 2, 1, 4};
        r = build(d);
        EXPECT_TRUE(has(r, SpaceIssueCode::DuplicateConnectivity));
        EXPECT_FALSE(r.space_);
    }
    TEST(FiniteElementSpace, RejectsMixedDimensionsAndUnsupportedGeometryWithoutDowngrade) {
        auto d = data();
        auto tet = data(true).elements_[0];
        tet.tags_ = {100, 200};
        d.elements_.push_back(tet);
        EXPECT_TRUE(has(build(d), SpaceIssueCode::DimensionMismatch));
        for (const std::string layout : {"custom", "edge-only"}) {
            d = data();
            d.elements_[0].type_.referenceLayout_ = layout;
            EXPECT_TRUE(has(build(d), SpaceIssueCode::IncompatibleGeometry));
        }
        d = data();
        d.elements_[0].type_.order_ = 2;
        EXPECT_TRUE(has(build(d), SpaceIssueCode::IncompatibleGeometry));
        d = data();
        d.elements_[0].type_.family_ = mesh::CellFamily::Custom;
        EXPECT_TRUE(has(build(d), SpaceIssueCode::UnsupportedElement));
        auto m = std::make_shared<const mesh::Mesh>(data());
        const std::vector<mesh::CellRef> active{{0, 0}, {0, 1}};
        const std::vector<ElementKey> highOrder(2, {mesh::CellFamily::Triangle, BasisFamily::ContinuousLagrange, 2});
        EXPECT_TRUE(has(buildFiniteElementSpace(m, active, highOrder), SpaceIssueCode::UnsupportedElement));
    }
    TEST(FiniteElementSpace, BudgetsAndDiagnosticTruncationDoNotPublishPartialSpace) {
        for (const auto limits : {SpaceLimits{1, 6, 4, 100}, SpaceLimits{2, 5, 4, 100}, SpaceLimits{2, 6, 3, 100}}) {
            const auto r = build(data(), limits);
            EXPECT_TRUE(r.hasErrors_);
            EXPECT_FALSE(r.space_);
            EXPECT_TRUE(has(r, SpaceIssueCode::ResourceLimit));
        }
        EXPECT_TRUE(build(data(), {2, 6, 4, 100}).space_);
        auto d = data();
        d.partitionCount_ = 1;
        d.elements_[0].type_.referenceLayout_ = "unknown";
        const auto r = build(d, {2, 6, 4, 1});
        ASSERT_EQ(r.issues_.size(), 1);
        EXPECT_FALSE(r.issues_[0].error_);
        EXPECT_TRUE(r.truncated_);
        EXPECT_TRUE(r.hasErrors_);
        EXPECT_FALSE(r.space_);
        const auto old = build(data());
        ASSERT_TRUE(old.space_);
        EXPECT_EQ(old.space_->dofCount(), 4);
    }
    TEST(FiniteElementSpace, MetadataNeverImplicitlyChangesNumbering) {
        auto d = data();
        d.periodic_.push_back({{0, 10}, {0, 20}, {}, {{1, 2}}});
        auto r = build(d);
        ASSERT_TRUE(r.space_);
        EXPECT_FALSE(r.hasErrors_);
        EXPECT_TRUE(has(r, SpaceIssueCode::UnappliedRelations));
        EXPECT_NE(r.space_->nodeDof(1), r.space_->nodeDof(2));
        EXPECT_STREQ(spaceIssueName(static_cast<SpaceIssueCode>(999)), "SPACE_UNKNOWN");
    }
    TEST(FiniteElementSpace, QueryBoundsAreCheckedWithoutNarrowing) {
        const auto r = build(data());
        ASSERT_TRUE(r.space_);
        const auto& s = *r.space_;
        EXPECT_THROW((void)s.cellDofs(s.cellCount()), std::out_of_range);
        EXPECT_THROW((void)s.elementKey(s.cellCount()), std::out_of_range);
        EXPECT_THROW((void)s.dofInfo(std::numeric_limits<DofIndex>::max()), std::out_of_range);
        EXPECT_THROW((void)s.nodeDof(7), std::out_of_range);
        EXPECT_THROW(facetDofs(s, {{0, 9}, 0}), std::out_of_range);
        EXPECT_THROW(facetDofs(s, {{0, 0}, 3}), std::out_of_range);
    }
} // namespace
