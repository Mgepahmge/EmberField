/** @file mesh_test.cpp
 * @brief 验证网格构造、独立拓扑表达和不变量，不依赖文件IO或GPU。
 */
#include <DataStructure/Mesh/Mesh.cuh>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>
namespace {
    using namespace emberfield::mesh;
    /** @brief 构造无分类的多边形网格。@pre 无。@return 测试数据。
     * @code auto d=polygon(); @endcode */
    MeshData polygon() {
        MeshData d;
        d.nodes_ = {{90, {0, 0, 0}, {}, {}},
                    {7, {1, 0, 0}, {}, {}},
                    {5000000000ULL, {1, 1, 0}, {}, {}},
                    {2, {0, 1, 0}, {}, {}}};
        ElementBlock b;
        b.type_ = {CellFamily::Polygon, 2, 0, -1, "vertex-cycle"};
        b.tags_ = {15};
        b.offsets_ = {0, 4};
        b.nodes_ = {0, 1, 2, 3};
        d.elements_.push_back(b);
        return d;
    }
    TEST(Mesh, KeepsTagsDistinctFromDenseConnectivity) {
        Mesh m(polygon());
        EXPECT_EQ(m.nodeIndex(5000000000ULL), 2);
        EXPECT_EQ(m.connectivity(0, 0)[3], 3);
        EXPECT_FALSE(m.data().nodes_[0].classification_);
        EXPECT_THROW((void)m.nodeIndex(3), std::out_of_range);
        EXPECT_THROW((void)m.connectivity(1, 0), std::out_of_range);
    }
    TEST(Mesh, OwnsDataByCompositionAndCopiesIndependently) {
        Mesh m(polygon());
        Mesh copy = m;
        auto data = copy.data();
        data.nodes_[0].position_[0] = 3;
        Mesh edited(std::move(data));
        EXPECT_EQ(m.data().nodes_[0].position_[0], 0);
        EXPECT_EQ(edited.data().nodes_[0].position_[0], 3);
        Mesh assigned;
        assigned = m;
        EXPECT_EQ(assigned.data(), m.data());
        Mesh moved(std::move(assigned));
        EXPECT_EQ(moved.data(), m.data());
        EXPECT_TRUE(assigned.data().nodes_.empty());
        assigned = std::move(moved);
        EXPECT_TRUE(moved.data().nodes_.empty());
        EXPECT_EQ(assigned.nodeIndex(90), 0);
        auto& alias = assigned;
        assigned = std::move(alias);
        EXPECT_EQ(assigned.nodeIndex(90), 0);
    }
    TEST(Mesh, SupportsExplicitOrientedCellBoundaryAndMixedDimensions) {
        auto d = polygon();
        ElementBlock edge;
        edge.type_ = {CellFamily::Segment, 1, 2, 1, "complete"};
        edge.tags_ = {16};
        edge.nodes_ = {0, 1};
        edge.offsets_ = {0, 2};
        d.elements_.push_back(edge);
        d.cellBoundaries_ = {{15, {{16, -1}}}};
        EXPECT_NO_THROW(Mesh{d});
        d.cellBoundaries_[0].boundary_[0].second = 0;
        EXPECT_THROW(Mesh{d}, std::invalid_argument);
    }
    TEST(Mesh, RejectsBrokenIdentityConnectivityAndParameters) {
        auto d = polygon();
        d.nodes_[1].tag_ = 90;
        EXPECT_THROW(Mesh{d}, std::invalid_argument);
        d = polygon();
        d.elements_[0].nodes_[0] = 4;
        EXPECT_THROW(Mesh{d}, std::invalid_argument);
        d = polygon();
        d.elements_[0].offsets_ = {0, 3};
        EXPECT_THROW(Mesh{d}, std::invalid_argument);
        d = polygon();
        d.nodes_[0].parameters_ = {0.2};
        EXPECT_THROW(Mesh{d}, std::invalid_argument);
        d = polygon();
        d.nodes_[0].position_[0] = std::numeric_limits<double>::infinity();
        EXPECT_THROW(Mesh{d}, std::invalid_argument);
    }
    TEST(Mesh, ValidatesPeriodicAndPartitionRelations) {
        auto d = polygon();
        d.periodic_.push_back({{1, 2}, {1, 1}, {}, {{0, 1}, {0, 2}}});
        EXPECT_THROW(Mesh{d}, std::invalid_argument);
        d = polygon();
        d.partitionCount_ = 2;
        d.ghostElements_ = {{15, 1, {2}}};
        EXPECT_NO_THROW(Mesh{d});
        d.ghostElements_[0].owner_ = 3;
        EXPECT_THROW(Mesh{d}, std::invalid_argument);
    }
} // namespace
