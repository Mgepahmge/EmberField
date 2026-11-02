/** @file FacetTopology.cu
 * @brief 实现不依赖物理组与几何容差的单元侧匹配。
 */
#include <DataStructure/Mesh/FacetTopology.cuh>
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
namespace thermofem::mesh {
    namespace {
        /** @brief 查询可靠的一阶局部侧目录。@param t 参考布局。
         * @pre 无。@return 局部节点位置表，未知时空。
         * @code auto sides=sideNodes(type); @endcode */
        std::vector<std::vector<Index>> sideNodes(const ElementType& t) {
            if (t.order_ != 1 || t.referenceLayout_ != "complete") {
                return {};
            }
            if (t.family_ == CellFamily::Triangle && t.dimension_ == 2 && t.nodeCount_ == 3) {
                return {{0, 1}, {1, 2}, {2, 0}};
            }
            if (t.family_ == CellFamily::Quadrilateral && t.dimension_ == 2 && t.nodeCount_ == 4) {
                return {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
            }
            if (t.family_ == CellFamily::Tetrahedron && t.dimension_ == 3 && t.nodeCount_ == 4) {
                return {{0, 2, 1}, {0, 1, 3}, {1, 2, 3}, {2, 0, 3}};
            }
            if (t.family_ == CellFamily::Hexahedron && t.dimension_ == 3 && t.nodeCount_ == 8) {
                return {{0, 3, 2, 1}, {0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}};
            }
            if (t.family_ == CellFamily::Prism && t.dimension_ == 3 && t.nodeCount_ == 6) {
                return {{0, 2, 1}, {3, 4, 5}, {0, 1, 4, 3}, {1, 2, 5, 4}, {2, 0, 3, 5}};
            }
            if (t.family_ == CellFamily::Pyramid && t.dimension_ == 3 && t.nodeCount_ == 5) {
                return {{0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
            }
            return {};
        }
    } // namespace
    FacetTopology buildFacetTopology(const Mesh& mesh, std::span<const CellRef> active, std::size_t maxFacets) {
        if (!maxFacets) {
            throw std::invalid_argument("侧访问限额必须为正");
        }
        FacetTopology result;
        std::set<CellRef> seen;
        std::map<std::vector<Index>, std::vector<FacetUse>> indexed;
        std::size_t count = 0;
        for (const auto cell : active) {
            if (!seen.insert(cell).second) {
                throw std::invalid_argument("活动单元重复");
            }
            const auto nodes = mesh.connectivity(cell.block_, cell.element_);
            const auto sides = sideNodes(mesh.data().elements_.at(cell.block_).type_);
            if (sides.empty()) {
                result.unsupported_.push_back(cell);
                continue;
            }
            for (std::size_t side = 0; side < sides.size(); ++side) {
                if (count++ == maxFacets) {
                    throw std::length_error("侧访问数超过限额");
                }
                std::vector<Index> key;
                for (const auto local : sides[side]) {
                    key.push_back(nodes[local]);
                }
                std::sort(key.begin(), key.end());
                indexed[std::move(key)].push_back({cell, side});
            }
        }
        for (auto& [nodes, uses] : indexed) {
            result.facets_.push_back({nodes, std::move(uses)});
        }
        return result;
    }
} // namespace thermofem::mesh
