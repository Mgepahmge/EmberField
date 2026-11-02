/** @file FacetTopology.cuh
 * @brief 从活动单元构造独立的余维一拓扑，不依赖物理模型和文件IO。
 */
#pragma once
#include <DataStructure/Mesh/Mesh.cuh>
#include <compare>

namespace thermofem::mesh {
    /** @brief 当前网格中的单元位置，与持久tag和未来DOF编号分离。
     * @code CellRef cell{0,2}; @endcode */
    struct CellRef {
        Index block_{}; ///< 块索引。
        Index element_{}; ///< 块内单元索引。
        auto operator<=>(const CellRef&) const = default; ///< 按位置进行确定性比较。
    };
    /** @brief 单元侧的拥有者及局部序号。
     * @code FacetUse use{{0,2},1}; @endcode */
    struct FacetUse {
        CellRef cell_; ///< 拥有该侧的活动单元。
        Index localSide_{}; ///< 参考单元内的侧序号，不等于法向方向。
    };
    /** @brief 按节点身份规范化的单元侧。
     * @code bool exterior=facet.uses_.size()==1; @endcode */
    struct Facet {
        std::vector<Index> nodes_; ///< 排序后的完整节点身份，只用于匹配，不用于积分。
        std::vector<FacetUse> uses_; ///< 1为外侧，2为内部侧，大于2为非流形。
    };
    /** @brief 余维一拓扑的构建结果；未知布局不自动降阶。
     * @code auto topology=buildFacetTopology(mesh,active); @endcode */
    struct FacetTopology {
        std::vector<Facet> facets_; ///< 已识别的侧，确定性排序。
        std::vector<CellRef> unsupported_; ///< 无可靠侧目录的单元；非空时不能判定全域外边界。
    };
    /** @brief 为完整一阶常见单元建立活动域侧关联。
     * @param mesh 借用不可变网格。@param active 唯一有效单元引用。
     * @param maxFacets 最大局部侧访问数，防止中间表无限增长。
     * @pre maxFacets为正；active不含重复；无几何质量前提。
     * @return 独立结果；支持Triangle3、Quad4、Tet4、Hex8、Prism6和Pyramid5。
     * @throws std::invalid_argument 重复单元或零限额；out_of_range引用越界；length_error超过限额。
     * @note 不检查Jacobian，不从坐标合并节点，不提供外法向。
     * @code auto t=buildFacetTopology(mesh,active); @endcode */
    [[nodiscard]] FacetTopology buildFacetTopology(const Mesh& mesh, std::span<const CellRef> active,
                                                   std::size_t maxFacets = 10000000);
} // namespace thermofem::mesh
