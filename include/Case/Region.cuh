/** @file Region.cuh
 * @brief 提供可独立复用的有效物理组索引与集合选择接口。
 */
#pragma once
#include <Case/Definition.cuh>
#include <Mesh/FacetTopology.cuh>
#include <map>
#include <set>

namespace thermofem::problem {
    using GroupKey = std::pair<int, std::int64_t>; ///< 维数与物理标签，不是几何实体身份。
    /** @brief 保存当前网格有效物理归属；支持无名组与块级空覆盖。
     * @code auto index=buildPhysicalMembership(mesh); @endcode */
    struct PhysicalMembership {
        std::map<GroupKey, std::vector<mesh::CellRef>> members_; ///< 组到排序、去重单元表。
        std::map<std::pair<int, std::string>, std::set<std::int64_t>> names_; ///< 名字到标签集合，保留歧义。
    };
    /** @brief 选择结果与诊断，不把错误当成合法空集合。
     * @code auto r=resolveRegion(selector,index); @endcode */
    struct RegionResult {
        std::vector<mesh::CellRef> cells_; ///< 去重排序结果。
        Diagnostics diagnostics_; ///< 未知组、名称歧义、空结果或限额错误。
    };
    /** @brief 为Mesh建立有效物理分组索引。
     * @param mesh 已验证网格。@param maxEntries 最大单元访问数和分组成员数。
     * @pre maxEntries为正。@return 拥有的索引，不借用网格。
     * @throws std::length_error 超限；invalid_argument零限额；标准分配异常向上传递。
     * @code auto index=buildPhysicalMembership(mesh); @endcode */
    [[nodiscard]] PhysicalMembership buildPhysicalMembership(const mesh::Mesh& mesh, std::size_t maxEntries = 1000000);
    /** @brief 解析物理组与并交差集合，可独立于算例绑定使用。
     * @param selector 选择表达式。@param index 有效分组索引。
     * @param limits 深度、对象和结果记录上限。@pre 限额为正；索引成员有序且去重。
     * @return 解析结果；空最终结果报告EMPTY_SELECTION，不静默选择全域。
     * @note 不判断外边界、材料或活动域。标准分配异常向上传递。
     * @code auto r=resolveRegion(selector,index); @endcode */
    [[nodiscard]] RegionResult resolveRegion(const RegionSelector& selector, const PhysicalMembership& index,
                                             CaseLimits limits = {});
} // namespace thermofem::problem
