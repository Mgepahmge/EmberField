/** @file ConditionSupport.cuh
 * @brief 提供条件与联合准备共用的身份、预算及诊断工具，不包含物理积分。
 */
#pragma once
#include <Case/Conditions.cuh>
#include <FiniteElement/Geometry.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
namespace thermofem::problem::detail {
    /** @brief 添加准备错误。@param d 诊断。@param code 稳定码。@param message 中文说明。
     * @param id 配置id。@param tag 可选网格tag。@pre 无。@note 追加无返回，系统异常传播。
     * @code issue(d,"CONTEXT","空间错配"); @endcode */
    inline void issue(Diagnostics& d, std::string code, std::string message, std::string id = {},
                      std::optional<std::uint64_t> tag = {}) {
        d.add({Severity::Error, std::move(code), "/preparation", std::move(id), tag, std::move(message)});
    }
    /** @brief 校验bound与space的精确关联。@param b 绑定。@param s 空间。@param d 追加诊断。
     * @pre 输入引用有效。@return 是否通过本次核对。@code if(!context(b,s,d)) return; @endcode */
    inline bool context(const BoundCase& b, const fem::FiniteElementSpace& s, Diagnostics& d) {
        const auto& data = b.data();
        const auto& def = data.definition_;
        if (&s.mesh() != data.mesh_.get() || s.cellCount() != data.active_.size() ||
            !std::equal(s.cells().begin(), s.cells().end(), data.active_.begin())) {
            issue(d, "PREPARATION_CONTEXT_MISMATCH", "空间必须使用同一Mesh对象和同序活动单元");
            return false;
        }
        const unsigned dimension = def.analysis_.space_ == SpaceModel::Planar2d ? 2 : 3;
        if (s.dimension() != dimension || !s.cellCount()) {
            issue(d, "PREPARATION_SPACE_MISMATCH", "分析维数和空间不匹配");
            return false;
        }
        for (std::size_t row = 0; row < s.cellCount(); ++row) {
            const fem::ElementKey expected{dimension == 2 ? mesh::CellFamily::Triangle : mesh::CellFamily::Tetrahedron,
                                           fem::BasisFamily::ContinuousLagrange, def.discretization_.order_};
            if (s.elementKey(row) != expected || expected.order_ != 1) {
                issue(d, "PREPARATION_UNSUPPORTED_ELEMENT", "仅支持对应维数的Triangle3/Tet4连续P1");
                return false;
            }
        }
        if (!data.mesh_->data().periodic_.empty()) {
            issue(d, "SPACE_PERIODIC_POLICY_REQUIRED", "尚未定义周期温度条件");
            return false;
        }
        if (data.capability_.diagnostics_.hasErrors()) {
            d.append(data.capability_.diagnostics_);
            return false;
        }
        return true;
    }
    /** @brief 消费受限计数。@param used 已消费。@param count 增量。@param limit 上限。@param d 诊断。
     * @pre used<=limit。@return 是否可增加；失败不改used。@code if(!consume(n,count,limit,d)) return; @endcode */
    inline bool consume(std::size_t& used, std::size_t count, std::size_t limit, Diagnostics& d) {
        if (count > limit - used) {
            issue(d, "PREPARATION_LIMIT", "条件展开访问数超过限额");
            return false;
        }
        used += count;
        return true;
    }
    /** @brief 获得DOF原始tag。@param s 空间。@param dof 全局DOF。@pre dof有效。
     * @return 节点tag或空。@code auto tag=dofTag(s,i); @endcode */
    inline std::optional<std::uint64_t> dofTag(const fem::FiniteElementSpace& s, fem::DofIndex dof) {
        const auto node = s.dofInfo(dof).node_;
        return node ? std::optional<std::uint64_t>{s.mesh().data().nodes_.at(*node).tag_} : std::nullopt;
    }
    /** @brief 读取有限非负温度并规范化零。@param field 场。@pre 已经静态校验。
     * @return 温度；若内部不变量被破坏则logic_error。@code auto t=temperature(field); @endcode */
    inline double temperature(const ScalarField& field) {
        const double t = evaluate(field);
        if (!std::isfinite(t) || t < 0) {
            throw std::logic_error("已绑定温度违反静态不变量");
        }
        return t == 0 ? 0 : t;
    }
} // namespace thermofem::problem::detail
