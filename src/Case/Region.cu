/** @file Region.cu
 * @brief 实现块覆盖优先的物理分组与确定性的选择器集合运算。
 */
#include <Case/Region.cuh>
#include <Case/Validation.cuh>
#include <algorithm>
#include <iterator>
#include <stdexcept>
namespace thermofem::problem {
    PhysicalMembership buildPhysicalMembership(const mesh::Mesh& mesh, std::size_t maxEntries) {
        if (!maxEntries) {
            throw std::invalid_argument("分组索引限额必须为正");
        }
        PhysicalMembership result;
        std::map<GroupKey, const mesh::Entity*> entities;
        for (const auto& e : mesh.data().entities_) {
            entities[{e.key_.dimension_, e.key_.tag_}] = &e;
        }
        for (const auto& e : mesh.data().partitionedEntities_) {
            entities[{e.key_.dimension_, e.key_.tag_}] = &e;
        }
        std::size_t count = 0;
        for (const auto& group : mesh.data().physicalGroups_) {
            if (count++ == maxEntries) {
                throw std::length_error("物理组索引超过限额");
            }
            result.members_[{group.key_.dimension_, group.key_.tag_}];
            if (!group.name_.empty()) {
                result.names_[{group.key_.dimension_, group.name_}].insert(group.key_.tag_);
            }
        }
        for (std::size_t block = 0; block < mesh.data().elements_.size(); ++block) {
            const auto& b = mesh.data().elements_[block];
            const std::vector<std::int64_t>* tags = nullptr;
            if (b.physicalTags_) {
                tags = &*b.physicalTags_;
            }
            else if (b.classification_) {
                auto it = entities.find({b.classification_->dimension_, b.classification_->tag_});
                if (it != entities.end()) {
                    tags = &it->second->physicalTags_;
                }
            }
            if (b.tags_.size() > maxEntries - count) {
                throw std::length_error("单元访问数超过限额");
            }
            count += b.tags_.size();
            if (!tags) {
                continue;
            }
            for (const auto tag : *tags) {
                if (b.tags_.size() > maxEntries - count) {
                    throw std::length_error("分组成员数超过限额");
                }
                count += b.tags_.size();
                auto& members = result.members_[{b.type_.dimension_, tag}];
                for (std::size_t element = 0; element < b.tags_.size(); ++element) {
                    members.push_back({block, element});
                }
            }
        }
        return result;
    }
    namespace {
        /** @brief 有限递归选择状态，每次请求独立计数。
         * @code Resolver resolver{index,limits}; @endcode */
        struct Resolver {
            const PhysicalMembership& index_; ///< 借用有效索引。
            CaseLimits limits_; ///< 工作限额。
            Diagnostics report_; ///< 聚合诊断。
            std::size_t visited_{}; ///< 节点访问预算。
            Resolver(const PhysicalMembership& index, CaseLimits limits) :
                index_(index), limits_(limits), report_(limits.maxDiagnostics_) {}
            void error(const std::string& code, const std::string& path, const std::string& message) {
                report_.add({Severity::Error, code, path, {}, {}, message, "检查分组身份、维数和集合表达式"});
            }
            std::pair<int, std::vector<mesh::CellRef>> walk(const RegionSelector& s, const std::string& path,
                                                            std::size_t depth) {
                if (depth > limits_.maxSelectorDepth_ || visited_++ >= limits_.maxValues_) {
                    error("RESOURCE_LIMIT", path, "选择器超过访问限额");
                    return {-1, {}};
                }
                if (s.kind_ == RegionKind::PhysicalGroup) {
                    if (s.dimension_ < 0 || s.dimension_ > 3 || (s.tag_.has_value() == s.name_.has_value()) ||
                        !s.operands_.empty() || (s.tag_ && *s.tag_ <= 0) ||
                        (s.name_ && (s.name_->empty() || s.name_->size() > limits_.maxStringBytes_))) {
                        error("INVALID_SELECTOR", path, "物理组身份非法");
                        return {-1, {}};
                    }
                    auto tag = s.tag_;
                    if (s.name_) {
                        const auto it = index_.names_.find({s.dimension_, *s.name_});
                        if (it == index_.names_.end()) {
                            error("UNKNOWN_GROUP", path, "物理组名称不存在");
                            return {s.dimension_, {}};
                        }
                        if (it->second.size() != 1) {
                            error("AMBIGUOUS_GROUP", path, "同维物理组名称对应多个标签");
                            return {s.dimension_, {}};
                        }
                        tag = *it->second.begin();
                    }
                    const auto it = index_.members_.find({s.dimension_, *tag});
                    if (it == index_.members_.end()) {
                        error("UNKNOWN_GROUP", path, "物理组标签不存在");
                        return {s.dimension_, {}};
                    }
                    if (it->second.size() > limits_.maxValues_) {
                        error("RESOURCE_LIMIT", path, "选择结果超过限额");
                        return {s.dimension_, {}};
                    }
                    return {s.dimension_, it->second};
                }
                if ((s.kind_ != RegionKind::Union && s.kind_ != RegionKind::Intersection &&
                     s.kind_ != RegionKind::Difference) ||
                    s.operands_.empty() || (s.kind_ == RegionKind::Difference && s.operands_.size() != 2) ||
                    s.dimension_ != -1 || s.tag_ || s.name_ || s.operands_.size() > limits_.maxValues_) {
                    error("INVALID_SELECTOR", path, "集合表达式非法");
                    return {-1, {}};
                }
                auto [dim, members] = walk(s.operands_[0], path + "/operands/0", depth + 1);
                for (std::size_t i = 1; i < s.operands_.size(); ++i) {
                    auto [nextDim, next] = walk(s.operands_[i], path + "/operands/" + std::to_string(i), depth + 1);
                    if (nextDim != dim) {
                        error("INVALID_SELECTOR", path, "集合运算不能混合维数");
                        return {-1, {}};
                    }
                    if (next.size() > limits_.maxValues_ - members.size()) {
                        error("RESOURCE_LIMIT", path, "集合运算中间规模超过限额");
                        return {dim, {}};
                    }
                    std::vector<mesh::CellRef> merged;
                    if (s.kind_ == RegionKind::Union) {
                        std::set_union(members.begin(), members.end(), next.begin(), next.end(),
                                       std::back_inserter(merged));
                    }
                    else if (s.kind_ == RegionKind::Intersection) {
                        std::set_intersection(members.begin(), members.end(), next.begin(), next.end(),
                                              std::back_inserter(merged));
                    }
                    else {
                        std::set_difference(members.begin(), members.end(), next.begin(), next.end(),
                                            std::back_inserter(merged));
                    }
                    members = std::move(merged);
                }
                return {dim, std::move(members)};
            }
        };
    } // namespace
    RegionResult resolveRegion(const RegionSelector& selector, const PhysicalMembership& index, CaseLimits limits) {
        validateLimits(limits);
        Resolver resolver(index, limits);
        auto [dim, cells] = resolver.walk(selector, "", 1);
        if (!resolver.report_.hasErrors() && cells.empty()) {
            resolver.error("EMPTY_SELECTION", "", "最终区域选择为空");
        }
        return {std::move(cells), std::move(resolver.report_)};
    }
} // namespace thermofem::problem
