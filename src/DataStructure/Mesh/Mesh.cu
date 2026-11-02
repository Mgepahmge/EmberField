/** @file Mesh.cu
 * @brief 验证宿主网格结构并建立独立于持久标签的连续索引。
 */
#include <DataStructure/Mesh/Mesh.cuh>
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace thermofem::mesh {
    namespace {
        /** @brief 拒绝非法结构。@param condition 要求满足的条件。@param message 上下文。
         * @pre message有效。@note 无返回值，失败抛invalid_argument。@code require(n>0,"size"); @endcode */
        void require(bool condition, const char* message) {
            if (!condition) {
                throw std::invalid_argument(message);
            }
        }
        /** @brief 检查实体键。@param key 实体身份。@pre 无。@note 无返回值。
         * @code validateKey({2,1}); @endcode */
        void validateKey(EntityKey key) {
            require(key.dimension_ >= 0 && key.dimension_ <= 3 && key.tag_ >= 0, "Invalid mesh entity key");
        }
        /** @brief 检查浮点样本。@param values 数值视图。@pre 无。@note 无返回值。
         * @code finite(node.position_); @endcode */
        void finite(std::span<const double> values) {
            for (double value : values) {
                require(std::isfinite(value), "Nonfinite mesh coordinate or attribute");
            }
        }
    } // namespace
    Mesh::Mesh(MeshData data) : data_(std::move(data)) {
        nodeIndices_.reserve(data_.nodes_.size());
        for (Index i = 0; i < data_.nodes_.size(); ++i) {
            const auto& node = data_.nodes_[i];
            require(node.tag_ > 0 && nodeIndices_.emplace(node.tag_, i).second, "Duplicate or zero node tag");
            finite(node.position_);
            finite(node.parameters_);
            if (node.classification_) {
                validateKey(*node.classification_);
                require(node.parameters_.empty() ||
                            node.parameters_.size() == static_cast<Index>(node.classification_->dimension_),
                        "Parametric coordinate dimension mismatch");
            }
            else {
                require(node.parameters_.empty(), "Unclassified node has parameter coordinates");
            }
        }
        std::unordered_set<Tag> elementTags;
        std::unordered_map<Tag, int> elementDimensions;
        for (const auto& block : data_.elements_) {
            if (block.classification_) {
                validateKey(*block.classification_);
                require(block.classification_->dimension_ == block.type_.dimension_,
                        "Element classification dimension mismatch");
            }
            require(block.type_.dimension_ >= 0 && block.type_.dimension_ <= 3 && block.type_.order_ >= -1 &&
                        !block.type_.referenceLayout_.empty(),
                    "Invalid reference element descriptor");
            require(block.offsets_.size() == block.tags_.size() + 1 && block.offsets_.front() == 0 &&
                        block.offsets_.back() == block.nodes_.size(),
                    "Invalid element offsets");
            for (Index i = 0; i < block.tags_.size(); ++i) {
                require(block.tags_[i] > 0 && elementTags.insert(block.tags_[i]).second,
                        "Duplicate or zero element tag");
                elementDimensions.emplace(block.tags_[i], block.type_.dimension_);
                const auto begin = block.offsets_[i], end = block.offsets_[i + 1];
                require(begin < end && end <= block.nodes_.size(), "Empty or unordered connectivity");
                require(!block.type_.nodeCount_ || end - begin == block.type_.nodeCount_,
                        "Reference element arity mismatch");
            }
            for (Index node : block.nodes_) {
                require(node < data_.nodes_.size(), "Dangling element node index");
            }
            if (block.physicalTags_) {
                for (auto tag : *block.physicalTags_) {
                    require(tag > 0, "Invalid cell physical group");
                }
            }
        }
        std::unordered_set<Tag> boundaryOwners;
        for (const auto& incidence : data_.cellBoundaries_) {
            require(elementDimensions.contains(incidence.cell_) && boundaryOwners.insert(incidence.cell_).second,
                    "Invalid or duplicate cell boundary owner");
            for (auto [tag, sign] : incidence.boundary_) {
                require(elementDimensions.contains(tag) &&
                            elementDimensions.at(tag) == elementDimensions.at(incidence.cell_) - 1 &&
                            (sign == 1 || sign == -1),
                        "Invalid oriented cell boundary");
            }
        }
        for (const auto* entities : {&data_.entities_, &data_.partitionedEntities_}) {
            std::set<std::pair<int, std::int64_t>> keys;
            for (const auto& entity : *entities) {
                validateKey(entity.key_);
                require(keys.emplace(entity.key_.dimension_, entity.key_.tag_).second, "Duplicate entity key");
                finite(entity.bounds_);
                for (auto tag : entity.physicalTags_) {
                    require(tag > 0, "Nonpositive physical tag");
                }
                for (const auto* incidences : {&entity.boundary_, &entity.embedded_}) {
                    for (const auto& edge : *incidences) {
                        validateKey(edge.entity_);
                        require(edge.entity_.dimension_ < entity.key_.dimension_ &&
                                    (edge.orientation_ == 1 || edge.orientation_ == -1),
                                "Invalid oriented entity incidence");
                    }
                }
                if (entity.parent_) {
                    validateKey(*entity.parent_);
                }
                for (int p : entity.partitions_) {
                    require(p > 0 && static_cast<Index>(p) <= data_.partitionCount_, "Invalid entity partition");
                }
            }
        }
        std::set<std::pair<int, std::int64_t>> groups;
        for (const auto& group : data_.physicalGroups_) {
            validateKey(group.key_);
            require(group.key_.tag_ > 0 && groups.emplace(group.key_.dimension_, group.key_.tag_).second,
                    "Duplicate or invalid physical group");
            require(group.name_.find_first_of("\r\n\0", 0, 3) == std::string::npos, "Invalid physical group name");
        }
        for (const auto& link : data_.periodic_) {
            validateKey(link.slave_);
            validateKey(link.master_);
            require(link.slave_.dimension_ == link.master_.dimension_ &&
                        (link.affine_.empty() || link.affine_.size() == 16),
                    "Invalid periodic entity map");
            finite(link.affine_);
            std::unordered_set<Index> slaveNodes;
            for (auto [slave, master] : link.nodes_) {
                require(slave < data_.nodes_.size() && master < data_.nodes_.size() && slaveNodes.insert(slave).second,
                        "Invalid periodic node correspondence");
            }
        }
        for (auto [tag, p] : data_.ghostEntities_) {
            require(tag > 0 && p > 0 && static_cast<Index>(p) <= data_.partitionCount_,
                    "Invalid ghost entity partition");
        }
        for (const auto& ghost : data_.ghostElements_) {
            require(ghost.elementTag_ > 0 && ghost.owner_ > 0 &&
                        static_cast<Index>(ghost.owner_) <= data_.partitionCount_,
                    "Invalid ghost element owner");
            for (int p : ghost.replicas_) {
                require(p > 0 && static_cast<Index>(p) <= data_.partitionCount_, "Invalid ghost replica partition");
            }
        }
        std::unordered_set<Tag> partitionedCells;
        for (const auto& membership : data_.elementPartitions_) {
            require(elementTags.contains(membership.elementTag_) &&
                        partitionedCells.insert(membership.elementTag_).second,
                    "Invalid element partition membership");
            std::unordered_set<int> seen;
            for (const auto* list : {&membership.partitions_, &membership.ghostPartitions_}) {
                for (int p : *list) {
                    require(p > 0 && static_cast<Index>(p) <= data_.partitionCount_ && seen.insert(p).second,
                            "Invalid or duplicate element partition");
                }
            }
        }
        for (const auto& parameter : data_.parametrizations_) {
            validateKey(parameter.entity_);
            require(parameter.entity_.dimension_ == 1 || parameter.entity_.dimension_ == 2,
                    "Invalid parametrization dimension");
            const Index stride = parameter.entity_.dimension_ == 1 ? 4 : 11;
            require(parameter.samples_.size() % stride == 0, "Invalid parametrization sample layout");
            finite(parameter.samples_);
            if (stride == 4) {
                require(parameter.triangles_.empty(), "Curve parametrization contains faces");
            }
            for (auto triangle : parameter.triangles_) {
                for (int index : triangle) {
                    require(index >= 0 && static_cast<Index>(index) < parameter.samples_.size() / stride,
                            "Invalid parametrization local index");
                }
            }
        }
    }
    Mesh::Mesh(Mesh&& other) noexcept { swap(other); }
    Mesh& Mesh::operator=(const Mesh& other) {
        if (this != &other) {
            Mesh copy(other);
            swap(copy);
        }
        return *this;
    }
    Mesh& Mesh::operator=(Mesh&& other) noexcept {
        if (this != &other) {
            Mesh moved(std::move(other));
            swap(moved);
        }
        return *this;
    }
    void Mesh::swap(Mesh& other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        nodeIndices_.swap(other.nodeIndices_);
    }
    Index Mesh::nodeIndex(Tag tag) const { return nodeIndices_.at(tag); }
    std::span<const Index> Mesh::connectivity(Index block, Index element) const {
        const auto& b = data_.elements_.at(block);
        if (element >= b.tags_.size()) {
            throw std::out_of_range("Mesh element index");
        }
        return std::span<const Index>(b.nodes_).subspan(b.offsets_[element],
                                                        b.offsets_[element + 1] - b.offsets_[element]);
    }
} // namespace thermofem::mesh
