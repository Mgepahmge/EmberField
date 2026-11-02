/** @file Plan.cu
 * @brief 实现预算受检的符号排序、局部槽映射和并查集分量。
 */
#include <FEM/Assembly/Plan.cuh>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace thermofem::assembly {
    std::shared_ptr<const AssemblyPlan> buildAssemblyPlan(std::size_t n, std::span<const DofList> stencils,
                                                          AssemblyLimits limits) {
        if (!n || !limits.maxDofs_ || !limits.maxStencils_ || !limits.maxLocalEntries_ || !limits.maxNonzeros_) {
            throw std::invalid_argument("组装维度和预算必须为正");
        }
        if (n > limits.maxDofs_ || stencils.size() > limits.maxStencils_ || !std::in_range<Index>(n) ||
            n == std::numeric_limits<std::size_t>::max() || n > limits.maxNonzeros_) {
            throw std::length_error("符号组装维度或模板超限");
        }
        std::size_t entries = 0, locals = 0;
        for (const auto& row : stencils) {
            if (row.empty()) {
                throw std::invalid_argument("局部模板不得为空");
            }
            if (row.size() > limits.maxLocalEntries_ / row.size() ||
                row.size() * row.size() > limits.maxLocalEntries_ - entries) {
                throw std::length_error("局部矩阵展开超限");
            }
            entries += row.size() * row.size();
            if (row.size() > std::numeric_limits<std::size_t>::max() - locals) {
                throw std::length_error("局部向量展开溢出");
            }
            locals += row.size();
            auto sorted = row;
            std::sort(sorted.begin(), sorted.end());
            if (sorted.back() >= n) {
                throw std::out_of_range("局部DOF超出全局空间");
            }
            if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
                throw std::invalid_argument("同一局部模板重复DOF");
            }
        }
        if (entries > std::numeric_limits<std::size_t>::max() - n ||
            entries + n > std::numeric_limits<std::size_t>::max() / sizeof(std::pair<Index, Index>)) {
            throw std::length_error("符号候选字节数溢出");
        }
        std::vector<std::pair<Index, Index>> coordinates;
        coordinates.reserve(entries + n);
        for (std::size_t i = 0; i < n; ++i) {
            coordinates.emplace_back(i, i);
        }
        for (const auto& row : stencils) {
            for (auto a : row) {
                for (auto b : row) {
                    coordinates.emplace_back(a, b);
                }
            }
        }
        std::sort(coordinates.begin(), coordinates.end());
        coordinates.erase(std::unique(coordinates.begin(), coordinates.end()), coordinates.end());
        if (coordinates.size() > limits.maxNonzeros_ || !std::in_range<Index>(coordinates.size())) {
            throw std::length_error("规范CSR非零数超限");
        }
        auto p = std::shared_ptr<AssemblyPlan>(new AssemblyPlan);
        p->stencils_.assign(stencils.begin(), stencils.end());
        p->offsets_.assign(n + 1, 0);
        p->columns_.reserve(coordinates.size());
        for (auto [r, c] : coordinates) {
            ++p->offsets_[static_cast<std::size_t>(r) + 1];
            p->columns_.push_back(c);
        }
        std::partial_sum(p->offsets_.begin(), p->offsets_.end(), p->offsets_.begin());
        p->matrixSlots_.reserve(entries);
        p->matrixOffsets_.push_back(0);
        p->vectorOffsets_.push_back(0);
        std::vector<std::size_t> parent(n);
        std::iota(parent.begin(), parent.end(), 0);
        const auto root = [&](std::size_t i) {
            while (parent[i] != i) {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            return i;
        };
        for (const auto& row : stencils) {
            for (auto a : row) {
                auto ra = root(static_cast<std::size_t>(a)), rb = root(static_cast<std::size_t>(row[0]));
                parent[std::max(ra, rb)] = std::min(ra, rb);
                auto begin = p->columns_.begin() + p->offsets_[a], end = p->columns_.begin() + p->offsets_[a + 1];
                for (auto b : row) {
                    p->matrixSlots_.push_back(std::lower_bound(begin, end, static_cast<Index>(b)) -
                                              p->columns_.begin());
                }
            }
            p->matrixOffsets_.push_back(p->matrixSlots_.size());
            p->vectorOffsets_.push_back(p->vectorOffsets_.back() + row.size());
        }
        std::vector<std::size_t> labels(n, std::numeric_limits<std::size_t>::max());
        p->components_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const auto r = root(i);
            if (labels[r] == std::numeric_limits<std::size_t>::max()) {
                labels[r] = p->componentCount_++;
            }
            p->components_[i] = labels[r];
        }
        return p;
    }
} // namespace thermofem::assembly
