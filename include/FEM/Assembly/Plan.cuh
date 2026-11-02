/** @file Plan.cuh
 * @brief 定义独立于网格和物理配置的规范CSR符号计划及有序局部编号模板。
 */
#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

/** @brief 提供局部贡献到全局规范稀疏结构的独立组装。 */
namespace emberfield::assembly {
    using Index = std::int64_t; ///< CSR行偏移和列索引。
    using DofList = std::vector<std::uint64_t>; ///< 一个局部模板的有序全局编号。
    /** @brief 控制符号展开规模的计数预算。
     * @code AssemblyLimits limits; limits.maxDofs_=10000; @endcode */
    struct AssemblyLimits {
        std::size_t maxDofs_{10000000}; ///< 全局未知量上限。
        std::size_t maxStencils_{10000000}; ///< 局部模板数上限。
        std::size_t maxLocalEntries_{64000000}; ///< 所有局部方阵条目合计上限。
        std::size_t maxNonzeros_{64000000}; ///< 规范CSR条目上限，包含显式零。
    };
    class AssemblyPlan;
    /** @brief 构建全对角、行内排序唯一的CSR模式和局部位置映射。
     * @param dofCount 全局维度。@param stencils 每个局部模板的有序编号。
     * @param limits 正预算。@pre 模板内编号唯一且小于dofCount。
     * @return 共享只读计划；不调用CUDA。
     * @throws std::invalid_argument 零维度、重复编号或非法预算。
     * @throws std::out_of_range 编号越界。@throws std::length_error 计数/索引超限。
     * @code auto p=buildAssemblyPlan(3,stencils); @endcode */
    [[nodiscard]] std::shared_ptr<const AssemblyPlan>
    buildAssemblyPlan(std::size_t dofCount, std::span<const DofList> stencils, AssemblyLimits limits = {});
    /** @brief 组合持有规范CSR图、局部条目槽映射及编号共现连通分量。
     * @details 所有索引保持64位；持久内存O(N+nnz+局部条目)。
     * 仅工厂构造成功对象，复制拥有独立数组，移动后仅可析构或赋值。
     * @code auto slot=p->matrixSlots()[p->matrixOffsets()[e]]; @endcode */
    class AssemblyPlan {
    public:
        /** @brief 全局维度，无参数无前提。@return N。@code auto n=p.dofCount(); @endcode */
        [[nodiscard]] std::size_t dofCount() const noexcept { return offsets_.size() - 1; }
        /** @brief 模板数，无参数无前提。@return 模板数。@code auto e=p.stencils().size(); @endcode */
        [[nodiscard]] const std::vector<DofList>& stencils() const noexcept { return stencils_; }
        /** @brief 行偏移，无参数无前提。@return 只读数组。@code auto& r=p.rowOffsets(); @endcode */
        [[nodiscard]] const std::vector<Index>& rowOffsets() const noexcept { return offsets_; }
        /** @brief 列索引，无参数无前提。@return 只读数组。@code auto& c=p.columns(); @endcode */
        [[nodiscard]] const std::vector<Index>& columns() const noexcept { return columns_; }
        /** @brief 矩阵模板偏移，无参数无前提。@return 局部宽度平方前缀和。@code auto& x=p.matrixOffsets(); @endcode */
        [[nodiscard]] const std::vector<std::size_t>& matrixOffsets() const noexcept { return matrixOffsets_; }
        /** @brief 向量模板偏移，无参数无前提。@return 局部宽度前缀和。@code auto& x=p.vectorOffsets(); @endcode */
        [[nodiscard]] const std::vector<std::size_t>& vectorOffsets() const noexcept { return vectorOffsets_; }
        /** @brief 局部矩阵槽映射，无参数无前提。@return 平铺局部条目对应CSR位置。@code auto& x=p.matrixSlots();
         * @endcode */
        [[nodiscard]] const std::vector<Index>& matrixSlots() const noexcept { return matrixSlots_; }
        /** @brief 逐DOF分量编号，无参数无前提。@return 稠密0起编号。@code auto& x=p.components(); @endcode */
        [[nodiscard]] const std::vector<std::size_t>& components() const noexcept { return components_; }
        /** @brief 分量数，无参数无前提。@return 包含孤立DOF的分量数。@code auto n=p.componentCount(); @endcode */
        [[nodiscard]] std::size_t componentCount() const noexcept { return componentCount_; }

    private:
        AssemblyPlan() = default; ///< 工厂候选。
        friend std::shared_ptr<const AssemblyPlan> buildAssemblyPlan(std::size_t, std::span<const DofList>,
                                                                     AssemblyLimits);
        std::vector<DofList> stencils_; ///< 原始模板序及局部编号。
        std::vector<Index> offsets_; ///< 规范CSR行偏移。
        std::vector<Index> columns_; ///< 行内唯一有序列。
        std::vector<std::size_t> matrixOffsets_; ///< 平铺局部方阵偏移。
        std::vector<std::size_t> vectorOffsets_; ///< 平铺局部向量偏移。
        std::vector<Index> matrixSlots_; ///< 局部方阵条目到全局CSR位置。
        std::vector<std::size_t> components_; ///< 每DOF的共现连通分量。
        std::size_t componentCount_{}; ///< 连通分量总数。
    };
} // namespace emberfield::assembly
