/**
 * @file Snapshot.cuh
 * @brief 定义独立于CUDA和文件编码的类型化数值数组与不可变场快照。
 */
#pragma once
#include <DataStructure/Field/Grid.cuh>
#include <memory>
#include <optional>

namespace emberfield::field {
    /** @brief 区分给定初态和通过数值接受检查的结果。 */
    enum class SampleKind { Initial, Accepted };

    /** @brief 共享只读网格并拥有一帧物理场，不借用求解器缓存。
     * @details 复制深复制场并共享网格；移动后仅允许析构或赋值。
     * @code FieldSnapshot f(grid,0,std::nullopt,SampleKind::Accepted,points,cells); @endcode */
    class FieldSnapshot {
    public:
        /** @brief 接管并校验场数据。
         * @param grid 共享网格。@param step 物理步号。@param time 瞬态秒数或稳态空值。
         * @param kind 状态类别。@param points 节点场。@param cells 单元场。
         * @pre 网格非空，数组形状/名称合法，浮点有限。
         * @throws std::invalid_argument 形状或状态非法；资源异常传播。
         * @code FieldSnapshot f(g,0,0.,SampleKind::Initial,p,c); @endcode */
        FieldSnapshot(std::shared_ptr<const VisualizationGrid> grid, std::uint64_t step, std::optional<double> time,
                      SampleKind kind, std::vector<NumericArray> points, std::vector<NumericArray> cells);
        /** @brief 查询共享网格，无参数无额外前提。@return 只读所有权。@code auto g=f.grid(); @endcode */
        const std::shared_ptr<const VisualizationGrid>& grid() const noexcept { return grid_; }
        /** @brief 查询物理步，无参数无额外前提。@return 步号。@code auto k=f.step(); @endcode */
        std::uint64_t step() const noexcept { return step_; }
        /** @brief 查询时间，无参数无额外前提。@return 瞬态秒数或空。@code auto t=f.time(); @endcode */
        std::optional<double> time() const noexcept { return time_; }
        /** @brief 查询状态，无参数无额外前提。@return 状态类别。@code auto k=f.kind(); @endcode */
        SampleKind kind() const noexcept { return kind_; }
        /** @brief 借用节点场，无参数无额外前提。@return 生命期内只读span。@code auto a=f.pointFields(); @endcode */
        std::span<const NumericArray> pointFields() const noexcept { return pointFields_; }
        /** @brief 借用单元场，无参数无额外前提。@return 生命期内只读span。@code auto a=f.cellFields(); @endcode */
        std::span<const NumericArray> cellFields() const noexcept { return cellFields_; }

    private:
        std::shared_ptr<const VisualizationGrid> grid_; ///< 不可变网格。
        std::uint64_t step_{}; ///< 物理步号。
        std::optional<double> time_; ///< 秒制瞬态时间。
        SampleKind kind_{}; ///< 初态或接受态。
        std::vector<NumericArray> pointFields_; ///< 拥有节点场。
        std::vector<NumericArray> cellFields_; ///< 拥有单元场。
    };
} // namespace emberfield::field
