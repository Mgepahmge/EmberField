/** @file Integration.cuh
 * @brief 提供拥有实际坐标、正权重与实际基值的独立体/侧积分表。
 */
#pragma once
#include <FiniteElement/Geometry.cuh>
#include <FiniteElement/Quadrature.cuh>
namespace thermofem::fem {
    class CellIntegrationData;
    class FacetIntegrationData;
    /** @brief 构建单元积分表。@param geometry 已准备几何。@param cellRow 活动行。
     * @param degree 请求精度0..2。@pre geometry非空。@return 拥有数组的表。
     * @throws std::invalid_argument 非法请求；out_of_range越界；EvaluationError数值失败。
     * @code auto t=evaluateCellIntegration(g,0,2); @endcode */
    [[nodiscard]] CellIntegrationData evaluateCellIntegration(std::shared_ptr<const PreparedGeometry> geometry,
                                                              std::size_t cellRow, unsigned degree = 2);
    /** @brief 构建外侧积分表。@param geometry 已准备几何。@param facetIndex 拓扑侧行。
     * @param degree 请求精度0..2。@pre geometry非空。
     * @return 拥有数组的迹表；内部侧invalid_argument，越界out_of_range，数值失败EvaluationError。
     * @code auto t=evaluateFacetIntegration(g,i,2); @endcode */
    [[nodiscard]] FacetIntegrationData evaluateFacetIntegration(std::shared_ptr<const PreparedGeometry> geometry,
                                                                std::size_t facetIndex, unsigned degree = 2);
    /** @brief 母单元实际基与梯度，权重不含厚度，查询不延长对象寿命。
     * @code auto t=evaluateCellIntegration(g,0,2); @endcode */
    class CellIntegrationData {
    public:
        /** @brief 空间来源；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.space(); @endcode */
        [[nodiscard]] const std::shared_ptr<const FiniteElementSpace>& space() const noexcept { return space_; }

        /** @brief 母单元空间行；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.cellRow(); @endcode */
        [[nodiscard]] std::size_t cellRow() const noexcept { return cellRow_; }

        /** @brief 母单元维数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.dimension(); @endcode */
        [[nodiscard]] unsigned dimension() const noexcept { return dimension_; }

        /** @brief 真实规则精度；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.exactDegree();
         * @endcode */
        [[nodiscard]] unsigned exactDegree() const noexcept { return exactDegree_; }

        /** @brief 米制实际积分点；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.points(); @endcode
         */
        [[nodiscard]] const std::vector<std::array<double, 3>>& points() const noexcept { return points_; }

        /** @brief 含实际测度的正权重；无参数无额外前提。@return 只读值或借用引用。@code auto
         * value=obj.physicalWeights(); @endcode */
        [[nodiscard]] const std::vector<double>& physicalWeights() const noexcept { return physicalWeights_; }

        /** @brief 逐点局部基值；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.values(); @endcode */
        [[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }

        /** @brief 逐点实际梯度；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.gradients(); @endcode
         */
        [[nodiscard]] const std::vector<double>& gradients() const noexcept { return gradients_; }

        /** @brief 积分点数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.pointCount(); @endcode */
        [[nodiscard]] std::size_t pointCount() const noexcept { return points_.size(); }

        /** @brief 局部自由度数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.dofCount(); @endcode
         */
        [[nodiscard]] std::size_t dofCount() const noexcept { return space_->cellDofs(cellRow_).size(); }

    private:
        CellIntegrationData() = default; ///< 仅积分工厂构造成功候选。
        friend CellIntegrationData evaluateCellIntegration(std::shared_ptr<const PreparedGeometry>, std::size_t,
                                                           unsigned);
        std::shared_ptr<const FiniteElementSpace> space_; ///< 空间生命周期。
        std::size_t cellRow_{}; ///< 母单元行。
        unsigned dimension_{}; ///< 母元维数。
        unsigned exactDegree_{}; ///< 实际规则精度。
        std::vector<std::array<double, 3>> points_; ///< 实际点坐标。
        std::vector<double> physicalWeights_; ///< 不含厚度的权重。
        std::vector<double> values_; ///< q*n+a的基值。
        std::vector<double> gradients_; ///< (q*n+a)*d+k实际梯度。
    };
    /** @brief 外侧迹基及外法向，权重不含厚度，查询不延长对象寿命。
     * @code auto t=evaluateFacetIntegration(g,0,2); @endcode */
    class FacetIntegrationData {
    public:
        /** @brief 空间来源；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.space(); @endcode */
        [[nodiscard]] const std::shared_ptr<const FiniteElementSpace>& space() const noexcept { return space_; }

        /** @brief 母单元空间行；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.cellRow(); @endcode */
        [[nodiscard]] std::size_t cellRow() const noexcept { return cellRow_; }

        /** @brief 拓扑侧行；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.facetIndex(); @endcode */
        [[nodiscard]] std::size_t facetIndex() const noexcept { return facetIndex_; }

        /** @brief 母单元侧；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.use(); @endcode */
        [[nodiscard]] mesh::FacetUse use() const noexcept { return use_; }

        /** @brief 迹到母局部；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.parentLocalDofs();
         * @endcode */
        [[nodiscard]] const std::vector<std::size_t>& parentLocalDofs() const noexcept { return parentLocalDofs_; }

        /** @brief 母单元维数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.dimension(); @endcode */
        [[nodiscard]] unsigned dimension() const noexcept { return dimension_; }

        /** @brief 真实规则精度；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.exactDegree();
         * @endcode */
        [[nodiscard]] unsigned exactDegree() const noexcept { return exactDegree_; }

        /** @brief 米制实际积分点；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.points(); @endcode
         */
        [[nodiscard]] const std::vector<std::array<double, 3>>& points() const noexcept { return points_; }

        /** @brief 含实际测度的正权重；无参数无额外前提。@return 只读值或借用引用。@code auto
         * value=obj.physicalWeights(); @endcode */
        [[nodiscard]] const std::vector<double>& physicalWeights() const noexcept { return physicalWeights_; }

        /** @brief 逐点局部基值；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.traceValues();
         * @endcode */
        [[nodiscard]] const std::vector<double>& traceValues() const noexcept { return traceValues_; }

        /** @brief 单位外法向；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.normals(); @endcode */
        [[nodiscard]] const std::vector<std::array<double, 3>>& normals() const noexcept { return normals_; }

        /** @brief 积分点数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.pointCount(); @endcode */
        [[nodiscard]] std::size_t pointCount() const noexcept { return points_.size(); }

        /** @brief 局部自由度数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.dofCount(); @endcode
         */
        [[nodiscard]] std::size_t dofCount() const noexcept { return parentLocalDofs_.size(); }

    private:
        FacetIntegrationData() = default; ///< 仅积分工厂构造成功候选。
        friend FacetIntegrationData evaluateFacetIntegration(std::shared_ptr<const PreparedGeometry>, std::size_t,
                                                             unsigned);
        std::shared_ptr<const FiniteElementSpace> space_; ///< 空间生命周期。
        std::size_t cellRow_{}; ///< 母单元行。
        std::size_t facetIndex_{}; ///< 侧行。
        mesh::FacetUse use_; ///< 母侧。
        std::vector<std::size_t> parentLocalDofs_; ///< 迹到母局部顺序。
        unsigned dimension_{}; ///< 母元维数。
        unsigned exactDegree_{}; ///< 实际规则精度。
        std::vector<std::array<double, 3>> points_; ///< 实际点坐标。
        std::vector<double> physicalWeights_; ///< 不含厚度的权重。
        std::vector<double> traceValues_; ///< q*nf+a的迹值。
        std::vector<std::array<double, 3>> normals_; ///< 每点外法向。
    };
} // namespace thermofem::fem
