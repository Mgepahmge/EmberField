/** @file Geometry.cuh
 * @brief 定义米制仿射几何与全活动域几何准备，不包含物理参数。
 */
#pragma once
#include <FiniteElement/EvaluationError.cuh>
#include <FiniteElement/Space.cuh>
#include <limits>

namespace thermofem::fem {
    /** @brief 坐标解释，不把任意嵌入曲面自动当平面。 */
    enum class GeometryFrame { PlanarXY, SolidXYZ };
    /** @brief 坐标与数值质量选项；入口统一验证。
     * @code GeometryOptions o; o.coordinateScale_=1e-3; @endcode */
    struct GeometryOptions {
        GeometryFrame frame_{GeometryFrame::PlanarXY}; ///< XY平面或三维。
        double coordinateScale_{1}; ///< 原坐标到米的正比例。
        double planarTolerance_{1e-12}; ///< 二维z容差，米。
        double detTolerance_{128 * std::numeric_limits<double>::epsilon()}; ///< 无量纲退化门槛。
        double maxCondition_{1e12}; ///< Frobenius条件数上限。
        double normalTolerance_{1e-10}; ///< 相邻法向反向残差上限。
    };
    /** @brief 校验数值策略。@param options 选项。@param dimension 空间维数。
     * @pre 无。@note 无返回值；非法选项抛invalid_argument。
     * @code validateGeometryOptions(o,2); @endcode */
    void validateGeometryOptions(const GeometryOptions& options, unsigned dimension);
    class AffineCellGeometry;
    class FacetGeometry;
    class PreparedGeometry;
    /** @brief 构建单元映射。@param space 空间。@param cellRow 活动行。@param options 几何策略。
     * @pre 空间不可经别名修改。@return 拥有米制几何的值。
     * @throws std::invalid_argument 策略错误；out_of_range行错误；EvaluationError数值/几何失败。
     * @code auto g=buildCellGeometry(space,0,o); @endcode */
    [[nodiscard]] AffineCellGeometry buildCellGeometry(const FiniteElementSpace& space, std::size_t cellRow,
                                                       const GeometryOptions& options);
    /** @brief 构建母单元侧及外向。@param space 空间。@param cell 母单元几何。
     * @param localSide 局部侧。@param options 原几何策略。@pre cell必须来自相同space与options。
     * @return 有序侧几何；非法参数/越界抛标准异常，数值失败抛EvaluationError。
     * @code auto f=buildFacetGeometry(space,g,1,o); @endcode */
    [[nodiscard]] FacetGeometry buildFacetGeometry(const FiniteElementSpace& space, const AffineCellGeometry& cell,
                                                   std::size_t localSide, const GeometryOptions& options);
    /** @brief 事务性准备活动域几何。@param space 共享空间。@param options 坐标策略。
     * @param maxCells 单元上限。@param maxFacetUses 侧use上限。@pre 正限额且空间不变。
     * @return 完整不可变几何；首个几何失败抛EvaluationError，无半成功对象。
     * @code auto g=buildPreparedGeometry(space,o); @endcode */
    [[nodiscard]] std::shared_ptr<const PreparedGeometry>
    buildPreparedGeometry(std::shared_ptr<const FiniteElementSpace> space, const GeometryOptions& options,
                          std::size_t maxCells = 10000000, std::size_t maxFacetUses = 40000000);
    /** @brief 一个活动单纯形的已验证仿射映射，负方向不改变局部节点序。
     * @code auto g=buildCellGeometry(space,0,o); auto v=g.measure(); @endcode */
    class AffineCellGeometry {
    public:
        /** @brief 母单元身份；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.cell(); @endcode */
        [[nodiscard]] mesh::CellRef cell() const noexcept { return cell_; }

        /** @brief 维数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.dimension(); @endcode */
        [[nodiscard]] unsigned dimension() const noexcept { return dimension_; }

        /** @brief 米制有序顶点；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.vertices(); @endcode
         */
        [[nodiscard]] const std::vector<std::array<double, 3>>& vertices() const noexcept { return vertices_; }

        /** @brief 行主序J；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.jacobian(); @endcode */
        [[nodiscard]] const std::vector<double>& jacobian() const noexcept { return jacobian_; }

        /** @brief 行主序J逆；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.inverseJacobian();
         * @endcode */
        [[nodiscard]] const std::vector<double>& inverseJacobian() const noexcept { return inverseJacobian_; }

        /** @brief 最大边长；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.edgeScale(); @endcode */
        [[nodiscard]] double edgeScale() const noexcept { return edgeScale_; }

        /** @brief 归一化有向行列式；无参数无额外前提。@return 只读值或借用引用。@code auto
         * value=obj.normalizedDeterminant(); @endcode */
        [[nodiscard]] double normalizedDeterminant() const noexcept { return normalizedDeterminant_; }

        /** @brief 物理行列式绝对值；无参数无额外前提。@return 只读值或借用引用。@code auto
         * value=obj.absoluteDeterminant(); @endcode */
        [[nodiscard]] double absoluteDeterminant() const noexcept { return absoluteDeterminant_; }

        /** @brief 面积或体积；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.measure(); @endcode */
        [[nodiscard]] double measure() const noexcept { return measure_; }

        /** @brief 映射符号；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.orientation(); @endcode */
        [[nodiscard]] int orientation() const noexcept { return orientation_; }

        /** @brief Frobenius条件数；无参数无额外前提。@return 只读值或借用引用。@code auto
         * value=obj.conditionEstimate(); @endcode */
        [[nodiscard]] double conditionEstimate() const noexcept { return conditionEstimate_; }

    private:
        AffineCellGeometry() = default; ///< 工厂候选，无公开非法状态。
        friend AffineCellGeometry buildCellGeometry(const FiniteElementSpace&, std::size_t, const GeometryOptions&);
        mesh::CellRef cell_; ///< 母单元身份。
        unsigned dimension_{}; ///< 维数。
        std::vector<std::array<double, 3>> vertices_; ///< 原顺序的米制顶点。
        std::vector<double> jacobian_; ///< 行主序J。
        std::vector<double> inverseJacobian_; ///< 行主序J逆。
        double edgeScale_{}; ///< 最大边长。
        double normalizedDeterminant_{}; ///< det(J/L)。
        double absoluteDeterminant_{}; ///< |det J|。
        double measure_{}; ///< 面积或体积。
        int orientation_{}; ///< +1或-1。
        double conditionEstimate_{}; ///< Frobenius条件数。
    };
    /** @brief 按母元侧序保存正测度和外法向，不携带物理厚度。
     * @code auto f=buildFacetGeometry(space,cell,0,o); @endcode */
    class FacetGeometry {
    public:
        /** @brief 母元局部侧；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.use(); @endcode */
        [[nodiscard]] mesh::FacetUse use() const noexcept { return use_; }

        /** @brief 母单元维数；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.dimension(); @endcode */
        [[nodiscard]] unsigned dimension() const noexcept { return dimension_; }

        /** @brief 迹到母局部映射；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.parentLocalDofs();
         * @endcode */
        [[nodiscard]] const std::vector<std::size_t>& parentLocalDofs() const noexcept { return parentLocalDofs_; }

        /** @brief 有序米制侧顶点；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.vertices(); @endcode
         */
        [[nodiscard]] const std::vector<std::array<double, 3>>& vertices() const noexcept { return vertices_; }

        /** @brief 单位外法向；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.normal(); @endcode */
        [[nodiscard]] const std::array<double, 3>& normal() const noexcept { return normal_; }

        /** @brief 侧Jacobian测度；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.surfaceJacobian();
         * @endcode */
        [[nodiscard]] double surfaceJacobian() const noexcept { return surfaceJacobian_; }

        /** @brief 长度或面积；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.measure(); @endcode */
        [[nodiscard]] double measure() const noexcept { return measure_; }

    private:
        FacetGeometry() = default; ///< 仅工厂候选。
        friend FacetGeometry buildFacetGeometry(const FiniteElementSpace&, const AffineCellGeometry&, std::size_t,
                                                const GeometryOptions&);
        mesh::FacetUse use_; ///< 母元局部侧。
        unsigned dimension_{}; ///< 母元维数。
        std::vector<std::size_t> parentLocalDofs_; ///< 有序迹到母局部映射。
        std::vector<std::array<double, 3>> vertices_; ///< 有序侧顶点。
        std::array<double, 3> normal_{}; ///< 外法向。
        double surfaceJacobian_{}; ///< 参考侧到物理侧测度比。
        double measure_{}; ///< 长度或面积。
    };
    /** @brief 组合共享空间、单元仿射记录和外侧几何；不保存全部积分点。
     * @code auto p=buildPreparedGeometry(space,o); @endcode */
    class PreparedGeometry {
    public:
        /** @brief 共享空间；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.space(); @endcode */
        [[nodiscard]] const std::shared_ptr<const FiniteElementSpace>& space() const noexcept { return space_; }

        /** @brief 几何策略快照；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.options(); @endcode */
        [[nodiscard]] const GeometryOptions& options() const noexcept { return options_; }

        /** @brief 按空间行的单元记录；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.cells();
         * @endcode */
        [[nodiscard]] const std::vector<AffineCellGeometry>& cells() const noexcept { return cells_; }

        /** @brief 独立侧拓扑；无参数无额外前提。@return 只读值或借用引用。@code auto value=obj.topology(); @endcode */
        [[nodiscard]] const mesh::FacetTopology& topology() const noexcept { return topology_; }

        /** @brief 查询一个外侧。@param index 拓扑侧行。@pre 无。
         * @return 只读侧记录；越界out_of_range，内部侧invalid_argument。
         * @code auto& side=p.exteriorFacet(i); @endcode */
        [[nodiscard]] const FacetGeometry& exteriorFacet(std::size_t index) const;

    private:
        PreparedGeometry() = default; ///< 仅工厂候选。
        friend std::shared_ptr<const PreparedGeometry> buildPreparedGeometry(std::shared_ptr<const FiniteElementSpace>,
                                                                             const GeometryOptions&, std::size_t,
                                                                             std::size_t);
        std::shared_ptr<const FiniteElementSpace> space_; ///< 不可变空间所有权。
        GeometryOptions options_; ///< 统一策略。
        std::vector<AffineCellGeometry> cells_; ///< 活动行几何。
        mesh::FacetTopology topology_; ///< 活动侧拓扑。
        std::vector<std::optional<FacetGeometry>> exterior_; ///< 仅外侧槽有记录。
    };
} // namespace thermofem::fem
