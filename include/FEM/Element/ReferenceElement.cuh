/** @file ReferenceElement.cuh
 * @brief 定义独立于实际几何、物理算例和积分规则的标量参考有限元。
 */
#pragma once
#include <DataStructure/Mesh/Mesh.cuh>

/** @brief 提供参考有限元、形函数和不施加物理约束的自由度空间。 */
namespace emberfield::fem {
    /** @brief 基函数族；当前仅支持连续节点型Lagrange。 */
    enum class BasisFamily { ContinuousLagrange };
    /** @brief 有限元键；温度阶次不等于网格几何阶次。
     * @code ElementKey key{mesh::CellFamily::Triangle}; @endcode */
    struct ElementKey {
        mesh::CellFamily family_{mesh::CellFamily::Triangle}; ///< 参考拓扑族。
        BasisFamily basisFamily_{BasisFamily::ContinuousLagrange}; ///< 插值族。
        unsigned order_{1}; ///< 场的插值阶次。
        bool operator==(const ElementKey&) const = default; ///< 完整键值比较。
    };
    /** @brief 不可变的参考有限元布局；不包含实际坐标或积分点。
     * @details 插值自由度顺序遵循网格complete布局的顶点顺序。侧顺序仅为参数化约定，
     * 不代表实际外法向。目录对象具有进程寿命；不允许调用者构造无效目录项。
     * @code const auto& e=referenceElement({mesh::CellFamily::Triangle}); @endcode */
    class ReferenceElement {
    public:
        /** @brief 获取键；无参数无额外前提。@return 键值。
         * @code auto k=e.key(); @endcode */
        [[nodiscard]] ElementKey key() const noexcept { return key_; }
        /** @brief 获取参考维数；无参数无额外前提。@return 2或3。
         * @code auto d=e.dimension(); @endcode */
        [[nodiscard]] unsigned dimension() const noexcept { return dimension_; }
        /** @brief 获取局部DOF数；无参数无额外前提。@return 3或4。
         * @code auto n=e.dofCount(); @endcode */
        [[nodiscard]] std::size_t dofCount() const noexcept { return interpolationPoints_.size(); }
        /** @brief 借用参考插值节点；无参数无额外前提。@return 与DOF同序的只读视图。
         * @code auto p=e.interpolationPoints(); @endcode */
        [[nodiscard]] std::span<const std::array<double, 3>> interpolationPoints() const noexcept {
            return interpolationPoints_;
        }
        /** @brief 获取局部侧数；无参数无额外前提。@return 3或4。
         * @code auto n=e.facetCount(); @endcode */
        [[nodiscard]] std::size_t facetCount() const noexcept { return facetOffsets_.size() - 1; }
        /** @brief 借用侧上的有序母单元DOF。
         * @param localSide 局部侧号。@pre localSide小于facetCount。
         * @return 目录寿命内有效的视图；越界抛out_of_range。
         * @code auto edge=e.facetLocalDofs(2); @endcode */
        [[nodiscard]] std::span<const std::size_t> facetLocalDofs(std::size_t localSide) const;

    private:
        /** @brief 创建固定的有效目录项。
         * @param key 有限元键。@param dimension 参考维数。@param points 插值节点。
         * @param offsets 侧行边界。@param facets 有序侧DOF。@pre 仅目录函数调用，所有布局合法。
         * @note 无返回值；标准分配异常传播。@code ReferenceElement e(key,2,points,offsets,facets); @endcode */
        ReferenceElement(ElementKey key, unsigned dimension, std::vector<std::array<double, 3>> points,
                         std::vector<std::size_t> offsets, std::vector<std::size_t> facets);
        friend const ReferenceElement& referenceElement(const ElementKey&);
        ElementKey key_; ///< 完整有限元键。
        unsigned dimension_; ///< 参考维数。
        std::vector<std::array<double, 3>> interpolationPoints_; ///< 有序插值节点。
        std::vector<std::size_t> facetOffsets_; ///< 侧布局的行边界。
        std::vector<std::size_t> facetDofs_; ///< 各侧有序母单元DOF。
    };
    /** @brief 独立拥有的批量参考基函数值与梯度。
     * @code auto table=evaluateBasis(key,points); @endcode */
    struct BasisTable {
        std::size_t pointCount_{}; ///< 点数；允许为0。
        std::size_t dofCount_{}; ///< 每点的基函数数。
        unsigned dimension_{}; ///< 每梯度的分量数。
        std::vector<double> values_; ///< values[q*dofCount+a]。
        std::vector<double> gradients_; ///< gradients[(q*dofCount+a)*dimension+k]，参考梯度。
    };
    /** @brief 查询完整有限元键；不检查实际几何兼容性。
     * @param key 已注册的Triangle-P1或Tet-P1键。@pre 无。
     * @return 进程寿命的常量目录引用；未知键抛invalid_argument，资源异常传播。
     * @code const auto& e=referenceElement(key); @endcode */
    [[nodiscard]] const ReferenceElement& referenceElement(const ElementKey& key);
    /** @brief 计算参考基函数及参考梯度，不选择积分点或变换到物理坐标。
     * @param key 已注册的有限元键。@param points 三分量参考坐标，二维第三分量须为0。
     * @pre 坐标有限，允许域外外插；空输入合法。
     * @return 独立拥有的表；不借用输入，无设备同步和可变缓存。
     * @throws std::invalid_argument 未知键或非法坐标；length_error表长度溢出；
     * overflow_error有限坐标导致求值溢出；分配异常传播，失败不发布部分表。
     * @code auto table=evaluateBasis(key,referenceElement(key).interpolationPoints()); @endcode */
    [[nodiscard]] BasisTable evaluateBasis(const ElementKey& key, std::span<const std::array<double, 3>> points);
} // namespace emberfield::fem
