/** @file Quadrature.cuh
 * @brief 提供含参考域测度的正权单纯形求积规则。
 */
#pragma once
#include <Mesh/Mesh.cuh>

namespace thermofem::fem {
    class QuadratureRule;
    /** @brief 查询完整规则。@param family 参考拓扑族。@param degree 请求精确的总次数0..2。
     * @pre 无。@return 进程生命周期内的只读规则；未知族或degree>2抛invalid_argument。
     * @code auto& r=quadratureRule(mesh::CellFamily::Triangle,2); @endcode */
    [[nodiscard]] const QuadratureRule& quadratureRule(mesh::CellFamily family, unsigned degree);
    /** @brief 不可变参考点权目录，权重已经包含参考长度、面积或体积。
     * @code auto& rule=quadratureRule(mesh::CellFamily::Tetrahedron,2); @endcode */
    class QuadratureRule {
    public:
        /** @brief 查询族；无参数无前提。@return 参考族。@code auto f=r.family(); @endcode */
        [[nodiscard]] mesh::CellFamily family() const noexcept { return family_; }
        /** @brief 查询真实精度；无参数无前提。@return 总次数。@code auto d=r.exactDegree(); @endcode */
        [[nodiscard]] unsigned exactDegree() const noexcept { return exactDegree_; }
        /** @brief 查询点；无参数无前提。@return 只读数组。@code auto& p=r.points(); @endcode */
        [[nodiscard]] const auto& points() const noexcept { return points_; }
        /** @brief 查询正权重；无参数无前提。@return 只读数组。@code auto& w=r.weights(); @endcode */
        [[nodiscard]] const auto& weights() const noexcept { return weights_; }

    private:
        /** @brief 接管目录数据。@param family 族。@param degree 精度。@param points 点。
         * @param weights 权重。@pre 仅目录函数提供已验证常量。@note 无返回值。
         * @code QuadratureRule r(family,2,points,weights); @endcode */
        QuadratureRule(mesh::CellFamily family, unsigned degree, std::vector<std::array<double, 3>> points,
                       std::vector<double> weights) :
            family_(family), exactDegree_(degree), points_(std::move(points)), weights_(std::move(weights)) {}
        friend const QuadratureRule& quadratureRule(mesh::CellFamily, unsigned);
        mesh::CellFamily family_; ///< 参考域族。
        unsigned exactDegree_; ///< 实际多项式精确度。
        std::vector<std::array<double, 3>> points_; ///< 参考坐标。
        std::vector<double> weights_; ///< 含参考测度的权重。
    };
} // namespace thermofem::fem
