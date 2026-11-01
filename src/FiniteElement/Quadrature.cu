/** @file Quadrature.cu
 * @brief 实现区间、三角形和四面体的一次与二次求积目录。
 */
#include <FiniteElement/Quadrature.cuh>
#include <cmath>
#include <stdexcept>
namespace thermofem::fem {
    const QuadratureRule& quadratureRule(mesh::CellFamily family, unsigned degree) {
        using F = mesh::CellFamily;
        if (degree > 2 || (family != F::Segment && family != F::Triangle && family != F::Tetrahedron)) {
            throw std::invalid_argument("求积族或请求精确度不支持");
        }
        static const QuadratureRule line1(F::Segment, 1, {{0.5, 0, 0}}, {1});
        static const QuadratureRule tri1(F::Triangle, 1, {{1.0 / 3, 1.0 / 3, 0}}, {0.5});
        static const QuadratureRule tet1(F::Tetrahedron, 1, {{0.25, 0.25, 0.25}}, {1.0 / 6});
        const double delta = 1 / std::sqrt(3.0);
        static const QuadratureRule line2(F::Segment, 2, {{(1 - delta) / 2, 0, 0}, {(1 + delta) / 2, 0, 0}},
                                          {0.5, 0.5});
        static const QuadratureRule tri2(F::Triangle, 2,
                                         {{1.0 / 6, 1.0 / 6, 0}, {2.0 / 3, 1.0 / 6, 0}, {1.0 / 6, 2.0 / 3, 0}},
                                         {1.0 / 6, 1.0 / 6, 1.0 / 6});
        const double a = (5 + 3 * std::sqrt(5.0)) / 20, b = (5 - std::sqrt(5.0)) / 20;
        static const QuadratureRule tet2(F::Tetrahedron, 2, {{b, b, b}, {a, b, b}, {b, a, b}, {b, b, a}},
                                         {1.0 / 24, 1.0 / 24, 1.0 / 24, 1.0 / 24});
        if (family == F::Segment) {
            return degree <= 1 ? line1 : line2;
        }
        if (family == F::Triangle) {
            return degree <= 1 ? tri1 : tri2;
        }
        return degree <= 1 ? tet1 : tet2;
    }
} // namespace thermofem::fem
