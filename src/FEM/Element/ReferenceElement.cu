/** @file ReferenceElement.cu
 * @brief 实现连续标量P1三角形、四面体的参考布局与解析求值。
 */
#include <FEM/Element/ReferenceElement.cuh>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace emberfield::fem {
    ReferenceElement::ReferenceElement(ElementKey key, unsigned dimension, std::vector<std::array<double, 3>> points,
                                       std::vector<std::size_t> offsets, std::vector<std::size_t> facets) :
        key_(key), dimension_(dimension), interpolationPoints_(std::move(points)), facetOffsets_(std::move(offsets)),
        facetDofs_(std::move(facets)) {}

    std::span<const std::size_t> ReferenceElement::facetLocalDofs(std::size_t localSide) const {
        if (localSide >= facetCount()) {
            throw std::out_of_range("参考单元局部侧号越界");
        }
        return std::span<const std::size_t>(facetDofs_)
            .subspan(facetOffsets_[localSide], facetOffsets_[localSide + 1] - facetOffsets_[localSide]);
    }

    const ReferenceElement& referenceElement(const ElementKey& key) {
        if (key.basisFamily_ != BasisFamily::ContinuousLagrange || key.order_ != 1) {
            throw std::invalid_argument("尚未支持此基族或场插值阶次");
        }
        if (key.family_ == mesh::CellFamily::Triangle) {
            static const ReferenceElement triangle(key, 2, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {0, 2, 4, 6},
                                                   {0, 1, 1, 2, 2, 0});
            return triangle;
        }
        if (key.family_ == mesh::CellFamily::Tetrahedron) {
            static const ReferenceElement tetrahedron(key, 3, {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                                                      {0, 3, 6, 9, 12}, {0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3});
            return tetrahedron;
        }
        throw std::invalid_argument("尚未支持此参考有限元拓扑族");
    }

    BasisTable evaluateBasis(const ElementKey& key, std::span<const std::array<double, 3>> points) {
        const auto& element = referenceElement(key);
        const auto n = element.dofCount();
        const auto d = element.dimension();
        const auto max = std::numeric_limits<std::size_t>::max();
        if (points.size() > max / n || points.size() * n > max / d) {
            throw std::length_error("参考基函数表长度溢出");
        }
        BasisTable result{points.size(), n, d, {}, {}};
        result.values_.resize(points.size() * n);
        result.gradients_.resize(points.size() * n * d);
        for (std::size_t q = 0; q < points.size(); ++q) {
            const auto& p = points[q];
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) || (d == 2 && p[2] != 0)) {
                throw std::invalid_argument("参考坐标必须有限，二维第三分量必须为零");
            }
            // 保留解析线性多项式；不将外插坐标裁剪到参考域。
            double first = 1;
            for (unsigned k = 0; k < d; ++k) {
                first -= p[k];
                result.values_[q * n + k + 1] = p[k];
            }
            if (!std::isfinite(first)) {
                throw std::overflow_error("参考形函数求值溢出");
            }
            result.values_[q * n] = first;
            for (std::size_t a = 0; a < n; ++a) {
                for (unsigned k = 0; k < d; ++k) {
                    result.gradients_[(q * n + a) * d + k] = a == 0 ? -1 : (a == k + 1 ? 1 : 0);
                }
            }
        }
        return result;
    }
} // namespace emberfield::fem
