/** @file Integration.cu
 * @brief 实现参考基与米制仿射映射的组合，严格区分体与侧。
 */
#include <FEM/Element/Integration.cuh>
#include <cmath>
#include <limits>

namespace thermofem::fem {
    namespace {
        /** @brief 检查表长度乘法。@param a 行数。@param b 列数。@pre 无。
         * @return 乘积；溢出抛length_error。@code auto n=product(q,n); @endcode */
        std::size_t product(std::size_t a, std::size_t b) {
            if (b && a > std::numeric_limits<std::size_t>::max() / b) {
                throw std::length_error("积分表大小溢出");
            }
            return a * b;
        }
        /** @brief 校验实际量有限且正权不消失。@param value 量。@param cell 母单元。
         * @param side 侧。@param q 积分点。@param positive 是否为正权。
         * @pre 无。@return 有限量，否则EvaluationError。@code auto w=checked(w,c,{},q,true); @endcode */
        double checked(long double value, mesh::CellRef cell, std::optional<std::size_t> side, std::size_t q,
                       bool positive = false) {
            const double v = static_cast<double>(value);
            if (!std::isfinite(v) || (positive && v <= 0)) {
                throw EvaluationError(positive && v == 0 ? EvaluationCode::NumericalUnderflow
                                                         : EvaluationCode::NonfiniteIntegration,
                                      EvaluationStage::Integration, "实际积分量不可表示", cell, side, q);
            }
            return v;
        }
    } // namespace
    CellIntegrationData evaluateCellIntegration(std::shared_ptr<const PreparedGeometry> geometry, std::size_t row,
                                                unsigned degree) {
        if (!geometry) {
            throw std::invalid_argument("几何对象不能为空");
        }
        const auto& g = geometry->cells().at(row);
        const auto key = geometry->space()->elementKey(row);
        const auto& rule = quadratureRule(key.family_, degree);
        auto basis = evaluateBasis(key, rule.points());
        CellIntegrationData t;
        t.space_ = geometry->space();
        t.cellRow_ = row;
        t.dimension_ = g.dimension();
        t.exactDegree_ = rule.exactDegree();
        const auto d = t.dimension_;
        const auto n = basis.dofCount_;
        const auto qCount = rule.points().size();
        t.points_.resize(qCount);
        t.physicalWeights_.resize(qCount);
        if (basis.values_.size() != product(qCount, n) || basis.gradients_.size() != product(product(qCount, n), d)) {
            throw std::logic_error("参考基表形状不一致");
        }
        t.values_ = std::move(basis.values_);
        t.gradients_.resize(product(product(qCount, n), d));
        for (std::size_t q = 0; q < qCount; ++q) {
            for (unsigned i = 0; i < d; ++i) {
                long double x = g.vertices()[0][i];
                for (unsigned j = 0; j < d; ++j) {
                    x += static_cast<long double>(g.jacobian()[i * d + j]) * rule.points()[q][j];
                }
                t.points_[q][i] = checked(x, g.cell(), {}, q);
            }
            t.physicalWeights_[q] =
                checked(static_cast<long double>(rule.weights()[q]) * g.absoluteDeterminant(), g.cell(), {}, q, true);
            for (std::size_t a = 0; a < n; ++a) {
                for (unsigned i = 0; i < d; ++i) {
                    long double value = 0;
                    for (unsigned j = 0; j < d; ++j) {
                        value += static_cast<long double>(g.inverseJacobian()[j * d + i]) *
                            basis.gradients_[(q * n + a) * d + j];
                    }
                    t.gradients_[(q * n + a) * d + i] = checked(value, g.cell(), {}, q);
                }
            }
        }
        return t;
    }
    FacetIntegrationData evaluateFacetIntegration(std::shared_ptr<const PreparedGeometry> geometry, std::size_t index,
                                                  unsigned degree) {
        if (!geometry) {
            throw std::invalid_argument("几何对象不能为空");
        }
        const auto& f = geometry->exteriorFacet(index);
        const auto& rule =
            quadratureRule(f.dimension() == 2 ? mesh::CellFamily::Segment : mesh::CellFamily::Triangle, degree);
        FacetIntegrationData t;
        t.space_ = geometry->space();
        t.use_ = f.use();
        t.facetIndex_ = index;
        const auto row = t.space_->cellPosition(t.use_.cell_);
        if (!row) {
            throw std::logic_error("外侧母单元不属于空间");
        }
        t.cellRow_ = *row;
        t.dimension_ = f.dimension();
        t.exactDegree_ = rule.exactDegree();
        t.parentLocalDofs_ = f.parentLocalDofs();
        const auto n = t.parentLocalDofs_.size(), qCount = rule.points().size();
        t.points_.resize(qCount);
        t.physicalWeights_.resize(qCount);
        t.normals_.assign(qCount, f.normal());
        t.traceValues_.resize(product(qCount, n));
        for (std::size_t q = 0; q < qCount; ++q) {
            const auto& p = rule.points()[q];
            std::array<double, 3> values{1 - p[0] - (n == 3 ? p[1] : 0), p[0], n == 3 ? p[1] : 0};
            for (std::size_t a = 0; a < n; ++a) {
                t.traceValues_[q * n + a] = values[a];
            }
            for (unsigned i = 0; i < t.dimension_; ++i) {
                long double x = 0;
                for (std::size_t a = 0; a < n; ++a) {
                    x += static_cast<long double>(values[a]) * f.vertices()[a][i];
                }
                t.points_[q][i] = checked(x, t.use_.cell_, t.use_.localSide_, q);
            }
            t.physicalWeights_[q] = checked(static_cast<long double>(rule.weights()[q]) * f.surfaceJacobian(),
                                            t.use_.cell_, t.use_.localSide_, q, true);
        }
        return t;
    }
} // namespace thermofem::fem
