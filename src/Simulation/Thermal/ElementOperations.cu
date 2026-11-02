/** @file ElementOperations.cu
 * @brief 以独立体/侧积分表计算热传导局部贡献，保证符号和单位一致。
 */
#include <Simulation/Thermal/ElementOperations.cuh>
#include <cmath>
#include <limits>
#include "LocalResultAccess.cuh"
namespace emberfield::thermal {
    namespace {
        /** @brief 校验厚度因子。@param d 母元维数。@param s 厚度/1。@pre 无。
         * @note 无返回值；非法参数invalid_argument。@code section(2,s); @endcode */
        void section(unsigned d, double s) {
            if (!std::isfinite(s) || s <= 0 || (d == 3 && s != 1) || (d != 2 && d != 3)) {
                throw std::invalid_argument("二维厚度必须正有限，三维sectionScale必须为1");
            }
        }
        /** @brief 校验系数采样。@param v 样本。@param q 点数。@param range 0有符号/1非负/2正。
         * @pre 无。@note 无返回值；非法形状/值invalid_argument。@code samples(k,q,2); @endcode */
        void samples(std::span<const double> v, std::size_t q, int range) {
            if (v.size() != q || !q) {
                throw std::invalid_argument("系数样本与积分点数不一致");
            }
            for (auto x : v) {
                if (!std::isfinite(x) || (range == 1 && x < 0) || (range == 2 && x <= 0)) {
                    throw std::invalid_argument("系数样本非有限或物理范围非法");
                }
            }
        }
        /** @brief 检查并恢复局部数值。@param x 扩展精度候选。@param cell 身份。
         * @param side 可选侧。@param q 点序。@param positive 是否要求有效正尺度。
         * @pre 无。@return 有限结果；失败EvaluationError。@code auto v=number(x,c,{},q); @endcode */
        double number(long double x, mesh::CellRef cell, std::optional<std::size_t> side, std::size_t q,
                      bool positive = false) {
            const double result = static_cast<double>(x);
            if (!std::isfinite(result) || (positive && result <= 0)) {
                throw fem::EvaluationError(result == 0 ? fem::EvaluationCode::NumericalUnderflow
                                                       : fem::EvaluationCode::NumericalOverflow,
                                           fem::EvaluationStage::Element, "局部热积分乘加不可表示", cell, side, q);
            }
            return result;
        }
        /** @brief 镜像上三角。@param m 矩阵。@param n 维数。@pre 形状正确。
         * @note 原地修改，无返回值。@code mirror(m,n); @endcode */
        void mirror(std::vector<double>& m, std::size_t n) {
            for (std::size_t a = 0; a < n; ++a) {
                for (std::size_t b = a + 1; b < n; ++b) {
                    m[b * n + a] = m[a * n + b];
                }
            }
        }
    } // namespace
    LocalMatrix integrateConductivity(const fem::CellIntegrationData& t, std::span<const double> k, double s) {
        section(t.dimension(), s);
        samples(k, t.pointCount(), 2);
        const auto n = t.dofCount();
        const auto d = t.dimension();
        const auto cell = t.space()->cells()[t.cellRow()];
        std::vector<double> out(n * n, 0);
        bool nonzero = false;
        for (std::size_t q = 0; q < t.pointCount(); ++q) {
            const long double scale = static_cast<long double>(s) * t.physicalWeights()[q] * k[q];
            for (std::size_t a = 0; a < n; ++a) {
                for (std::size_t b = a; b < n; ++b) {
                    long double dot = 0;
                    for (unsigned i = 0; i < d; ++i) {
                        dot += static_cast<long double>(t.gradients()[(q * n + a) * d + i]) *
                            t.gradients()[(q * n + b) * d + i];
                    }
                    const double term = number(scale * dot, cell, {}, q);
                    out[a * n + b] = number(static_cast<long double>(out[a * n + b]) + term, cell, {}, q);
                    nonzero |= term != 0;
                }
            }
        }
        if (!nonzero) {
            throw fem::EvaluationError(fem::EvaluationCode::NumericalUnderflow, fem::EvaluationStage::Element,
                                       "导热刚度整体消失", cell);
        }
        mirror(out, n);
        return detail::LocalResultAccess::matrix(n, std::move(out));
    }
    LocalMatrix integrateCapacity(const fem::CellIntegrationData& t, std::span<const double> c, double s) {
        section(t.dimension(), s);
        samples(c, t.pointCount(), 2);
        if (t.exactDegree() < 2) {
            throw std::invalid_argument("一致容量需要二次精确规则");
        }
        const auto n = t.dofCount();
        const auto cell = t.space()->cells()[t.cellRow()];
        std::vector<double> out(n * n, 0);
        bool nonzero = false;
        for (std::size_t q = 0; q < t.pointCount(); ++q) {
            const long double scale = static_cast<long double>(s) * t.physicalWeights()[q] * c[q];
            for (std::size_t a = 0; a < n; ++a) {
                for (std::size_t b = a; b < n; ++b) {
                    const double term = number(scale * t.values()[q * n + a] * t.values()[q * n + b], cell, {}, q);
                    out[a * n + b] = number(static_cast<long double>(out[a * n + b]) + term, cell, {}, q);
                    nonzero |= term != 0;
                }
            }
        }
        if (!nonzero) {
            throw fem::EvaluationError(fem::EvaluationCode::NumericalUnderflow, fem::EvaluationStage::Element,
                                       "一致容量整体消失", cell);
        }
        mirror(out, n);
        return detail::LocalResultAccess::matrix(n, std::move(out));
    }
    LocalVector integrateVolumeSource(const fem::CellIntegrationData& t, std::span<const double> source, double s) {
        section(t.dimension(), s);
        samples(source, t.pointCount(), 0);
        if (t.exactDegree() < 1) {
            throw std::invalid_argument("体源需要一次精确规则");
        }
        const auto n = t.dofCount();
        const auto cell = t.space()->cells()[t.cellRow()];
        std::vector<double> out(n, 0);
        for (std::size_t q = 0; q < t.pointCount(); ++q) {
            for (std::size_t a = 0; a < n; ++a) {
                const long double term =
                    static_cast<long double>(s) * t.physicalWeights()[q] * source[q] * t.values()[q * n + a];
                out[a] = number(static_cast<long double>(out[a]) + number(term, cell, {}, q), cell, {}, q);
            }
        }
        return detail::LocalResultAccess::vector(n, std::move(out));
    }
    LocalVector integrateOutwardFlux(const fem::FacetIntegrationData& t, std::span<const double> flux, double s) {
        section(t.dimension(), s);
        samples(flux, t.pointCount(), 0);
        if (t.exactDegree() < 1) {
            throw std::invalid_argument("热流载荷需要一次精确规则");
        }
        const auto n = t.dofCount();
        std::vector<double> out(n, 0);
        for (std::size_t q = 0; q < t.pointCount(); ++q) {
            for (std::size_t a = 0; a < n; ++a) {
                const long double term =
                    -static_cast<long double>(s) * t.physicalWeights()[q] * flux[q] * t.traceValues()[q * n + a];
                out[a] = number(static_cast<long double>(out[a]) + number(term, t.use().cell_, t.use().localSide_, q),
                                t.use().cell_, t.use().localSide_, q);
            }
        }
        return detail::LocalResultAccess::vector(n, std::move(out));
    }
    ConvectionContribution integrateConvection(const fem::FacetIntegrationData& t, std::span<const double> h,
                                               std::span<const double> ambient, double s) {
        section(t.dimension(), s);
        samples(h, t.pointCount(), 1);
        samples(ambient, t.pointCount(), 1);
        if (t.exactDegree() < 2) {
            throw std::invalid_argument("对流矩阵需要二次精确规则");
        }
        const auto n = t.dofCount();
        std::vector<double> matrix(n * n, 0), load(n, 0);
        bool positive = false, nonzero = false;
        for (std::size_t q = 0; q < t.pointCount(); ++q) {
            positive |= h[q] > 0;
            const long double scale = static_cast<long double>(s) * t.physicalWeights()[q] * h[q];
            for (std::size_t a = 0; a < n; ++a) {
                const double f =
                    number(scale * ambient[q] * t.traceValues()[q * n + a], t.use().cell_, t.use().localSide_, q);
                load[a] = number(static_cast<long double>(load[a]) + f, t.use().cell_, t.use().localSide_, q);
                for (std::size_t b = a; b < n; ++b) {
                    const double term = number(scale * t.traceValues()[q * n + a] * t.traceValues()[q * n + b],
                                               t.use().cell_, t.use().localSide_, q);
                    matrix[a * n + b] = number(static_cast<long double>(matrix[a * n + b]) + term, t.use().cell_,
                                               t.use().localSide_, q);
                    nonzero |= term != 0;
                }
            }
        }
        if (positive && !nonzero) {
            throw fem::EvaluationError(fem::EvaluationCode::NumericalUnderflow, fem::EvaluationStage::Element,
                                       "正对流矩阵整体消失", t.use().cell_, t.use().localSide_);
        }
        mirror(matrix, n);
        return detail::LocalResultAccess::convection(detail::LocalResultAccess::matrix(n, std::move(matrix)),
                                                     detail::LocalResultAccess::vector(n, std::move(load)));
    }
} // namespace emberfield::thermal
