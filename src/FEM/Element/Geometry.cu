/** @file Geometry.cu
 * @brief 实现尺度归一化带主元映射、母单元外向和相邻侧检查。
 */
#include <FEM/Element/Geometry.cuh>
#include <NVTXProfiler.cuh>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace emberfield::fem {
    namespace {
        using Point = std::array<double, 3>;
        /** @brief 统一抛出几何失败。@param code 代码。@param message 描述。@param cell 单元。
         * @param side 可选侧。@pre 无。@note 无返回；总是抛异常。
         * @code fail(code,"退化",cell); @endcode */
        [[noreturn]] void fail(EvaluationCode code, const char* message, mesh::CellRef cell,
                               std::optional<std::size_t> side = {}) {
            throw EvaluationError(code, EvaluationStage::Geometry, message, cell, side);
        }
        /** @brief 向量差。@param a 被减数。@param b 减数。@pre 无。
         * @return 三分量差。@code auto d=difference(a,b); @endcode */
        Point difference(const Point& a, const Point& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }
        /** @brief 稳定长度。@param p 向量。@pre 无。@return 欧氏范数。
         * @code auto n=norm(p); @endcode */
        double norm(const Point& p) { return std::hypot(p[0], p[1], p[2]); }
        /** @brief 内积。@param a 向量。@param b 向量。@pre 有限且乘加可表示。
         * @return 内积。@code auto s=dot(a,b); @endcode */
        double dot(const Point& a, const Point& b) { return std::fma(a[0], b[0], std::fma(a[1], b[1], a[2] * b[2])); }
        /** @brief 从扩展精度恢复正double测度，避免非必要的中间范围问题。
         * @param value 正测度。@param cell 身份。@param side 可选侧。@pre 无。
         * @return 有限正double，否则抛EvaluationError。@code auto v=positive(value,cell); @endcode */
        double positive(long double value, mesh::CellRef cell, std::optional<std::size_t> side = {}) {
            const auto result = static_cast<double>(value);
            if (!std::isfinite(result)) {
                fail(EvaluationCode::NumericalOverflow, "几何测度溢出", cell, side);
            }
            if (!(result > 0)) {
                fail(EvaluationCode::NumericalUnderflow, "正几何测度不可表示", cell, side);
            }
            return result;
        }
    } // namespace
    void validateGeometryOptions(const GeometryOptions& o, unsigned d) {
        if ((d != 2 && d != 3) || (o.frame_ != GeometryFrame::PlanarXY && o.frame_ != GeometryFrame::SolidXYZ) ||
            (d == 2) != (o.frame_ == GeometryFrame::PlanarXY) || !std::isfinite(o.coordinateScale_) ||
            o.coordinateScale_ <= 0 || !std::isfinite(o.planarTolerance_) || o.planarTolerance_ < 0 ||
            !std::isfinite(o.detTolerance_) || o.detTolerance_ <= 0 || o.detTolerance_ >= 1 ||
            !std::isfinite(o.maxCondition_) || o.maxCondition_ < d || !std::isfinite(o.normalTolerance_) ||
            o.normalTolerance_ <= 0 || o.normalTolerance_ >= 1) {
            throw std::invalid_argument("几何策略与空间维数不相容或数值阈值非法");
        }
    }
    AffineCellGeometry buildCellGeometry(const FiniteElementSpace& s, std::size_t row, const GeometryOptions& o) {
        validateGeometryOptions(o, s.dimension());
        if (row >= s.cellCount()) {
            throw std::out_of_range("活动单元行越界");
        }
        AffineCellGeometry g;
        g.cell_ = s.cells()[row];
        g.dimension_ = s.dimension();
        const auto d = g.dimension_;
        const auto nodes = s.mesh().connectivity(g.cell_.block_, g.cell_.element_);
        if (nodes.size() != d + 1 || referenceElement(s.elementKey(row)).dofCount() != d + 1) {
            throw std::invalid_argument("仿射几何仅支持Triangle3/Tet4");
        }
        for (auto node : nodes) {
            Point p = s.mesh().data().nodes_.at(node).position_;
            for (auto& x : p) {
                x *= o.coordinateScale_;
                if (!std::isfinite(x)) {
                    fail(EvaluationCode::CoordinateRange, "米制坐标非有限", g.cell_);
                }
            }
            if (d == 2) {
                if (std::abs(p[2]) > o.planarTolerance_) {
                    fail(EvaluationCode::NonplanarCell, "节点超出XY平面容差", g.cell_);
                }
                p[2] = 0;
            }
            g.vertices_.push_back(p);
        }
        for (std::size_t a = 0; a < g.vertices_.size(); ++a) {
            for (std::size_t b = 0; b < a; ++b) {
                auto edge = difference(g.vertices_[a], g.vertices_[b]);
                const double length = norm(edge);
                if (!std::isfinite(length)) {
                    fail(EvaluationCode::CoordinateRange, "边长不可表示", g.cell_);
                }
                g.edgeScale_ = std::max(g.edgeScale_, length);
            }
        }
        if (g.edgeScale_ == 0) {
            fail(EvaluationCode::DegenerateCell, "单元全部顶点重合", g.cell_);
        }
        g.jacobian_.resize(d * d);
        std::array<double, 9> lu{}, inv{};
        std::array<unsigned, 3> permutation{0, 1, 2};
        for (unsigned i = 0; i < d; ++i) {
            for (unsigned j = 0; j < d; ++j) {
                const double value = g.vertices_[j + 1][i] - g.vertices_[0][i];
                g.jacobian_[i * d + j] = value;
                lu[i * d + j] = value / g.edgeScale_;
            }
        }
        double det = 1;
        // 对归一化矩阵作部分主元LU；单元原局部顺序不随主元交换变化。
        for (unsigned k = 0; k < d; ++k) {
            unsigned pivot = k;
            for (unsigned i = k + 1; i < d; ++i) {
                if (std::abs(lu[i * d + k]) > std::abs(lu[pivot * d + k])) {
                    pivot = i;
                }
            }
            if (lu[pivot * d + k] == 0) {
                fail(EvaluationCode::DegenerateCell, "单元Jacobian奇异", g.cell_);
            }
            if (pivot != k) {
                for (unsigned j = 0; j < d; ++j) {
                    std::swap(lu[k * d + j], lu[pivot * d + j]);
                }
                std::swap(permutation[k], permutation[pivot]);
                det = -det;
            }
            det *= lu[k * d + k];
            for (unsigned i = k + 1; i < d; ++i) {
                lu[i * d + k] /= lu[k * d + k];
                for (unsigned j = k + 1; j < d; ++j) {
                    lu[i * d + j] = std::fma(-lu[i * d + k], lu[k * d + j], lu[i * d + j]);
                }
            }
        }
        if (!std::isfinite(det) || std::abs(det) <= o.detTolerance_) {
            fail(EvaluationCode::DegenerateCell, "归一化行列式低于门槛", g.cell_);
        }
        g.normalizedDeterminant_ = det;
        g.orientation_ = det > 0 ? 1 : -1;
        for (unsigned col = 0; col < d; ++col) {
            std::array<double, 3> solution{};
            for (unsigned i = 0; i < d; ++i) {
                solution[i] = permutation[i] == col ? 1 : 0;
                for (unsigned j = 0; j < i; ++j) {
                    solution[i] -= lu[i * d + j] * solution[j];
                }
            }
            for (int i = static_cast<int>(d) - 1; i >= 0; --i) {
                for (unsigned j = i + 1; j < d; ++j) {
                    solution[i] -= lu[i * d + j] * solution[j];
                }
                solution[i] /= lu[i * d + i];
                inv[i * d + col] = solution[i];
            }
        }
        double normA = 0, normInv = 0;
        g.inverseJacobian_.resize(d * d);
        for (unsigned i = 0; i < d * d; ++i) {
            normA = std::hypot(normA, g.jacobian_[i] / g.edgeScale_);
            normInv = std::hypot(normInv, inv[i]);
            g.inverseJacobian_[i] = inv[i] / g.edgeScale_;
            if (!std::isfinite(g.inverseJacobian_[i])) {
                fail(EvaluationCode::NumericalOverflow, "逆映射不可表示", g.cell_);
            }
        }
        g.conditionEstimate_ = normA * normInv;
        if (!std::isfinite(g.conditionEstimate_) || g.conditionEstimate_ > o.maxCondition_) {
            fail(EvaluationCode::IllConditionedCell, "Jacobian条件数超过上限", g.cell_);
        }
        long double physical = std::abs(static_cast<long double>(det));
        for (unsigned i = 0; i < d; ++i) {
            physical *= g.edgeScale_;
        }
        g.absoluteDeterminant_ = positive(physical, g.cell_);
        g.measure_ = positive(physical / (d == 2 ? 2 : 6), g.cell_);
        return g;
    }
    FacetGeometry buildFacetGeometry(const FiniteElementSpace& s, const AffineCellGeometry& g, std::size_t side,
                                     const GeometryOptions& o) {
        validateGeometryOptions(o, s.dimension());
        const auto row = s.cellPosition(g.cell());
        if (!row || g.dimension() != s.dimension()) {
            throw std::invalid_argument("侧几何与母空间不匹配");
        }
        const auto local = referenceElement(s.elementKey(*row)).facetLocalDofs(side);
        FacetGeometry f;
        f.use_ = {g.cell(), side};
        f.dimension_ = g.dimension();
        f.parentLocalDofs_.assign(local.begin(), local.end());
        for (auto i : local) {
            f.vertices_.push_back(g.vertices().at(i));
        }
        const Point u = difference(f.vertices_[1], f.vertices_[0]);
        const double uLength = norm(u);
        if (!(uLength > 0) || !std::isfinite(uLength)) {
            fail(EvaluationCode::DegenerateCell, "侧边长非法", g.cell(), side);
        }
        Point n{};
        if (f.dimension_ == 2) {
            n = {u[1] / uLength, -u[0] / uLength, 0};
            f.surfaceJacobian_ = uLength;
        }
        else {
            const auto v = difference(f.vertices_[2], f.vertices_[0]);
            const double vLength = norm(v);
            if (!(vLength > 0) || !std::isfinite(vLength)) {
                fail(EvaluationCode::DegenerateCell, "侧面边长非法", g.cell(), side);
            }
            Point a{}, b{};
            for (unsigned i = 0; i < 3; ++i) {
                a[i] = u[i] / uLength;
                b[i] = v[i] / vLength;
            }
            n = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
            const double length = norm(n);
            if (!(length > 0) || !std::isfinite(length)) {
                fail(EvaluationCode::DegenerateCell, "侧面叉积退化", g.cell(), side);
            }
            f.surfaceJacobian_ = positive(static_cast<long double>(uLength) * vLength * length, g.cell(), side);
            for (auto& x : n) {
                x /= length;
            }
        }
        f.measure_ =
            positive(static_cast<long double>(f.surfaceJacobian_) / (f.dimension_ == 2 ? 1 : 2), g.cell(), side);
        std::size_t opposite = 0;
        while (std::find(local.begin(), local.end(), opposite) != local.end()) {
            ++opposite;
        }
        auto offset = difference(g.vertices().at(opposite), f.vertices_[0]);
        for (auto& x : offset) {
            x /= g.edgeScale();
        }
        const double sign = dot(n, offset);
        if (!std::isfinite(sign) || std::abs(sign) <= o.detTolerance_) {
            fail(EvaluationCode::OrientationUnresolved, "无法稳定确定外法向", g.cell(), side);
        }
        if (sign > 0) {
            for (auto& x : n) {
                x = -x;
            }
        }
        if (std::abs(norm(n) - 1) > o.normalTolerance_) {
            fail(EvaluationCode::OrientationUnresolved, "侧法向单位长度检查失败", g.cell(), side);
        }
        for (std::size_t i = 1; i < f.vertices_.size(); ++i) {
            auto tangent = difference(f.vertices_[i], f.vertices_[0]);
            const auto length = norm(tangent);
            for (auto& x : tangent) {
                x /= length;
            }
            if (std::abs(dot(n, tangent)) > o.normalTolerance_) {
                fail(EvaluationCode::OrientationUnresolved, "侧法向与切向不正交", g.cell(), side);
            }
        }
        f.normal_ = n;
        return f;
    }
    const FacetGeometry& PreparedGeometry::exteriorFacet(std::size_t i) const {
        const auto& f = exterior_.at(i);
        if (!f) {
            throw std::invalid_argument("请求的侧不是活动域外侧");
        }
        return *f;
    }
    std::shared_ptr<const PreparedGeometry> buildPreparedGeometry(std::shared_ptr<const FiniteElementSpace> space,
                                                                  const GeometryOptions& o, std::size_t maxCells,
                                                                  std::size_t maxFacetUses) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::fem::buildPreparedGeometry", emberfield::NvtxColor::Green);
        if (!space || !maxCells || !maxFacetUses) {
            throw std::invalid_argument("空间不能为空且几何限额必须为正");
        }
        validateGeometryOptions(o, space->dimension());
        if (space->cellCount() > maxCells) {
            throw EvaluationError(EvaluationCode::PreparationLimit, EvaluationStage::Geometry, "活动单元超过限额");
        }
        std::size_t visits = 0;
        for (std::size_t row = 0; row < space->cellCount(); ++row) {
            const auto sides = referenceElement(space->elementKey(row)).facetCount();
            if (sides > maxFacetUses - visits) {
                throw EvaluationError(EvaluationCode::PreparationLimit, EvaluationStage::Geometry, "侧use超过限额");
            }
            visits += sides;
        }
        auto p = std::shared_ptr<PreparedGeometry>(new PreparedGeometry);
        p->space_ = std::move(space);
        p->options_ = o;
        p->cells_.reserve(p->space_->cellCount());
        for (std::size_t row = 0; row < p->space_->cellCount(); ++row) {
            p->cells_.push_back(buildCellGeometry(*p->space_, row, o));
        }
        p->topology_ = mesh::buildFacetTopology(p->space_->mesh(), p->space_->cells(), maxFacetUses);
        if (!p->topology_.unsupported_.empty()) {
            fail(EvaluationCode::NonmanifoldTopology, "未知单元侧布局", p->topology_.unsupported_[0]);
        }
        p->exterior_.resize(p->topology_.facets_.size());
        for (std::size_t i = 0; i < p->topology_.facets_.size(); ++i) {
            const auto& uses = p->topology_.facets_[i].uses_;
            if (uses.empty() || uses.size() > 2) {
                throw EvaluationError(EvaluationCode::NonmanifoldTopology, EvaluationStage::Geometry, "活动侧非流形");
            }
            const auto make = [&](mesh::FacetUse use) {
                const auto row = p->space_->cellPosition(use.cell_);
                if (!row) {
                    throw std::logic_error("拓扑包含非活动单元");
                }
                return buildFacetGeometry(*p->space_, p->cells_[*row], use.localSide_, o);
            };
            auto first = make(uses[0]);
            if (uses.size() == 1) {
                p->exterior_[i] = std::move(first);
                continue;
            }
            const auto second = make(uses[1]);
            Point sum{};
            for (unsigned k = 0; k < 3; ++k) {
                sum[k] = first.normal()[k] + second.normal()[k];
            }
            const double relative =
                std::abs(first.measure() - second.measure()) / std::max(first.measure(), second.measure());
            if (norm(sum) > o.normalTolerance_ || relative > o.normalTolerance_ || uses[0].cell_ == uses[1].cell_) {
                fail(EvaluationCode::InconsistentNeighbors, "共享侧两母单元不在相反半空间或测度不相容", uses[0].cell_,
                     uses[0].localSide_);
            }
        }
        return p;
    }
} // namespace emberfield::fem
