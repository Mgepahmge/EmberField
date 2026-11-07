/** @file Postprocess.cu
 * @brief 复用有限元几何建立身份映射、AABB定位以及GPU物理场和离散热量诊断。
 */
#define PICOJSON_USE_INT64
#include <NVTXProfiler.cuh>
#include <Numerics/LinearAlgebra/IterationContext.cuh>
#include <Numerics/LinearAlgebra/Operations.cuh>
#include <Simulation/Case/Postprocess.cuh>
#include <Simulation/Case/Region.cuh>
#include <Simulation/Case/System.cuh>
#include <Simulation/Case/Temporal.cuh>
#include <Simulation/Thermal/PostprocessOperations.cuh>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <picojson.h>
namespace emberfield::problem {
    namespace {
        using J = picojson::value;
        using O = picojson::object;
        using A = picojson::array;
        using Point = std::array<double, 3>;
        /** @brief 受检预算乘法。@param a 数量。@param b 大小。@pre 无。@return 字节；超限length_error。@code auto
         * n=bytes(c,64); @endcode */
        std::size_t bytes(std::size_t a, std::size_t b) {
            if (b && a > SIZE_MAX / b) {
                throw std::length_error("后处理预算溢出");
            }
            return a * b;
        }
        /** @brief 三维平方距离。@param a 点。@param b 点。@pre 有限。@return 平方距离。@code auto d=distance(a,b);
         * @endcode */
        double distance(const Point& a, const Point& b) {
            double s = 0;
            for (int k = 0; k < 3; ++k) {
                s += (a[k] - b[k]) * (a[k] - b[k]);
            }
            return s;
        }
        /** @brief 单纯形闭包最近点：枚举非空面，在面仿射空间投影并检查重心系数。
         * @param vertices 有序顶点。@param p 查询。@pre 非退化三角形或四面体。
         * @return 非负重心权重；枚举最多15个面。@code auto w=closest(v,p); @endcode */
        std::array<double, 4> closest(std::span<const Point> vertices, const Point& p) {
            std::array<double, 4> best{};
            double bestDistance = INFINITY;
            for (unsigned mask = 1; mask < (1u << vertices.size()); ++mask) {
                std::array<unsigned, 4> ids{};
                unsigned count = 0;
                for (unsigned i = 0; i < vertices.size(); ++i) {
                    if (mask & (1u << i)) {
                        ids[count++] = i;
                    }
                }
                double matrix[3][4]{};
                const auto& origin = vertices[ids[0]];
                const auto n = count - 1;
                for (unsigned i = 0; i < n; ++i) {
                    for (unsigned j = 0; j < n; ++j) {
                        for (int k = 0; k < 3; ++k) {
                            matrix[i][j] +=
                                (vertices[ids[i + 1]][k] - origin[k]) * (vertices[ids[j + 1]][k] - origin[k]);
                        }
                    }
                    for (int k = 0; k < 3; ++k) {
                        matrix[i][n] += (vertices[ids[i + 1]][k] - origin[k]) * (p[k] - origin[k]);
                    }
                }
                bool valid = true;
                for (unsigned k = 0; k < n; ++k) {
                    unsigned pivot = k;
                    for (unsigned j = k + 1; j < n; ++j) {
                        if (std::abs(matrix[j][k]) > std::abs(matrix[pivot][k])) {
                            pivot = j;
                        }
                    }
                    if (matrix[pivot][k] == 0) {
                        valid = false;
                        break;
                    }
                    for (unsigned j = k; j <= n; ++j) {
                        std::swap(matrix[k][j], matrix[pivot][j]);
                    }
                    double divisor = matrix[k][k];
                    for (unsigned j = k; j <= n; ++j) {
                        matrix[k][j] /= divisor;
                    }
                    for (unsigned i = 0; i < n; ++i) {
                        if (i != k) {
                            const double factor = matrix[i][k];
                            for (unsigned j = k; j <= n; ++j) {
                                matrix[i][j] -= factor * matrix[k][j];
                            }
                        }
                    }
                }
                if (!valid) {
                    continue;
                }
                std::array<double, 4> w{};
                w[ids[0]] = 1;
                for (unsigned i = 0; i < n; ++i) {
                    w[ids[i + 1]] = matrix[i][n];
                    w[ids[0]] -= matrix[i][n];
                }
                for (double x : w) {
                    if (x < 0 || !std::isfinite(x)) {
                        valid = false;
                    }
                }
                if (!valid) {
                    continue;
                }
                Point q{};
                for (unsigned a = 0; a < vertices.size(); ++a) {
                    for (int k = 0; k < 3; ++k) {
                        q[k] += w[a] * vertices[a][k];
                    }
                }
                const double d = distance(p, q);
                if (d < bestDistance) {
                    bestDistance = d;
                    best = w;
                }
            }
            return best;
        }
        /** @brief AABB节点；内部组件不公开为通用模块。@code Box b; @endcode */
        struct Box {
            Point low_{}, high_{};
            std::size_t first_{}, end_{}, left_{SIZE_MAX}, right_{SIZE_MAX};
        }; // ///< 包围盒、行段和子节点。
        /** @brief 已定位的监测，不在每步搜索。@code Resolved r; @endcode */
        struct Resolved {
            std::vector<std::size_t> rows_; ///< 区域活动行。
            std::vector<std::uint64_t> dofs_; ///< 插值DOF。
            std::vector<double> weights_; ///< 插值权重。
            J dictionary_; ///< 有效定位JSON。
        };
    } // namespace
    struct ThermalPostprocessPlan::Impl {
        std::shared_ptr<const PreparedThermalCase> prepared_; ///< 同源准备。
        std::shared_ptr<const field::VisualizationGrid> grid_; ///< 静态可视化网格。
        std::vector<thermal::P1CellSample> cells_; ///< 单元GPU批上传源。
        std::unique_ptr<thermal::detail::P1PostprocessWorkspace> cellWorkspace_; ///< 借用cells_的有界设备链。
        std::optional<AssembledThermalModel> spatialOwner_; ///< 保留不可变S身份，防止地址复用。
        std::optional<AssembledThermalModel> capacityOwner_; ///< 保留固定M身份。
        std::unique_ptr<linalg::IterationContext<double, assembly::Index>> spatialOperation_,
            capacityOperation_; ///< 缓存热平衡乘法资源。
        /** @brief 对不可变模型矩阵复用执行计划。@param model 同源模型。@param x 向量。
         * @param capacity 是否使用M。@pre 模型来自封闭组装链。@return 新独立设备结果。
         * @code auto value=s.product(model,t,true); @endcode */
        Vector<double> product(const AssembledThermalModel& model, const Vector<double>& x, bool capacity) {
            auto& owner = capacity ? capacityOwner_ : spatialOwner_;
            auto& operation = capacity ? capacityOperation_ : spatialOperation_;
            const auto& matrix = capacity ? *model.capacity() : model.spatial();
            const bool same = owner &&
                (capacity ? owner->capacity()->deviceValues() : owner->spatial().deviceValues()) ==
                    matrix.deviceValues();
            if (!same) {
                auto candidate = std::make_unique<linalg::IterationContext<double, assembly::Index>>(matrix);
                owner = model;
                operation = std::move(candidate);
            }
            Vector<double> result(x.size());
            operation->multiply(x, result);
            return result;
        }
        std::vector<thermal::BoundaryPowerSample> boundary_; ///< 已积分自然功率项。
        std::vector<thermal::BoundaryPowerSample> boundaryUnits_; ///< 单位h迹积分，通量使用面积。
        std::vector<double> boundaryAreas_; ///< 自然侧面积乘二维厚度。
        std::vector<std::size_t> boundaryDefinitions_; ///< 功率项定义行。
        std::vector<Resolved> monitors_; ///< 静态定位。
        std::vector<std::size_t> order_; ///< AABB叶子行顺序。
        std::vector<Box> boxes_; ///< AABB树。
        double sourcePower_{}; ///< 体源积分总功率。
        bool requiresTime_{}; ///< 物理定义含时间表，禁止省略快照。
        /** @brief 校验快照并借用对应约束。@param state 当前参数或空。@pre 同源。
         * @return g数组。@code auto& g=s.constraints(state); @endcode */
        const std::vector<double>& constraints(const EvaluatedThermalState* state) const {
            if (state) {
                if (!state->plan() || state->plan()->assemblyPlan()->prepared() != prepared_) {
                    throw std::invalid_argument("后处理时间快照来源错配");
                }
                return state->constraints();
            }
            if (requiresTime_) {
                throw std::invalid_argument("时变后处理必须提供参数快照");
            }
            return prepared_->conditions()->constraints().temperatures();
        }
        /** @brief 建立平衡AABB树。@param first 起始叶。@param end 结束叶。@pre 非空段。@return 树节点。@code auto
         * root=tree(0,n); @endcode */
        std::size_t tree(std::size_t first, std::size_t end) {
            Box b;
            b.first_ = first;
            b.end_ = end;
            b.low_.fill(INFINITY);
            b.high_.fill(-INFINITY);
            for (auto i = first; i < end; ++i) {
                for (const auto& p : prepared_->geometry()->cells()[order_[i]].vertices()) {
                    for (int k = 0; k < 3; ++k) {
                        b.low_[k] = std::min(b.low_[k], p[k]);
                        b.high_[k] = std::max(b.high_[k], p[k]);
                    }
                }
            }
            const auto index = boxes_.size();
            boxes_.push_back(b);
            if (end - first > 8) {
                int axis = 0;
                for (int k = 1; k < 3; ++k) {
                    if (b.high_[k] - b.low_[k] > b.high_[axis] - b.low_[axis]) {
                        axis = k;
                    }
                }
                const auto middle = first + (end - first) / 2;
                std::nth_element(order_.begin() + first, order_.begin() + middle, order_.begin() + end,
                                 [&](auto a, auto c) {
                                     const auto& va = prepared_->geometry()->cells()[a].vertices();
                                     const auto& vc = prepared_->geometry()->cells()[c].vertices();
                                     double sa = 0, sc = 0;
                                     for (auto p : va) {
                                         sa += p[axis];
                                     }
                                     for (auto p : vc) {
                                         sc += p[axis];
                                     }
                                     return sa < sc || (sa == sc && a < c);
                                 });
                const auto left = tree(first, middle);
                const auto right = tree(middle, end);
                boxes_[index].left_ = left;
                boxes_[index].right_ = right;
            }
            return index;
        }
        /** @brief 查询容差内单元，按最小行选择，防止材料界面歧义。@param index 子树。@param p 点。
         * @param tolerance 米制容差。@param row 最佳行。@param weights 重心系数。@pre 已建树。
         * @note 无返回值。@code locate(0,p,tol,row,w); @endcode */
        void locate(std::size_t index, const Point& p, double tolerance, std::size_t& row,
                    std::array<double, 4>& weights) const {
            const auto& b = boxes_[index];
            for (int k = 0; k < 3; ++k) {
                if (p[k] < b.low_[k] - tolerance || p[k] > b.high_[k] + tolerance) {
                    return;
                }
            }
            if (b.left_ != SIZE_MAX) {
                locate(b.left_, p, tolerance, row, weights);
                locate(b.right_, p, tolerance, row, weights);
                return;
            }
            for (auto i = b.first_; i < b.end_; ++i) {
                const auto r = order_[i];
                if (r >= row) {
                    continue;
                }
                const auto& v = prepared_->geometry()->cells()[r].vertices();
                auto w = closest(v, p);
                Point q{};
                for (std::size_t a = 0; a < v.size(); ++a) {
                    for (int k = 0; k < 3; ++k) {
                        q[k] += w[a] * v[a][k];
                    }
                }
                if (std::sqrt(distance(q, p)) <= tolerance) {
                    row = r;
                    weights = w;
                }
            }
        }
    };
    ThermalPostprocessPlan::ThermalPostprocessPlan(std::shared_ptr<const PreparedThermalCase> prepared) :
        impl_(std::make_unique<Impl>()) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::problem::ThermalPostprocessPlan::ThermalPostprocessPlan",
                                    emberfield::NvtxColor::Yellow);
        if (!prepared) {
            throw std::invalid_argument("后处理准备对象为空");
        }
        auto& s = *impl_;
        s.prepared_ = std::move(prepared);
        const auto& p = *s.prepared_;
        const auto& space = *p.space();
        const auto& d = p.bound()->data();
        const auto& def = d.definition_;
        const auto& m = d.mesh_->data();
        const auto& limits = def.output_.limits_;
        const auto count = space.cells().size(), points = space.dofCount();
        if (count > limits.maxCells_ || points > limits.maxPoints_ || bytes(count, 1400) > limits.maxHostBytes_ ||
            bytes(points, 192) > limits.maxHostBytes_ - bytes(count, 1400)) {
            throw std::length_error("后处理主机预算不足");
        }
        const auto baseBytes = bytes(count, 1400) + bytes(points, 192);
        if (bytes(p.conditions()->naturalTerms().size(),
                  2 * sizeof(thermal::BoundaryPowerSample) + sizeof(double) + sizeof(std::size_t)) >
            limits.maxHostBytes_ - baseBytes) {
            throw std::length_error("后处理自然边界缓存超过主机预算");
        }
        const auto batch = std::min(count, limits.batchCells_);
        const auto nnz = bytes(count, 16); // P1局部条目上界，仅预算估计，不创建矩阵。
        if (bytes(batch, 1024) > limits.maxDeviceBytes_ ||
            bytes(points, 128) > limits.maxDeviceBytes_ - bytes(batch, 1024) ||
            bytes(nnz, 64) > limits.maxDeviceBytes_ - bytes(batch, 1024) - bytes(points, 128)) {
            throw std::length_error("后处理设备预算不足");
        }
        const auto membership = buildPhysicalMembership(*d.mesh_, std::max<std::size_t>(1, limits.maxHostBytes_ / 256));
        std::vector<Point> xyz(points);
        std::vector<std::uint64_t> tags(points), dofs(points), offsets{0}, connectivity, cellTags, cellRows, materials,
            groupOffsets{0};
        std::vector<std::uint8_t> hasEntity, entityDimensions, groupDimensions;
        std::vector<std::int64_t> entities, groupTags;
        std::vector<mesh::ElementType> layouts;
        for (std::size_t i = 0; i < points; ++i) {
            const auto node = *space.dofInfo(i).node_;
            xyz[i] = p.bound()->position(node);
            if (space.dimension() == 2) {
                xyz[i][2] = 0;
            }
            tags[i] = m.nodes_[node].tag_;
            dofs[i] = i;
        }
        std::map<mesh::CellRef, std::vector<GroupKey>> groups;
        for (const auto& [key, members] : membership.members_) {
            for (auto ref : members) {
                groups[ref].push_back(key);
            }
        }
        const double scale = def.analysis_.thickness_.value_or(1);
        s.cells_.reserve(count);
        for (std::size_t row = 0; row < count; ++row) {
            const auto ref = space.cells()[row];
            const auto& block = m.elements_[ref.block_];
            const auto indices = space.cellDofs(row);
            const auto& geometry = p.geometry()->cells()[row];
            auto table = fem::evaluateCellIntegration(p.geometry(), row, 1);
            thermal::P1CellSample sample{};
            sample.dimension_ = space.dimension();
            sample.volume_ = geometry.measure() * scale;
            for (std::size_t a = 0; a < indices.size(); ++a) {
                sample.dofs_[a] = indices[a];
                connectivity.push_back(indices[a]);
                for (unsigned k = 0; k < sample.dimension_; ++k) {
                    sample.gradients_[a * 3 + k] = table.gradients()[a * sample.dimension_ + k];
                }
            }
            const auto material = p.conditions()->materialIndices()[row];
            const auto& mat = def.materials_[material];
            sample.conductivity_ = problem::evaluate(mat.conductivity_);
            if (def.time_) {
                sample.capacity_ = mat.volumetricHeatCapacity_
                    ? problem::evaluate(*mat.volumetricHeatCapacity_)
                    : problem::evaluate(*mat.density_) * problem::evaluate(*mat.specificHeat_);
            }
            for (auto j = p.conditions()->sourceOffsets()[row]; j < p.conditions()->sourceOffsets()[row + 1]; ++j) {
                const auto& field = def.sources_[p.conditions()->sourceIndices()[j]].value_;
                s.requiresTime_ |= std::holds_alternative<TimeTableScalar>(field);
                sample.source_ += problem::evaluate(field, def.time_ ? def.time_->start_ : 0);
            }
            s.sourcePower_ += sample.source_ * sample.volume_;
            s.cells_.push_back(sample);
            offsets.push_back(connectivity.size());
            layouts.push_back(block.type_);
            cellTags.push_back(block.tags_[ref.element_]);
            cellRows.push_back(row);
            materials.push_back(material);
            hasEntity.push_back(block.classification_.has_value());
            entityDimensions.push_back(block.classification_ ? block.classification_->dimension_ : 0);
            entities.push_back(block.classification_ ? block.classification_->tag_ : 0);
            for (const auto& key : groups[ref]) {
                groupDimensions.push_back(key.first);
                groupTags.push_back(key.second);
            }
            groupOffsets.push_back(groupTags.size());
        }
        using field::NumericArray;
        std::vector<NumericArray> pi, ci, metadata;
        pi.push_back({"sourceNodeTag", 1, points, "1", std::move(tags)});
        pi.push_back({"dofIndex", 1, points, "1", std::move(dofs)});
        ci.push_back({"sourceCellTag", 1, count, "1", std::move(cellTags)});
        ci.push_back({"sourceCellRow", 1, count, "1", std::move(cellRows)});
        ci.push_back({"materialIndex", 1, count, "1", std::move(materials)});
        ci.push_back({"hasEntity", 1, count, "1", std::move(hasEntity)});
        ci.push_back({"entityDimension", 1, count, "1", std::move(entityDimensions)});
        ci.push_back({"entityTag", 1, count, "1", std::move(entities)});
        metadata.push_back({"emberfieldSchemaVersion", 1, 1, "1", std::vector<std::uint32_t>{1}});
        metadata.push_back(
            {"dimension", 1, 1, "1", std::vector<std::uint8_t>{static_cast<std::uint8_t>(space.dimension())}});
        metadata.push_back({"coordinateScale", 1, 1, "1", std::vector<double>{def.mesh_.coordinateScale_}});
        if (space.dimension() == 2) {
            metadata.push_back({"thickness", 1, 1, "m", std::vector<double>{scale}});
        }
        metadata.push_back({"physicalGroupOffsets", 1, groupOffsets.size(), "1", std::move(groupOffsets)});
        metadata.push_back({"physicalGroupDimensions", 1, groupDimensions.size(), "1", std::move(groupDimensions)});
        metadata.push_back({"physicalGroupTags", 1, groupTags.size(), "1", std::move(groupTags)});
        s.grid_ = std::make_shared<const field::VisualizationGrid>(std::move(xyz), std::move(offsets),
                                                                   std::move(connectivity), std::move(layouts),
                                                                   std::move(pi), std::move(ci), std::move(metadata));
        for (const auto& term : p.conditions()->naturalTerms()) {
            const auto table = fem::evaluateFacetIntegration(p.geometry(), term.facetIndex(), 2);
            thermal::BoundaryPowerSample sample{};
            thermal::BoundaryPowerSample unit{};
            double area = 0;
            sample.count_ = term.globalDofs().size();
            for (unsigned a = 0; a < sample.count_; ++a) {
                sample.dofs_[a] = term.globalDofs()[a];
                unit.dofs_[a] = term.globalDofs()[a];
            }
            unit.count_ = sample.count_;
            const auto& value = def.boundaries_[term.definitionIndex()].value_;
            for (std::size_t q = 0; q < table.pointCount(); ++q) {
                const double w = table.physicalWeights()[q] * scale;
                area += w;
                for (unsigned a = 0; a < sample.count_; ++a) {
                    unit.weights_[a] += w * table.traceValues()[q * sample.count_ + a];
                }
                if (const auto* f = std::get_if<FluxBoundary>(&value)) {
                    s.requiresTime_ |= std::holds_alternative<TimeTableScalar>(f->flux_);
                    sample.constant_ += w * problem::evaluate(f->flux_, def.time_ ? def.time_->start_ : 0);
                }
                else {
                    const auto& h = std::get<ConvectionBoundary>(value);
                    s.requiresTime_ |= std::holds_alternative<TimeTableScalar>(h.coefficient_) ||
                        std::holds_alternative<TimeTableScalar>(h.ambientTemperature_);
                    const double hValue = problem::evaluate(h.coefficient_, def.time_ ? def.time_->start_ : 0);
                    sample.constant_ -=
                        w * hValue * problem::evaluate(h.ambientTemperature_, def.time_ ? def.time_->start_ : 0);
                    for (unsigned a = 0; a < sample.count_; ++a) {
                        sample.weights_[a] += w * hValue * table.traceValues()[q * sample.count_ + a];
                    }
                }
            }
            s.boundary_.push_back(sample);
            s.boundaryUnits_.push_back(unit);
            s.boundaryAreas_.push_back(area);
            s.boundaryDefinitions_.push_back(term.definitionIndex());
        }
        for (const auto& boundary : def.boundaries_) {
            if (const auto* g = std::get_if<TemperatureBoundary>(&boundary.value_)) {
                s.requiresTime_ |= std::holds_alternative<TimeTableScalar>(g->temperature_);
            }
        }
        if (std::any_of(def.output_.monitors_.begin(), def.output_.monitors_.end(),
                        [](const MonitorDefinition& x) { return std::holds_alternative<PointMonitor>(x.value_); })) {
            s.order_.resize(count);
            std::iota(s.order_.begin(), s.order_.end(), 0);
            s.boxes_.reserve(count * 2);
            s.tree(0, count);
        }
        std::size_t monitorBytes = 0;
        for (const auto& monitor : def.output_.monitors_) {
            Resolved r;
            O dictionary{{"id", J(monitor.id_)}};
            if (const auto* n = std::get_if<NodeMonitor>(&monitor.value_)) {
                std::optional<std::uint64_t> found;
                for (std::size_t i = 0; i < points; ++i) {
                    if (m.nodes_[*space.dofInfo(i).node_].tag_ == n->nodeTag_) {
                        found = i;
                        break;
                    }
                }
                if (!found) {
                    throw std::invalid_argument("监测节点不在活动域: " + monitor.id_);
                }
                r.dofs_ = {*found};
                r.weights_ = {1};
                dictionary["kind"] = J("node");
                dictionary["resolved"] =
                    J(O{{"nodeTag", J(std::to_string(n->nodeTag_))}, {"dofIndex", J(std::to_string(*found))}});
            }
            else if (const auto* point = std::get_if<PointMonitor>(&monitor.value_)) {
                std::size_t row = SIZE_MAX;
                std::array<double, 4> weights{};
                const double effectiveTolerance = point->toleranceM_ +
                    64 * std::numeric_limits<double>::epsilon() *
                        std::max({1., std::abs(point->positionM_[0]), std::abs(point->positionM_[1]),
                                  std::abs(point->positionM_[2])});
                if (!std::isfinite(effectiveTolerance)) {
                    throw std::invalid_argument("定位有效容差溢出");
                }
                s.locate(0, point->positionM_, effectiveTolerance, row, weights);
                if (row == SIZE_MAX) {
                    throw std::invalid_argument("监测点不在活动域闭包内: " + monitor.id_);
                }
                const auto indices = space.cellDofs(row);
                r.dofs_.assign(indices.begin(), indices.end());
                r.weights_.assign(weights.begin(), weights.begin() + indices.size());
                Point actual{};
                const auto& v = p.geometry()->cells()[row].vertices();
                for (std::size_t a = 0; a < v.size(); ++a) {
                    for (int k = 0; k < 3; ++k) {
                        actual[k] += weights[a] * v[a][k];
                    }
                }
                A bary, position;
                for (double x : r.weights_) {
                    bary.emplace_back(x);
                }
                for (double x : actual) {
                    position.emplace_back(x);
                }
                dictionary["kind"] = J("point");
                dictionary["resolved"] = J(O{{"cellRow", J(std::to_string(row))},
                                             {"barycentric", J(bary)},
                                             {"samplePositionM", J(position)},
                                             {"effectiveToleranceM", J(effectiveTolerance)}});
            }
            else {
                const auto& region = std::get<RegionMonitor>(monitor.value_);
                auto selected = resolveRegion(region.region_, membership);
                if (selected.diagnostics_.hasErrors()) {
                    throw std::invalid_argument("区域监测选择失败: " + monitor.id_);
                }
                double volume = 0;
                for (std::size_t row = 0; row < count; ++row) {
                    if (std::binary_search(selected.cells_.begin(), selected.cells_.end(), space.cells()[row])) {
                        r.rows_.push_back(row);
                        volume += s.cells_[row].volume_;
                    }
                }
                if (r.rows_.empty()) {
                    throw std::invalid_argument("区域监测活动交集为空: " + monitor.id_);
                }
                dictionary["kind"] = J("region");
                dictionary["resolved"] =
                    J(O{{"cellCount", J(std::to_string(r.rows_.size()))}, {"volumeM3", J(volume)}});
            }
            const auto additional = bytes(r.rows_.size() + r.dofs_.size() + r.weights_.size() + 64, 32);
            const auto base = bytes(count, 1400) + bytes(points, 192) +
                bytes(p.conditions()->naturalTerms().size(),
                      2 * sizeof(thermal::BoundaryPowerSample) + sizeof(double) + sizeof(std::size_t));
            if (additional > limits.maxHostBytes_ - base - monitorBytes) {
                throw std::length_error("监测定位缓存超出主机预算");
            }
            monitorBytes += additional;
            r.dictionary_ = J(dictionary);
            s.monitors_.push_back(std::move(r));
        }
        const auto estimated = bytes(batch, 1024) + bytes(points, 128) + bytes(nnz, 64);
        const auto cacheBudget = std::min<std::size_t>(192ULL * 1024 * 1024, limits.maxDeviceBytes_ - estimated);
        s.cellWorkspace_ = std::make_unique<thermal::detail::P1PostprocessWorkspace>(
            s.cells_, std::max<std::size_t>(1, batch), cacheBudget);
    }
    ThermalPostprocessPlan::~ThermalPostprocessPlan() noexcept = default;
    ThermalPostprocessPlan::ThermalPostprocessPlan(ThermalPostprocessPlan&&) noexcept = default;
    ThermalPostprocessPlan& ThermalPostprocessPlan::operator=(ThermalPostprocessPlan&&) noexcept = default;
    const std::shared_ptr<const field::VisualizationGrid>& ThermalPostprocessPlan::grid() const noexcept {
        return impl_->grid_;
    }
    std::string ThermalPostprocessPlan::monitorDictionary() const {
        A a;
        for (const auto& r : impl_->monitors_) {
            a.push_back(r.dictionary_);
        }
        return J(a).serialize();
    }
    field::FieldSnapshot ThermalPostprocessPlan::evaluateThermalFields(const Vector<double>& temperature,
                                                                       std::uint64_t step, std::optional<double> time,
                                                                       bool initial,
                                                                       const EvaluatedThermalState* state) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::problem::ThermalPostprocessPlan::evaluateThermalFields",
                                    emberfield::NvtxColor::Yellow);
        auto& s = *impl_;
        const auto& def = s.prepared_->bound()->data().definition_;
        const auto& output = def.output_;
        systems::validateTemperatureState(temperature, s.prepared_->conditions()->constraints().dofs(),
                                          s.constraints(state));
        if (state && (!time || *time != state->time())) {
            throw std::invalid_argument("后处理场时间错配");
        }
        const auto n = s.cells_.size();
        if (temperature.size() != s.grid_->pointCount() || def.time_.has_value() != time.has_value() ||
            (time && !std::isfinite(*time)) || (initial && (!def.time_ || step != 0))) {
            throw std::invalid_argument("结果场身份或维度不符");
        }
        std::vector<double> host(temperature.size());
        temperature.copyToHost(host);
        std::vector<thermal::P1CellValue> values(output.fields_.size() > 1 ? n : 0);
        for (std::size_t first = 0; output.fields_.size() > 1 && first < n;) {
            const auto count = std::min(output.limits_.batchCells_, n - first);
            auto batch = s.cellWorkspace_->fields(first, count, temperature);
            for (std::size_t i = 0; i < count; ++i) {
                values[first + i] = batch[i];
            }
            first += count;
        }
        std::vector<field::NumericArray> pointFields{{"temperature", 1, host.size(), "K", std::move(host)}}, cellFields;
        for (const auto field : output.fields_) {
            const auto f = std::string(outputFieldName(field));
            if (f != "temperature") {
                std::vector<double> v;
                v.reserve(n * 3);
                for (const auto& x : values) {
                    const auto* a = f == "heatFlux" ? x.flux_ : x.gradient_;
                    v.insert(v.end(), a, a + 3);
                }
                cellFields.push_back({f, 3, n, f == "heatFlux" ? "W/m^2" : "K/m", std::move(v)});
            }
        }
        return field::FieldSnapshot(s.grid_, step, time,
                                    initial ? field::SampleKind::Initial : field::SampleKind::Accepted,
                                    std::move(pointFields), std::move(cellFields));
    }
    std::vector<MonitorValue> ThermalPostprocessPlan::evaluateMonitors(const Vector<double>& temperature,
                                                                       const EvaluatedThermalState* state) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::problem::ThermalPostprocessPlan::evaluateMonitors",
                                    emberfield::NvtxColor::Yellow);
        auto& s = *impl_;
        const auto& def = s.prepared_->bound()->data().definition_;
        const auto& output = def.output_;
        systems::validateTemperatureState(temperature, s.prepared_->conditions()->constraints().dofs(),
                                          s.constraints(state));
        if (temperature.size() != s.grid_->pointCount()) {
            throw std::invalid_argument("监测温度维度不符");
        }
        std::vector<MonitorValue> result;
        std::vector<thermal::TemperatureSample> samples;
        for (const auto& r : s.monitors_) {
            if (!r.dofs_.empty()) {
                thermal::TemperatureSample sample{};
                sample.count_ = r.dofs_.size();
                for (unsigned a = 0; a < sample.count_; ++a) {
                    sample.dofs_[a] = r.dofs_[a];
                    sample.weights_[a] = r.weights_[a];
                }
                samples.push_back(sample);
            }
        }
        auto temperatures = thermal::interpolateTemperature(samples, temperature);
        std::size_t sampleIndex = 0;
        for (std::size_t i = 0; i < output.monitors_.size(); ++i) {
            const auto& monitor = output.monitors_[i];
            const auto& r = s.monitors_[i];
            if (!r.dofs_.empty()) {
                double t = temperatures[sampleIndex++];
                result.push_back({monitor.id_, std::holds_alternative<NodeMonitor>(monitor.value_) ? "node" : "point",
                                  "temperature", "K", t});
            }
            else {
                const auto& region = std::get<RegionMonitor>(monitor.value_);
                double min = INFINITY, max = -INFINITY, integral = 0, energy = 0, volume = 0;
                for (std::size_t first = 0; first < r.rows_.size();) {
                    const auto count = std::min(output.limits_.batchCells_, r.rows_.size() - first);
                    std::vector<thermal::P1CellSample> selectedCells(count);
                    std::vector<double> volumes(count);
                    for (std::size_t j = 0; j < count; ++j) {
                        const auto row = r.rows_[first + j];
                        selectedCells[j] = s.cells_[row];
                        volumes[j] = s.cells_[row].volume_;
                    }
                    const auto selected = thermal::evaluateP1Gradient(selectedCells, temperature);
                    const auto batch = thermal::reduceRegion(selected, volumes);
                    min = std::min(min, batch.minimum_);
                    max = std::max(max, batch.maximum_);
                    integral += batch.integral_;
                    energy += batch.energy_;
                    volume += batch.volume_;
                    first += count;
                }
                for (const auto statistic : region.statistics_) {
                    const auto f = std::string(regionStatisticName(statistic));
                    result.push_back({monitor.id_, "region", f, f == "storedEnergy" ? "J" : "K",
                                      f == "temperatureMin"        ? min
                                          : f == "temperatureMax"  ? max
                                          : f == "temperatureMean" ? integral / volume
                                                                   : energy});
                }
            }
        }
        for (const auto& r : result) {
            if (!std::isfinite(r.value_)) {
                throw std::overflow_error("监测结果非有限");
            }
        }
        return result;
    }
    HeatBalanceReport ThermalPostprocessPlan::evaluateHeatBalance(const AssembledThermalModel& model,
                                                                  const Vector<double>& temperature,
                                                                  const Vector<double>* previous,
                                                                  std::optional<double> dt, bool initial,
                                                                  const EvaluatedThermalState* state,
                                                                  const EvaluatedThermalState* previousState) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::problem::ThermalPostprocessPlan::evaluateHeatBalance",
                                    emberfield::NvtxColor::Yellow);
        if (model.parameters().get() != state) {
            throw std::invalid_argument("后处理模型与参数快照身份错配");
        }
        auto& s = *impl_;
        const auto& def = s.prepared_->bound()->data().definition_;
        const auto& output = def.output_;
        systems::validateTemperatureState(temperature, s.prepared_->conditions()->constraints().dofs(),
                                          s.constraints(state));
        if (temperature.size() != s.grid_->pointCount() || (previous && previous->size() != temperature.size()) ||
            model.plan()->prepared().get() != s.prepared_.get() ||
            (def.time_ && !initial && (!previous || !dt || !std::isfinite(*dt) || *dt <= 0)) ||
            ((!def.time_ || initial) && (previous || dt))) {
            throw std::invalid_argument("热平衡模型或前态不相容");
        }
        if (previous) {
            systems::validateTemperatureState(*previous, s.prepared_->conditions()->constraints().dofs(),
                                              s.constraints(previousState));
            if (state &&
                (!previousState || previousState->plan() != state->plan() || state->time() <= previousState->time() ||
                 *dt != state->time() - previousState->time())) {
                throw std::invalid_argument("后处理历史时间错配");
            }
        }
        HeatBalanceReport result;
        if (output.energy_.enabled_) {
            auto& e = result;
            e.status_ = "unavailable";
            if (model.capacity()) {
                auto energy = s.product(model, temperature, true);
                e.storedEnergyJ_ = thermal::sumValues(energy);
            }
            if (!initial) {
                auto residual = linalg::subtract(s.product(model, temperature, false), model.load());
                e.storagePowerW_ = 0;
                if (previous) {
                    const auto change = linalg::subtract(temperature, *previous);
                    const auto power = s.product(model, change, true);
                    e.storagePowerW_ = thermal::sumValues(power) / *dt;
                    residual = linalg::add(residual, power, 1 / *dt);
                }
                const auto& constraints = s.prepared_->conditions()->constraints().dofs();
                e.dirichletOutW_ = constraints.empty() ? 0 : -thermal::sumValues(residual, constraints);
            }
            e.sourcePowerW_ = s.sourcePower_;
            if (state) {
                *e.sourcePowerW_ = 0;
                const auto& c = *s.prepared_->conditions();
                for (std::size_t row = 0; row < s.cells_.size(); ++row) {
                    for (auto j = c.sourceOffsets()[row]; j < c.sourceOffsets()[row + 1]; ++j) {
                        *e.sourcePowerW_ += state->sources()[c.sourceIndices()[j]] * s.cells_[row].volume_;
                    }
                }
            }
            e.prescribedFluxOutW_ = 0;
            e.convectionOutW_ = 0;
            std::vector<double> powers(def.boundaries_.size(), 0);
            for (std::size_t first = 0; first < s.boundary_.size();) {
                const auto count = std::min(output.limits_.batchCells_, s.boundary_.size() - first);
                auto samples = std::vector<thermal::BoundaryPowerSample>(s.boundary_.begin() + first,
                                                                         s.boundary_.begin() + first + count);
                if (state) {
                    for (std::size_t i = 0; i < count; ++i) {
                        const auto row = first + i, definition = s.boundaryDefinitions_[row];
                        samples[i] = s.boundaryUnits_[row];
                        if (std::holds_alternative<FluxBoundary>(def.boundaries_[definition].value_)) {
                            std::fill(std::begin(samples[i].weights_), std::end(samples[i].weights_), 0);
                            samples[i].constant_ = s.boundaryAreas_[row] * state->primary()[definition];
                        }
                        else {
                            for (auto& w : samples[i].weights_) {
                                w *= state->primary()[definition];
                            }
                            samples[i].constant_ =
                                -s.boundaryAreas_[row] * state->primary()[definition] * state->ambient()[definition];
                        }
                    }
                }
                auto power = thermal::integrateBoundaryPowerBatch(samples, temperature);
                for (std::size_t i = 0; i < count; ++i) {
                    powers[s.boundaryDefinitions_[first + i]] += power[i];
                }
                first += count;
            }
            for (std::size_t i = 0; i < def.boundaries_.size(); ++i) {
                const auto& b = def.boundaries_[i];
                if (std::holds_alternative<TemperatureBoundary>(b.value_)) {
                    continue;
                }
                const bool isFlux = std::holds_alternative<FluxBoundary>(b.value_);
                (isFlux ? *e.prescribedFluxOutW_ : *e.convectionOutW_) += powers[i];
                e.boundaries_.push_back({b.id_, isFlux ? "flux" : "convection", powers[i]});
            }
            if (!std::isfinite(*e.sourcePowerW_) || !std::isfinite(*e.prescribedFluxOutW_) ||
                !std::isfinite(*e.convectionOutW_)) {
                throw std::overflow_error("热源或自然边界总功率非有限");
            }
            if (!initial) {
                e.boundaries_.push_back({"", "temperature", *e.dirichletOutW_});
                e.errorW_ = *e.storagePowerW_ + *e.prescribedFluxOutW_ + *e.convectionOutW_ + *e.dirichletOutW_ -
                    *e.sourcePowerW_;
                e.toleranceW_ = output.energy_.absoluteToleranceW_ +
                    output.energy_.relativeTolerance_ *
                        std::max({std::abs(*e.storagePowerW_), std::abs(*e.prescribedFluxOutW_),
                                  std::abs(*e.convectionOutW_), std::abs(*e.dirichletOutW_),
                                  std::abs(*e.sourcePowerW_)});
                if (!std::isfinite(*e.errorW_) || !std::isfinite(*e.toleranceW_)) {
                    throw std::overflow_error("热量平衡非有限");
                }
                e.status_ = std::abs(*e.errorW_) <= *e.toleranceW_ ? "passed" : "warning";
            }
        }
        return result;
    }
    ThermalFields ThermalPostprocessPlan::evaluate(const AssembledThermalModel& model,
                                                   const Vector<double>& temperature, const Vector<double>* previous,
                                                   std::optional<double> dt, std::uint64_t step,
                                                   std::optional<double> time, bool initial, bool includeFrame,
                                                   const EvaluatedThermalState* state,
                                                   const EvaluatedThermalState* previousState) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::problem::ThermalPostprocessPlan::evaluate",
                                    emberfield::NvtxColor::Yellow);
        auto& s = *impl_;
        const auto& definition = s.prepared_->bound()->data().definition_;
        if (state && (!time || *time != state->time())) {
            throw std::invalid_argument("后处理记录时刻错配");
        }
        if (temperature.size() != s.grid_->pointCount() || definition.time_.has_value() != time.has_value() ||
            (time && !std::isfinite(*time)) || (initial && (!definition.time_ || step != 0))) {
            throw std::invalid_argument("结果场身份或维度不符");
        }
        std::optional<field::FieldSnapshot> snapshot;
        double low = INFINITY, high = -INFINITY;
        if (includeFrame) {
            snapshot = evaluateThermalFields(temperature, step, time, initial, state);
            const auto& nodal = std::get<std::vector<double>>(snapshot->pointFields()[0].values_);
            const auto [minimum, maximum] = std::minmax_element(nodal.begin(), nodal.end());
            low = *minimum;
            high = *maximum;
        }
        else {
            systems::validateTemperatureState(temperature, s.prepared_->conditions()->constraints().dofs(),
                                              s.constraints(state));
            for (std::size_t first = 0; first < s.cells_.size();) {
                const auto count = std::min(definition.output_.limits_.batchCells_, s.cells_.size() - first);
                const auto statistics = s.cellWorkspace_->region(first, count, temperature);
                low = std::min(low, statistics.minimum_);
                high = std::max(high, statistics.maximum_);
                first += count;
            }
        }
        auto monitors = evaluateMonitors(temperature, state);
        auto energy = evaluateHeatBalance(model, temperature, previous, dt, initial, state, previousState);
        return {std::move(snapshot), std::move(energy), std::move(monitors), low, high};
    }
    ThermalFields ThermalPostprocessPlan::evaluate(const ThermalNumericState& numeric,
                                                   const Vector<double>& temperature, const Vector<double>* previous,
                                                   const EvaluatedThermalState* previousState, std::optional<double> dt,
                                                   std::uint64_t step, bool initial, bool includeFrame) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::problem::ThermalPostprocessPlan::evaluate",
                                    emberfield::NvtxColor::Yellow);
        return evaluate(numeric.model(), temperature, previous, dt, step, numeric.parameters().time(), initial,
                        includeFrame, &numeric.parameters(), previousState);
    }
} // namespace emberfield::problem
