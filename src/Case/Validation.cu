/** @file Validation.cu
 * @brief 实现物性、选择器、分析模式与运行选项的聚合静态校验。
 */
#include <Case/Validation.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace thermofem::problem {
    double evaluate(const ScalarField& field) noexcept { return std::get<ConstantScalar>(field).value_; }
    void validateLimits(const CaseLimits& limits) {
        if (!limits.maxBytes_ || !limits.maxValues_ || !limits.maxStringBytes_ || !limits.maxNesting_ ||
            !limits.maxSelectorDepth_ || !limits.maxDiagnostics_) {
            throw std::invalid_argument("算例资源限额必须为正");
        }
    }
    namespace {
        /** @brief 聚合静态检查状态，只借用定义，不保存全局状态。
         * @code Checker checker(limits); @endcode */
        struct Checker {
            CaseLimits limits_; ///< 资源限制。
            Diagnostics report_; ///< 结果。
            std::size_t count_{}; ///< 已访问选择器/对象数。
            /** @brief 初始化独立检查状态。@param limits 正限额。@pre 限额合法。
             * @note 无返回值。@code Checker v(limits); @endcode */
            explicit Checker(CaseLimits limits) : limits_(limits), report_(limits.maxDiagnostics_) {}
            /** @brief 追加静态错误。@param code 错误码。@param path 路径。@param message 消息。
             * @pre 无。@note 无返回值。@code error("INVALID","/","非法"); @endcode */
            void error(std::string code, const std::string& path, std::string message) {
                report_.add({Severity::Error, std::move(code), path, {}, {}, std::move(message), "按字段约定修正输入"});
            }
            /** @brief 消耗记录预算。@param path 路径。@param amount 记录数。@pre 无。
             * @return 是否仍在限额内；失败同时追加诊断。@code if(!record("/")) return; @endcode */
            bool record(const std::string& path, std::size_t amount = 1) {
                if (amount > limits_.maxValues_ - count_) {
                    error("RESOURCE_LIMIT", path, "算例对象数超过限额");
                    return false;
                }
                count_ += amount;
                return true;
            }
            /** @brief 检查文本长度。@param value 文本。@param path 路径。@param required 是否非空。
             * @pre 无。@note 无返回值。@code text(id,"/id",true); @endcode */
            void text(const std::string& value, const std::string& path, bool required = false) {
                if ((required && value.empty()) || value.size() > limits_.maxStringBytes_) {
                    error("INVALID_STRING", path, "字符串为空或超过长度限额");
                }
            }
            /** @brief 检查标量范围。@param field 场。@param path 路径。@param range 0有限、1正、2非负。
             * @pre range为0至2。@note 无返回值。@code scalar(k,"/conductivity",1); @endcode */
            void scalar(const ScalarField& field, const std::string& path, int range = 0) {
                const double value = evaluate(field);
                if (!std::isfinite(value) || (range == 1 && value <= 0) || (range == 2 && value < 0)) {
                    error("INVALID_VALUE", path, "字段必须有限并满足物理范围约束");
                }
            }
            /** @brief 检查选择器结构、维数与递归预算。@param r 选择器。@param path 路径。
             * @param expected 期望维数。@param depth 当前深度。@pre depth从1开始。
             * @note 无返回值。@code selector(r,"/domain",2); @endcode */
            void selector(const RegionSelector& r, const std::string& path, int expected, std::size_t depth = 1) {
                if (depth > limits_.maxSelectorDepth_) {
                    error("RESOURCE_LIMIT", path, "选择器过深");
                    return;
                }
                if (!record(path)) {
                    return;
                }
                if (r.kind_ == RegionKind::PhysicalGroup) {
                    if (r.dimension_ != expected || (r.tag_.has_value() == r.name_.has_value()) ||
                        !r.operands_.empty() || (r.tag_ && *r.tag_ <= 0)) {
                        error("INVALID_SELECTOR", path, "物理组维数或身份字段非法");
                    }
                    if (r.name_) {
                        text(*r.name_, path + "/name", true);
                    }
                }
                else {
                    if (r.kind_ != RegionKind::Union && r.kind_ != RegionKind::Intersection &&
                        r.kind_ != RegionKind::Difference) {
                        error("INVALID_SELECTOR", path, "未知选择器类型");
                        return;
                    }
                    if (r.dimension_ != -1 || r.tag_ || r.name_ || r.operands_.empty() ||
                        (r.kind_ == RegionKind::Difference && r.operands_.size() != 2)) {
                        error("INVALID_SELECTOR", path, "集合运算的参数或子项数非法");
                    }
                    if (r.operands_.size() > limits_.maxValues_ - count_) {
                        error("RESOURCE_LIMIT", path, "选择器子项过多");
                        return;
                    }
                    for (std::size_t i = 0; i < r.operands_.size(); ++i) {
                        selector(r.operands_[i], path + "/operands/" + std::to_string(i), expected, depth + 1);
                    }
                }
            }
            /** @brief 检查同类id唯一性。@param value id。@param path 路径。@param ids 当前类别集合。
             * @pre ids仅包含当前类别。@note 无返回值，更新集合。@code id(value,"/id",ids); @endcode */
            void id(const std::string& value, const std::string& path, std::unordered_set<std::string>& ids) {
                text(value, path, true);
                if (!ids.insert(value).second) {
                    error("DUPLICATE_ID", path, "同类id重复");
                }
            }
        };
    } // namespace
    Diagnostics validateCase(const CaseDefinition& c, CaseLimits limits) {
        validateLimits(limits);
        Checker v(limits);
        // 先检查总记录预算，再建立id集合，避免诊断路径触发不受控的二次分配。
        for (auto size : {c.materials_.size(), c.materialAssignments_.size(), c.sources_.size(), c.boundaries_.size(),
                          c.initialState_ ? c.initialState_->overrides_.size() : std::size_t{0}}) {
            if (!v.record("/", size)) {
                return v.report_;
            }
        }
        if (c.schemaVersion_ != 1) {
            v.error("UNSUPPORTED_VERSION", "/schemaVersion", "只支持算例协议1");
        }
        v.text(c.name_, "/name");
        v.text(c.description_, "/description");
        v.text(c.mesh_.path_.generic_string(), "/mesh/path");
        v.text(c.output_.directory_.generic_string(), "/output/directory", true);
        if (c.mesh_.path_.generic_string().find('\0') != std::string::npos ||
            c.output_.directory_.generic_string().find('\0') != std::string::npos) {
            v.error("INVALID_VALUE", "/", "文件路径不得含嵌入NUL字符");
        }
        if (c.mesh_.format_ != "msh") {
            v.error("UNSUPPORTED_CAPABILITY", "/mesh/format", "网格格式必须为msh");
        }
        if (!std::isfinite(c.mesh_.coordinateScale_) || c.mesh_.coordinateScale_ <= 0) {
            v.error("INVALID_VALUE", "/mesh/coordinateScale", "坐标缩放必须有限且严格正");
        }
        if (c.analysis_.type_ != AnalysisType::Steady && c.analysis_.type_ != AnalysisType::Transient) {
            v.error("INVALID_VALUE", "/analysis/type", "未知分析类型");
        }
        if (c.analysis_.space_ != SpaceModel::Planar2d && c.analysis_.space_ != SpaceModel::Solid3d) {
            v.error("UNSUPPORTED_CAPABILITY", "/analysis/space", "未知空间模型");
        }
        const int dimension = c.analysis_.space_ == SpaceModel::Solid3d ? 3 : 2;
        if ((dimension == 2 && !c.analysis_.thickness_) || (dimension == 3 && c.analysis_.thickness_)) {
            v.error("INVALID_VALUE", "/analysis/thickness", "二维必须指定厚度，三维禁止厚度");
        }
        if (c.analysis_.thickness_ && (!std::isfinite(*c.analysis_.thickness_) || *c.analysis_.thickness_ <= 0)) {
            v.error("INVALID_VALUE", "/analysis/thickness", "厚度必须有限且严格正");
        }
        if (!std::isfinite(c.analysis_.geometryTolerance_) || c.analysis_.geometryTolerance_ < 0) {
            v.error("INVALID_VALUE", "/analysis/geometryTolerance", "几何容差必须有限且非负");
        }
        v.selector(c.analysis_.domain_, "/analysis/domain", dimension);
        if (c.materials_.empty()) {
            v.error("MATERIAL_GAP", "/materials", "材料表不能为空");
        }
        if (c.materialAssignments_.empty()) {
            v.error("MATERIAL_GAP", "/materialAssignments", "材料分配不能为空");
        }
        std::unordered_set<std::string> materials;
        for (std::size_t i = 0; i < c.materials_.size(); ++i) {
            const auto& m = c.materials_[i];
            const auto p = "/materials/" + std::to_string(i);
            v.id(m.id_, p + "/id", materials);
            v.text(m.displayName_, p + "/displayName");
            v.scalar(m.conductivity_, p + "/conductivity", 1);
            if (m.density_.has_value() != m.specificHeat_.has_value() || (m.density_ && m.volumetricHeatCapacity_)) {
                v.error("INVALID_VALUE", p, "热容量两种表示互斥，密度与比热必须成对");
            }
            if (m.density_) {
                v.scalar(*m.density_, p + "/density", 1);
            }
            if (m.specificHeat_) {
                v.scalar(*m.specificHeat_, p + "/specificHeat", 1);
            }
            if (m.volumetricHeatCapacity_) {
                v.scalar(*m.volumetricHeatCapacity_, p + "/volumetricHeatCapacity", 1);
            }
            if (m.density_ && m.specificHeat_ &&
                (!std::isfinite(evaluate(*m.density_) * evaluate(*m.specificHeat_)) ||
                 evaluate(*m.density_) * evaluate(*m.specificHeat_) <= 0)) {
                v.error("INVALID_VALUE", p, "体热容量乘积溢出、下溢为零或非正");
            }
            if (c.analysis_.type_ == AnalysisType::Transient && !m.volumetricHeatCapacity_ &&
                !(m.density_ && m.specificHeat_)) {
                v.error("MISSING_FIELD", p, "瞬态材料必须提供热容量");
            }
        }
        std::unordered_set<std::string> ids;
        for (std::size_t i = 0; i < c.materialAssignments_.size(); ++i) {
            const auto& a = c.materialAssignments_[i];
            const auto p = "/materialAssignments/" + std::to_string(i);
            v.id(a.id_, p + "/id", ids);
            v.selector(a.region_, p + "/region", dimension);
            v.text(a.material_, p + "/material", true);
            if (!materials.contains(a.material_)) {
                v.error("UNKNOWN_MATERIAL", p + "/material", "引用的材料不存在");
            }
        }
        ids.clear();
        for (std::size_t i = 0; i < c.sources_.size(); ++i) {
            const auto& s = c.sources_[i];
            const auto p = "/sources/" + std::to_string(i);
            v.id(s.id_, p + "/id", ids);
            v.selector(s.region_, p + "/region", dimension);
            v.scalar(s.value_, p + "/value");
        }
        ids.clear();
        for (std::size_t i = 0; i < c.boundaries_.size(); ++i) {
            const auto& b = c.boundaries_[i];
            const auto p = "/boundaries/" + std::to_string(i);
            v.id(b.id_, p + "/id", ids);
            v.selector(b.region_, p + "/region", dimension - 1);
            if (const auto* t = std::get_if<TemperatureBoundary>(&b.value_)) {
                v.scalar(t->temperature_, p + "/temperature", 2);
            }
            if (const auto* f = std::get_if<FluxBoundary>(&b.value_)) {
                v.scalar(f->flux_, p + "/flux");
            }
            if (const auto* h = std::get_if<ConvectionBoundary>(&b.value_)) {
                v.scalar(h->coefficient_, p + "/coefficient", 2);
                v.scalar(h->ambientTemperature_, p + "/ambientTemperature", 2);
            }
        }
        if (c.naturalOverlap_ != NaturalOverlap::Error && c.naturalOverlap_ != NaturalOverlap::Add) {
            v.error("INVALID_VALUE", "/boundaryPolicy/naturalOverlap", "未知重叠策略");
        }
        const bool transient = c.analysis_.type_ == AnalysisType::Transient;
        if (transient != c.initialState_.has_value() || transient != c.time_.has_value()) {
            v.error("INVALID_MODE", "/", "瞬态必须提供初温和时间，稳态禁止两者");
        }
        if (c.initialState_) {
            v.scalar(c.initialState_->defaultTemperature_, "/initialState/defaultTemperature", 2);
            ids.clear();
            for (std::size_t i = 0; i < c.initialState_->overrides_.size(); ++i) {
                const auto& o = c.initialState_->overrides_[i];
                const auto p = "/initialState/overrides/" + std::to_string(i);
                v.id(o.id_, p + "/id", ids);
                v.selector(o.region_, p + "/region", dimension);
                v.scalar(o.temperature_, p + "/temperature", 2);
            }
        }
        if (c.time_) {
            const auto& t = *c.time_;
            if (!std::isfinite(t.start_) || !std::isfinite(t.end_) || !std::isfinite(t.step_) || t.step_ <= 0 ||
                t.end_ <= t.start_ || !t.maxSteps_) {
                v.error("INVALID_VALUE", "/time", "时间区间、步长或最大步数非法");
            }
            else {
                const long double count = std::ceil((static_cast<long double>(t.end_) - t.start_) / t.step_);
                if (!std::isfinite(count) || count > static_cast<long double>(t.maxSteps_)) {
                    v.error("RESOURCE_LIMIT", "/time", "计划时间步数超过maxSteps");
                }
                if (std::min(t.start_ + t.step_, t.end_) <= t.start_ ||
                    std::max(t.end_ - t.step_, t.start_) >= t.end_) {
                    v.error("INVALID_VALUE", "/time/step", "步长在当前时刻精度下无法推进");
                }
            }
        }
        if (!c.discretization_.order_) {
            v.error("INVALID_VALUE", "/discretization/order", "近似阶次必须为正整数");
        }
        const auto& s = c.linearSolver_;
        if (s.method_ != LinearMethod::BiCgStab && s.method_ != LinearMethod::Cg) {
            v.error("UNSUPPORTED_CAPABILITY", "/linearSolver/method", "未知线性方法");
        }
        if (!std::isfinite(s.relativeTolerance_) || !std::isfinite(s.absoluteTolerance_) || s.relativeTolerance_ < 0 ||
            s.absoluteTolerance_ < 0 || (s.relativeTolerance_ == 0 && s.absoluteTolerance_ == 0) || !s.maxIterations_) {
            v.error("INVALID_VALUE", "/linearSolver", "容差或迭代上限非法");
        }
        if ((s.initialGuess_ != InitialGuess::Zero && s.initialGuess_ != InitialGuess::PreviousStep) ||
            (!transient && s.initialGuess_ == InitialGuess::PreviousStep)) {
            v.error("INVALID_MODE", "/linearSolver/initialGuess", "当前分析不支持该初猜策略");
        }
        if (!transient && (c.output_.everySteps_ || c.output_.includeInitial_ || !c.output_.includeFinal_)) {
            v.error("INVALID_MODE", "/output", "稳态禁止时间频率和初始输出，且必须输出最终状态");
        }
        if (c.output_.everySteps_ && !*c.output_.everySteps_) {
            v.error("INVALID_VALUE", "/output/schedule/interval", "输出步数间隔必须为正");
        }
        return v.report_;
    }
} // namespace thermofem::problem
