/** @file Snapshot.cu
 * @brief 实现类型化数组、UTF-8文本、可视化拓扑与快照的完整性检查。
 */
#include <DataStructure/Field/Snapshot.cuh>
#include <NVTXProfiler.cuh>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace emberfield::field {
    namespace {
        /** @brief 校验数组集合和名称。@param arrays 数组。@param tuples 所需元组数，空表示任意。
         * @param names 已使用名称。@pre 无。@note 无返回值，错误invalid_argument。
         * @code arraysChecked(a,3,names); @endcode */
        void arraysChecked(std::span<const NumericArray> arrays, std::optional<std::uint64_t> tuples,
                           std::set<std::string>& names) {
            for (const auto& a : arrays) {
                a.validate();
                if ((tuples && a.tuples_ != *tuples) || !names.insert(a.name_).second) {
                    throw std::invalid_argument("场数组元组数不匹配或名称重复");
                }
            }
        }
    } // namespace
    void validateText(std::string_view text) {
        for (std::size_t i = 0; i < text.size();) {
            const auto first = static_cast<unsigned char>(text[i++]);
            std::uint32_t code = first;
            unsigned following = 0;
            if (first >= 0xC2 && first <= 0xDF) {
                code = first & 0x1F;
                following = 1;
            }
            else if (first >= 0xE0 && first <= 0xEF) {
                code = first & 0x0F;
                following = 2;
            }
            else if (first >= 0xF0 && first <= 0xF4) {
                code = first & 0x07;
                following = 3;
            }
            else if (first >= 0x80) {
                throw std::invalid_argument("非法UTF-8首字节");
            }
            if (following > text.size() - i) {
                throw std::invalid_argument("UTF-8字符截断");
            }
            for (unsigned j = 0; j < following; ++j) {
                const auto next = static_cast<unsigned char>(text[i++]);
                if ((next & 0xC0) != 0x80) {
                    throw std::invalid_argument("非法UTF-8续字节");
                }
                code = (code << 6) | (next & 0x3F);
            }
            if ((following == 1 && code < 0x80) || (following == 2 && code < 0x800) ||
                (following == 3 && code < 0x10000) ||
                !(code == 9 || code == 10 || code == 13 || (code >= 0x20 && code <= 0xD7FF) ||
                  (code >= 0xE000 && code <= 0xFFFD) || (code >= 0x10000 && code <= 0x10FFFF))) {
                throw std::invalid_argument("文本含非法UTF-8或XML控制字符");
            }
        }
    }
    std::uint64_t NumericArray::sizeBytes() const {
        return std::visit(
            []<class V>(const V& v)
                requires requires { typename V::value_type; }
            {
                if (v.size() > std::numeric_limits<std::uint64_t>::max() / sizeof(typename V::value_type)) {
                    throw std::length_error("场数组字节数溢出");
                }
                return static_cast<std::uint64_t>(v.size()) * sizeof(typename V::value_type);
            },
            values_);
    }
    void NumericArray::validate() const {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::field::NumericArray::validate", emberfield::NvtxColor::SkyBlue);
        validateText(name_);
        validateText(unit_);
        if (name_.empty() || !components_ || tuples_ > std::numeric_limits<std::uint64_t>::max() / components_) {
            throw std::invalid_argument("场数组名称或形状非法");
        }
        const auto count = tuples_ * components_;
        std::visit(
            [&]<class V>(const V& v)
                requires requires { typename V::value_type; }
            {
                if (v.size() != count) {
                    throw std::invalid_argument("场数组形状不一致");
                }
                if constexpr (std::same_as<typename V::value_type, double>) {
                    for (double value : v) {
                        if (!std::isfinite(value)) {
                            throw std::invalid_argument("场数组含非有限值");
                        }
                    }
                }
            },
            values_);
        (void)sizeBytes();
    }
    VisualizationGrid::VisualizationGrid(std::vector<std::array<double, 3>> points, std::vector<std::uint64_t> offsets,
                                         std::vector<std::uint64_t> connectivity,
                                         std::vector<mesh::ElementType> layouts,
                                         std::vector<NumericArray> pointIdentity,
                                         std::vector<NumericArray> cellIdentity, std::vector<NumericArray> metadata) :
        points_(std::move(points)), offsets_(std::move(offsets)), connectivity_(std::move(connectivity)),
        layouts_(std::move(layouts)), pointIdentity_(std::move(pointIdentity)), cellIdentity_(std::move(cellIdentity)),
        metadata_(std::move(metadata)) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::field::VisualizationGrid::VisualizationGrid",
                                    emberfield::NvtxColor::SkyBlue);
        if (points_.empty() || layouts_.empty() || offsets_.size() != layouts_.size() + 1 || offsets_.front() != 0 ||
            offsets_.back() != connectivity_.size()) {
            throw std::invalid_argument("输出网格为空或CSR形状非法");
        }
        for (const auto& p : points_) {
            for (double x : p) {
                if (!std::isfinite(x)) {
                    throw std::invalid_argument("坐标非有限");
                }
            }
        }
        for (std::size_t i = 0; i < layouts_.size(); ++i) {
            if (offsets_[i] >= offsets_[i + 1] || offsets_[i + 1] - offsets_[i] != layouts_[i].nodeCount_) {
                throw std::invalid_argument("单元布局与连接长度不一致");
            }
        }
        for (auto point : connectivity_) {
            if (point >= points_.size()) {
                throw std::invalid_argument("输出连接越界");
            }
        }
        std::set<std::string> names;
        arraysChecked(pointIdentity_, points_.size(), names);
        names.clear();
        arraysChecked(cellIdentity_, layouts_.size(), names);
        names.clear();
        arraysChecked(metadata_, std::nullopt, names);
    }
    FieldSnapshot::FieldSnapshot(std::shared_ptr<const VisualizationGrid> grid, std::uint64_t step,
                                 std::optional<double> time, SampleKind kind, std::vector<NumericArray> points,
                                 std::vector<NumericArray> cells) :
        grid_(std::move(grid)), step_(step), time_(time), kind_(kind), pointFields_(std::move(points)),
        cellFields_(std::move(cells)) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::field::FieldSnapshot::FieldSnapshot", emberfield::NvtxColor::SkyBlue);
        if (!grid_ || (time_ && !std::isfinite(*time_)) ||
            (kind_ != SampleKind::Initial && kind_ != SampleKind::Accepted)) {
            throw std::invalid_argument("场快照来源或状态非法");
        }
        std::set<std::string> names;
        arraysChecked(grid_->pointIdentity(), grid_->pointCount(), names);
        arraysChecked(pointFields_, grid_->pointCount(), names);
        names.clear();
        arraysChecked(grid_->cellIdentity(), grid_->cellCount(), names);
        arraysChecked(cellFields_, grid_->cellCount(), names);
    }
} // namespace emberfield::field
