/** @file Vtu.cu
 * @brief 实现受检的ASCII和小端UInt64块头原始追加VTU编码。
 */
#include <IO/Result/Vtu.cuh>
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <set>
#include <stdexcept>
#include "Encoding.cuh"

namespace emberfield::io {
    namespace {
        using field::NumericArray;
        /** @brief 输出数组的几何来源类别。 */
        enum class PayloadKind { Numeric, Points, Connectivity, Offsets, Types };
        /** @brief 借用载荷描述，不复制完整场或连接数组。@code Payload p; @endcode */
        struct Payload {
            std::string name_; ///< 数组名。
            std::string type_; ///< VTK标量类型。
            std::uint32_t components_{1}; ///< 元组分量。
            std::uint64_t tuples_{}; ///< 元组数。
            std::uint64_t bytes_{}; ///< 载荷字节数。
            PayloadKind kind_{PayloadKind::Numeric}; ///< 数据源。
            const NumericArray* array_{}; ///< Numeric时借用。
        };
        /** @brief 检查字节累加。@param a 左数。@param b 右数。@pre 无。
         * @return 无溢出和，否则length_error。@code auto n=add(a,b); @endcode */
        std::uint64_t add(std::uint64_t a, std::uint64_t b) {
            if (a > std::numeric_limits<std::uint64_t>::max() - b) {
                throw std::length_error("VTU字节偏移溢出");
            }
            return a + b;
        }
        /** @brief 检查字节乘法。@param a 数量。@param b 单项字节。@pre 无。
         * @return 无溢出乘积。@code auto n=product(n,8); @endcode */
        std::uint64_t product(std::uint64_t a, std::uint64_t b) {
            if (b && a > std::numeric_limits<std::uint64_t>::max() / b) {
                throw std::length_error("VTU载荷字节数溢出");
            }
            return a * b;
        }
        /** @brief 获取必需数组并验证类型形状。
         * @param arrays 数组集合。@param name 名字。@param variant 标量变体下标。@param components 分量。
         * @pre 无。@return 借用数组；缺失或不相容invalid_argument。
         * @code auto& a=require(arrays,"temperature",4,1); @endcode */
        const NumericArray& require(std::span<const NumericArray> arrays, std::string_view name, std::size_t variant,
                                    std::uint32_t components) {
            for (const auto& a : arrays) {
                if (a.name_ == name) {
                    if (a.values_.index() != variant || a.components_ != components) {
                        throw std::invalid_argument("VTU必需数组类型或分量不符: " + std::string(name));
                    }
                    return a;
                }
            }
            throw std::invalid_argument("VTU缺失必需数组: " + std::string(name));
        }
        /** @brief 获取精确布局对应的VTK编号。@param t 中立类型。@pre 无。
         * @return 三角形5或四面体10；不支持抛domain_error。@code auto c=cellType(t); @endcode */
        std::uint8_t cellType(const mesh::ElementType& t) {
            if (t.order_ == 1 && t.referenceLayout_ == "complete") {
                if (t.family_ == mesh::CellFamily::Triangle && t.dimension_ == 2 && t.nodeCount_ == 3) {
                    return 5;
                }
                if (t.family_ == mesh::CellFamily::Tetrahedron && t.dimension_ == 3 && t.nodeCount_ == 4) {
                    return 10;
                }
            }
            throw std::domain_error("OUTPUT_UNSUPPORTED_ELEMENT: VTU无精确单元映射");
        }
        /** @brief 在开始写文件前校验EmberField结果字段。
         * @param s 快照。@pre 来源有效。@note 无返回值；非法invalid_argument。
         * @code validate(s); @endcode */
        void validate(const field::FieldSnapshot& s) {
            const auto& g = *s.grid();
            const auto& t = require(s.pointFields(), "temperature", 4, 1);
            if (t.unit_ != "K") {
                throw std::invalid_argument("温度单位必须为K");
            }
            for (double v : std::get<std::vector<double>>(t.values_)) {
                if (v < 0) {
                    throw std::invalid_argument("绝对温度不能为负");
                }
            }
            require(g.pointIdentity(), "sourceNodeTag", 3, 1);
            require(g.pointIdentity(), "dofIndex", 3, 1);
            for (const auto name : {"sourceCellTag", "sourceCellRow", "materialIndex"}) {
                require(g.cellIdentity(), name, 3, 1);
            }
            require(g.cellIdentity(), "hasEntity", 0, 1);
            require(g.cellIdentity(), "entityDimension", 0, 1);
            require(g.cellIdentity(), "entityTag", 2, 1);
            const auto& schema = require(g.metadata(), "emberfieldSchemaVersion", 1, 1);
            const auto& dims = require(g.metadata(), "dimension", 0, 1);
            const auto& scale = require(g.metadata(), "coordinateScale", 4, 1);
            if (schema.tuples_ != 1 || std::get<std::vector<std::uint32_t>>(schema.values_)[0] != 1 ||
                dims.tuples_ != 1 || scale.tuples_ != 1 || std::get<std::vector<double>>(scale.values_)[0] <= 0) {
                throw std::invalid_argument("VTU静态元数据不合法");
            }
            const auto dimension = std::get<std::vector<std::uint8_t>>(dims.values_)[0];
            if (dimension != 2 && dimension != 3) {
                throw std::invalid_argument("VTU空间维数不合法");
            }
            if (dimension == 2) {
                const auto& h = require(g.metadata(), "thickness", 4, 1);
                if (h.tuples_ != 1 || std::get<std::vector<double>>(h.values_)[0] <= 0) {
                    throw std::invalid_argument("二维厚度不合法");
                }
            }
            for (const auto& a : g.metadata()) {
                if (a.name_ == "stepIndex" || a.name_ == "sampleKind" || a.name_ == "TimeValue") {
                    throw std::invalid_argument("静态元数据与状态字段重名");
                }
            }
            const auto& offsets =
                std::get<std::vector<std::uint64_t>>(require(g.metadata(), "physicalGroupOffsets", 3, 1).values_);
            const auto& tags = require(g.metadata(), "physicalGroupTags", 2, 1);
            const auto& groups = require(g.metadata(), "physicalGroupDimensions", 0, 1);
            if (offsets.size() != g.cellCount() + 1 || offsets.front() != 0 || offsets.back() != tags.tuples_ ||
                tags.tuples_ != groups.tuples_ || !std::is_sorted(offsets.begin(), offsets.end())) {
                throw std::invalid_argument("VTU物理组CSR不合法");
            }
            for (const auto& a : s.cellFields()) {
                if (a.name_ == "temperatureGradient" || a.name_ == "heatFlux") {
                    if (a.components_ != 3 || a.values_.index() != 4 ||
                        a.unit_ != (a.name_ == "heatFlux" ? "W/m^2" : "K/m")) {
                        throw std::invalid_argument("VTU矢量场形状或单位不符");
                    }
                }
            }
            const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
            if (g.pointCount() > limit || g.cellCount() > limit || g.connectivity().size() > limit) {
                throw std::length_error("VTU拓扑超出Int64范围");
            }
            for (const auto& layout : g.layouts()) {
                (void)cellType(layout);
                if (layout.dimension_ != dimension) {
                    throw std::invalid_argument("VTU布局维数与元数据不同");
                }
            }
        }
        /** @brief 借用NumericArray生成载荷描述。@param a 数组。@pre 数组已验证。
         * @return 载荷。@code auto p=payload(a); @endcode */
        Payload payload(const NumericArray& a) {
            constexpr std::array<const char*, 5> names{"UInt8", "UInt32", "Int64", "UInt64", "Float64"};
            return {a.name_, names.at(a.values_.index()), a.components_, a.tuples_, a.sizeBytes(), PayloadKind::Numeric,
                    &a};
        }
        /** @brief 分块编码小端二进制，避免逐标量系统写入。@code Emitter out(stream,true); @endcode */
        class Emitter {
        public:
            /** @brief 借用流。@param stream 流。@param binary 二进制开关。@pre 流有效。@note 无返回值。
             * @code Emitter out(s,true); @endcode */
            Emitter(std::ostream& stream, bool binary) : stream_(stream), binary_(binary) {}
            /** @brief 编码一个数。@tparam T 算术类型。@param value 值。@pre 有限/范围已校验。
             * @note 无返回值。@code out.scalar(3.); @endcode */
            template <class T>
                requires(std::integral<T> || std::floating_point<T>)
            void scalar(T value) {
                if (!binary_) {
                    stream_ << +value << ' ';
                    return;
                }
                auto bytes = std::bit_cast<std::array<char, sizeof(T)>>(value);
                if constexpr (std::endian::native == std::endian::big) {
                    std::reverse(bytes.begin(), bytes.end());
                }
                if (used_ + bytes.size() > buffer_.size()) {
                    flush();
                }
                std::copy(bytes.begin(), bytes.end(), buffer_.begin() + static_cast<std::ptrdiff_t>(used_));
                used_ += bytes.size();
            }
            /** @brief 写出尾缓冲，无参数无前提。@note 无返回值，IO错误抛异常。@code out.flush(); @endcode */
            void flush() {
                if (used_) {
                    stream_.write(buffer_.data(), static_cast<std::streamsize>(used_));
                    used_ = 0;
                }
                detail::checked(stream_);
            }

        private:
            std::ostream& stream_; ///< 借用流。
            bool binary_; ///< 编码方式。
            std::array<char, 65536> buffer_{}; ///< 有界缓冲。
            std::size_t used_{}; ///< 已用字节。
        };
        /** @brief 写载荷数值。@param stream 流。@param p 载荷。@param g 网格。@param binary 编码。
         * @pre 已完整校验。@note 无返回值，写失败抛IO异常。
         * @code writeValues(s,p,g,true); @endcode */
        void writeValues(std::ostream& stream, const Payload& p, const field::VisualizationGrid& g, bool binary) {
            Emitter out(stream, binary);
            if (binary) {
                out.scalar(p.bytes_);
            }
            switch (p.kind_) {
            case PayloadKind::Numeric:
                std::visit(
                    [&]<class V>(const V& values)
                        requires requires { typename V::value_type; }
                    {
                        for (auto value : values) {
                            out.scalar(value);
                        }
                    },
                    p.array_->values_);
                break;
            case PayloadKind::Points:
                for (const auto& point : g.points()) {
                    for (double value : point) {
                        out.scalar(value);
                    }
                }
                break;
            case PayloadKind::Connectivity:
                for (auto value : g.connectivity()) {
                    out.scalar(static_cast<std::int64_t>(value));
                }
                break;
            case PayloadKind::Offsets:
                for (auto value : g.offsets().subspan(1)) {
                    out.scalar(static_cast<std::int64_t>(value));
                }
                break;
            case PayloadKind::Types:
                for (const auto& type : g.layouts()) {
                    out.scalar(cellType(type));
                }
                break;
            }
            out.flush();
        }
    } // namespace
    void writeVtu(std::ostream& stream, const field::FieldSnapshot& s, VtuEncoding encoding) {
        if (encoding != VtuEncoding::Ascii && encoding != VtuEncoding::AppendedRaw) {
            throw std::invalid_argument("未知VTU编码");
        }
        validate(s);
        const auto& g = *s.grid();
        std::vector<NumericArray> state{
            {"sampleKind", 1, 1, "1",
             std::vector<std::uint8_t>{static_cast<std::uint8_t>(s.kind() == field::SampleKind::Initial ? 0 : 1)}},
            {"stepIndex", 1, 1, "1", std::vector<std::uint64_t>{s.step()}}};
        if (s.time()) {
            state.push_back({"TimeValue", 1, 1, "s", std::vector<double>{*s.time()}});
        }
        std::vector<Payload> metadata, points, cells, geometry, all;
        for (const auto& a : g.metadata()) {
            metadata.push_back(payload(a));
        }
        for (const auto& a : state) {
            metadata.push_back(payload(a));
        }
        for (const auto& a : g.pointIdentity()) {
            points.push_back(payload(a));
        }
        for (const auto& a : s.pointFields()) {
            points.push_back(payload(a));
        }
        for (const auto& a : g.cellIdentity()) {
            cells.push_back(payload(a));
        }
        for (const auto& a : s.cellFields()) {
            cells.push_back(payload(a));
        }
        geometry.push_back({"points", "Float64", 3, g.pointCount(), product(g.pointCount(), 24), PayloadKind::Points});
        geometry.push_back({"connectivity", "Int64", 1, g.connectivity().size(), product(g.connectivity().size(), 8),
                            PayloadKind::Connectivity});
        geometry.push_back({"offsets", "Int64", 1, g.cellCount(), product(g.cellCount(), 8), PayloadKind::Offsets});
        geometry.push_back({"types", "UInt8", 1, g.cellCount(), g.cellCount(), PayloadKind::Types});
        for (const auto* group : {&metadata, &points, &cells, &geometry}) {
            all.insert(all.end(), group->begin(), group->end());
        }
        std::uint64_t total = 0;
        for (const auto& p : all) {
            total = add(total, add(8, p.bytes_));
        }
        if (total > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            throw std::length_error("VTU文件超出流位置范围");
        }
        detail::StreamFormat guard(stream);
        const bool binary = encoding == VtuEncoding::AppendedRaw;
        std::uint64_t offset = 0;
        const auto writeArray = [&](const Payload& p, bool fieldData) {
            stream << "      <DataArray type=\"" << p.type_ << "\" Name=\"" << detail::xml(p.name_)
                   << "\" NumberOfComponents=\"" << p.components_ << "\"";
            if (fieldData) {
                stream << " NumberOfTuples=\"" << p.tuples_ << "\"";
            }
            if (binary) {
                stream << " format=\"appended\" offset=\"" << offset << "\"/>\n";
                offset = add(offset, add(8, p.bytes_));
            }
            else {
                stream << " format=\"ascii\">";
                writeValues(stream, p, g, false);
                stream << "</DataArray>\n";
            }
        };
        stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                  "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" "
                  "header_type=\"UInt64\">\n"
                  "  <UnstructuredGrid>\n    <FieldData>\n";
        for (const auto& p : metadata) {
            writeArray(p, true);
        }
        stream << "    </FieldData>\n    <Piece NumberOfPoints=\"" << g.pointCount() << "\" NumberOfCells=\""
               << g.cellCount()
               << "\">\n"
                  "      <PointData Scalars=\"temperature\">\n";
        for (const auto& p : points) {
            writeArray(p, false);
        }
        stream << "      </PointData>\n      <CellData";
        const auto hasField = [&](std::string_view name) {
            return std::any_of(s.cellFields().begin(), s.cellFields().end(),
                               [&](const NumericArray& a) { return a.name_ == name; });
        };
        if (hasField("heatFlux")) {
            stream << " Vectors=\"heatFlux\"";
        }
        else if (hasField("temperatureGradient")) {
            stream << " Vectors=\"temperatureGradient\"";
        }
        stream << ">\n";
        for (const auto& p : cells) {
            writeArray(p, false);
        }
        stream << "      </CellData>\n      <Points>\n";
        writeArray(geometry[0], false);
        stream << "      </Points>\n      <Cells>\n";
        for (std::size_t i = 1; i < geometry.size(); ++i) {
            writeArray(geometry[i], false);
        }
        stream << "      </Cells>\n    </Piece>\n  </UnstructuredGrid>\n";
        if (binary) {
            stream << "  <AppendedData encoding=\"raw\">_";
            for (const auto& p : all) {
                writeValues(stream, p, g, true);
            }
            stream << "</AppendedData>\n";
        }
        stream << "</VTKFile>\n";
        detail::checked(stream);
    }
} // namespace emberfield::io
