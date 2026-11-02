/** @file Case.cu
 * @brief 适配picojson的限额/重复键校验回调，执行类型化协议转换。
 */
#include <IO/Case/Case.cuh>
#define PICOJSON_USE_INT64
#include <algorithm>
#include <charconv>
#include <cmath>
#include <concepts>
#include <fstream>
#include <limits>
#include <picojson.h>
#include <set>
#include <stdexcept>
#include <string_view>

namespace thermofem::io {
    namespace {
        using namespace problem;
        using Json = picojson::value;
        /** @brief 内部输入失败，仅在IO边界转为诊断。
         * @code throw InputFailure{"INVALID_VALUE","/time","时间非法",0}; @endcode */
        struct InputFailure {
            std::string code_; ///< 稳定错误码。
            std::string path_; ///< JSON Pointer。
            std::string message_; ///< 中文解释。
            std::optional<std::size_t> offset_; ///< 可选原始字节偏移。
        };
        /** @brief 转义JSON Pointer成员。@param key 原始键。@pre 无。
         * @return 规范化路径片段。@code auto k=escapeKey("a/b"); @endcode */
        std::string escapeKey(const std::string& key) {
            std::string result;
            for (char ch : key) {
                if (ch == '~') {
                    result += "~0";
                }
                else if (ch == '/') {
                    result += "~1";
                }
                else {
                    result += ch;
                }
            }
            return result;
        }
        /** @brief 根据原文字节偏移确定来源位置。@param text 原文。@param offset 偏移。
         * @param source 来源名。@pre 无。@return 一基行列。@code auto p=location(text,0,"case.json"); @endcode */
        SourceLocation location(const std::string& text, std::size_t offset, const std::string& source) {
            SourceLocation result{source, 1, 1};
            for (std::size_t i = 0; i < std::min(offset, text.size()); ++i) {
                if (text[i] == '\n') {
                    ++result.line_;
                    result.column_ = 1;
                }
                else {
                    ++result.column_;
                }
            }
            return result;
        }
        /** @brief 检查UTF-8、严格数字词法与整数范围，解析语法仍交给picojson。
         * @param text 原文。@pre 经过字节限额检查。
         * @note 无返回值；失败抛InputFailure。@code lexicalCheck(text); @endcode */
        void lexicalCheck(const std::string& text) {
            // picojson保留int64但会将溢出整数退化为double，预检阻止这种有损转换。
            for (std::size_t i = 0; i < text.size();) {
                const auto c = static_cast<unsigned char>(text[i]);
                if (c < 0x80) {
                    ++i;
                    continue;
                }
                const unsigned length = c >= 0xc2 && c <= 0xdf ? 2
                    : c >= 0xe0 && c <= 0xef                   ? 3
                    : c >= 0xf0 && c <= 0xf4                   ? 4
                                                               : 0;
                if (!length || length > text.size() - i) {
                    throw InputFailure{"INVALID_UTF8", "", "无效UTF-8", i};
                }
                std::uint32_t cp = c & ((1u << (7 - length)) - 1);
                for (unsigned j = 1; j < length; ++j) {
                    const auto next = static_cast<unsigned char>(text[i + j]);
                    if ((next & 0xc0) != 0x80) {
                        throw InputFailure{"INVALID_UTF8", "", "无效UTF-8续字节", i + j};
                    }
                    cp = (cp << 6) | (next & 0x3f);
                }
                if ((length == 2 && cp < 0x80) || (length == 3 && cp < 0x800) || (length == 4 && cp < 0x10000) ||
                    (cp >= 0xd800 && cp <= 0xdfff) || cp > 0x10ffff) {
                    throw InputFailure{"INVALID_UTF8", "", "非法Unicode码点", i};
                }
                i += length;
            }
            bool quoted = false;
            for (std::size_t i = 0; i < text.size(); ++i) {
                const char c = text[i];
                if (quoted) {
                    if (c == '\\') {
                        ++i;
                        continue;
                    }
                    if (c == '"') {
                        quoted = false;
                    }
                    continue;
                }
                if (c == '"') {
                    quoted = true;
                    continue;
                }
                if (c != '-' && (c < '0' || c > '9')) {
                    continue;
                }
                const auto begin = i;
                if (text[i] == '-') {
                    ++i;
                }
                if (i == text.size() || text[i] < '0' || text[i] > '9') {
                    throw InputFailure{"JSON_SYNTAX", "", "数字缺少整数部分", begin};
                }
                if (text[i] == '0') {
                    ++i;
                }
                else {
                    while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
                        ++i;
                    }
                }
                bool integer = true;
                if (i < text.size() && text[i] == '.') {
                    integer = false;
                    ++i;
                    const auto start = i;
                    while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
                        ++i;
                    }
                    if (start == i) {
                        throw InputFailure{"JSON_SYNTAX", "", "数字小数部分缺失", begin};
                    }
                }
                if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
                    integer = false;
                    ++i;
                    if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
                        ++i;
                    }
                    const auto start = i;
                    while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
                        ++i;
                    }
                    if (start == i) {
                        throw InputFailure{"JSON_SYNTAX", "", "数字指数缺失", begin};
                    }
                }
                if (i < text.size() && text[i] != ',' && text[i] != ']' && text[i] != '}' && text[i] != ' ' &&
                    text[i] != '\t' && text[i] != '\r' && text[i] != '\n') {
                    throw InputFailure{"JSON_SYNTAX", "", "非法数字词法", begin};
                }
                if (integer) {
                    std::int64_t value{};
                    const auto result = std::from_chars(text.data() + begin, text.data() + i, value);
                    if (result.ec != std::errc{}) {
                        throw InputFailure{"INTEGER_RANGE", "", "整数超过有符号64位范围；不会转换为double", begin};
                    }
                }
                --i;
            }
        }
        /** @brief picojson递归解析的共享预算与来源表。
         * @code ParseState state{limits,text,source,locations}; @endcode */
        struct ParseState {
            CaseLimits limits_; ///< 调用者限额。
            const std::string& text_; ///< 原文借用。
            const std::string& source_; ///< 来源借用。
            std::map<std::string, SourceLocation>& locations_; ///< 输出位置表。
            std::size_t values_{}; ///< 累计JSON值。
            std::vector<std::size_t> lineStarts_{0}; ///< 一次扫描建立的行起点，避免每个值重复扫描全文。
            /** @brief 借用输入并建立行索引。@param limits 限额。@param text 原文。
             * @param source 来源。@param locations 位置输出。@pre 借用对象存活。
             * @note 无返回值。@code ParseState state(limits,text,source,locations); @endcode */
            ParseState(CaseLimits limits, const std::string& text, const std::string& source,
                       std::map<std::string, SourceLocation>& locations) :
                limits_(limits), text_(text), source_(source), locations_(locations) {
                for (std::size_t i = 0; i < text.size(); ++i) {
                    if (text[i] == '\n') {
                        lineStarts_.push_back(i + 1);
                    }
                }
            }
            /** @brief 记录节点并检查预算。@param path 路径。@param depth 深度。
             * @param offset 原文偏移。@pre 无。@note 无返回值，超限抛InputFailure。
             * @code state.enter("",1,0); @endcode */
            void enter(const std::string& path, std::size_t depth, std::size_t offset) {
                if (depth > limits_.maxNesting_ || values_++ >= limits_.maxValues_) {
                    throw InputFailure{"RESOURCE_LIMIT", path, "JSON嵌套或值数量超过限额", offset};
                }
                while (
                    offset < text_.size() &&
                    (text_[offset] == ' ' || text_[offset] == '\t' || text_[offset] == '\r' || text_[offset] == '\n')) {
                    ++offset;
                }
                const auto it = std::upper_bound(lineStarts_.begin(), lineStarts_.end(), offset);
                const auto line = static_cast<std::size_t>(it - lineStarts_.begin());
                locations_.emplace(path, SourceLocation{source_, line, offset - lineStarts_[line - 1] + 1});
            }
        };
        /** @brief 复用picojson解析引擎的受限回调；拒绝对象重复键。
         * @details 只适配库提供的解析上下文，不实现另一套JSON语法。
         * @code StrictContext context(value,state,"",1); @endcode */
        class StrictContext : public picojson::default_parse_context {
        public:
            /** @brief 借用解析输出与共享预算。@param output 输出。@param state 共享状态。
             * @param path 路径。@param depth 深度。@pre 借用对象存活。@note 无返回值。
             * @code StrictContext ctx(value,state,"",1); @endcode */
            StrictContext(Json& output, ParseState& state, std::string path, std::size_t depth) :
                picojson::default_parse_context(&output), state_(state), path_(std::move(path)), depth_(depth) {}
            /** @brief 拒绝溢出的浮点输入。@param value 已解析数值。@pre 无。
             * @return 成功标志；非有限数抛InputFailure。@code ctx.set_number(1.0); @endcode */
            bool set_number(double value) {
                if (!std::isfinite(value)) {
                    throw InputFailure{"INVALID_VALUE", path_, "JSON数值溢出", {}};
                }
                return picojson::default_parse_context::set_number(value);
            }
            /** @brief 复用库的字符串/Unicode解析并限制长度。
             * @tparam Iter 输入迭代器。@param input 借用输入。@pre 满足input_iterator。
             * @return 库语法结果；超限抛InputFailure。@code ctx.parse_string(input); @endcode */
            template <class Iter>
                requires std::input_iterator<Iter>
            bool parse_string(picojson::input<Iter>& input) {
                const bool ok = picojson::default_parse_context::parse_string(input);
                if (ok && out_->get<std::string>().size() > state_.limits_.maxStringBytes_) {
                    throw InputFailure{"RESOURCE_LIMIT", path_, "字符串超过限额", {}};
                }
                return ok;
            }
            /** @brief 递归解析数组元素并共享预算。@tparam Iter 输入迭代器。
             * @param input 输入。@param index 元素序号。@pre 当前输出为数组且Iter满足input_iterator。
             * @return 解析成功标志。@code ctx.parse_array_item(input,0); @endcode */
            template <class Iter>
                requires std::input_iterator<Iter>
            bool parse_array_item(picojson::input<Iter>& input, std::size_t index) {
                auto& values = out_->get<picojson::array>();
                values.emplace_back();
                const auto child = path_ + "/" + std::to_string(index);
                state_.enter(child, depth_ + 1, valueOffset(input));
                StrictContext context(values.back(), state_, child, depth_ + 1);
                return picojson::_parse(context, input);
            }
            /** @brief 解析对象成员并拒绝重复键。@tparam Iter 输入迭代器。
             * @param input 输入。@param key 已解码键。@pre 当前输出为对象且Iter满足input_iterator。
             * @return 解析成功标志；重复键或超限抛InputFailure。@code ctx.parse_object_item(input,"name"); @endcode */
            template <class Iter>
                requires std::input_iterator<Iter>
            bool parse_object_item(picojson::input<Iter>& input, const std::string& key) {
                if (key.size() > state_.limits_.maxStringBytes_) {
                    throw InputFailure{"RESOURCE_LIMIT", path_, "键名超过限额", {}};
                }
                auto& object = out_->get<picojson::object>();
                const auto child = path_ + "/" + escapeKey(key);
                if (object.contains(key)) {
                    throw InputFailure{"DUPLICATE_KEY", child, "对象含重复键",
                                       static_cast<std::size_t>(input.cur() - state_.text_.begin())};
                }
                state_.enter(child, depth_ + 1, valueOffset(input));
                StrictContext context(object[key], state_, child, depth_ + 1);
                return picojson::_parse(context, input);
            }

        private:
            /** @brief 在不改变逻辑流位置的情况下找到值的首字节。
             * @tparam Iter 输入迭代器。@param input picojson输入。@pre input_iterator且由当前上下文独占借用。
             * @return 原文字节偏移；兼容库内部预读。@code auto offset=valueOffset(input); @endcode */
            template <class Iter>
                requires std::input_iterator<Iter>
            std::size_t valueOffset(picojson::input<Iter>& input) {
                input.skip_ws();
                const auto ch = input.getc();
                const auto offset = static_cast<std::size_t>(input.cur() - state_.text_.begin()) - (ch != -1 ? 1 : 0);
                input.ungetc();
                return offset;
            }
            ParseState& state_; ///< 共享限额和位置。
            std::string path_; ///< 当前逻辑位置。
            std::size_t depth_; ///< 当前嵌套深度。
        };
        /** @brief 类型化转换器，所有对象严格限定字段。
         * @code Decoder decoder; auto definition=decoder.decode(root); @endcode */
        class Decoder {
        public:
            /** @brief 解析协议根对象。@param root 已解析JSON。@pre 无。
             * @return 类型化定义；失败抛InputFailure。@code auto c=decode(root); @endcode */
            CaseDefinition decode(const Json& root);

        private:
            /** @brief 检查对象类型与未知键。@param v JSON节点。@param p 路径。
             * @param allowed 合法字段集合。@pre 无。@note 无返回值，非法抛InputFailure。
             * @code object(v,p,{"kind"}); @endcode */
            void object(const Json& v, const std::string& p, std::initializer_list<std::string_view> allowed) {
                if (!v.is<picojson::object>()) {
                    fail("INVALID_TYPE", p, "必须为对象");
                }
                for (const auto& [key, value] : v.get<picojson::object>()) {
                    if (std::find(allowed.begin(), allowed.end(), key) == allowed.end()) {
                        fail("UNKNOWN_FIELD", p + "/" + escapeKey(key), "未知字段");
                    }
                }
            }
            /** @brief 抛出待IO边界转换的输入错误。@param code 错误码。@param p 路径。
             * @param message 消息。@pre 无。@note 无返回值，始终抛InputFailure。@code fail("INVALID","/","非法");
             * @endcode */
            [[noreturn]] void fail(std::string code, const std::string& p, std::string message) {
                throw InputFailure{std::move(code), p, std::move(message), {}};
            }
            /** @brief 查询对象键。@param v 节点。@param key 键。@pre v为对象。
             * @return 是否存在。@code auto found=has(v,"kind"); @endcode */
            bool has(const Json& v, const std::string& key) { return v.get<picojson::object>().contains(key); }
            /** @brief 读取必需成员。@param v 对象。@param key 键。@param p 路径。@pre v为对象。
             * @return 借用成员；缺失抛InputFailure。@code auto& kind=required(v,"kind",p); @endcode */
            const Json& required(const Json& v, const std::string& key, const std::string& p) {
                if (!has(v, key)) {
                    fail("MISSING_FIELD", p + "/" + key, "缺少必填字段");
                }
                return v.get(key);
            }
            /** @brief 严格读取字符串。@param v 节点。@param p 路径。@pre 无。
             * @return 字符串副本；错类型抛InputFailure。@code auto s=text(v,p); @endcode */
            std::string text(const Json& v, const std::string& p) {
                if (!v.is<std::string>()) {
                    fail("INVALID_TYPE", p, "必须为字符串");
                }
                return v.get<std::string>();
            }
            /** @brief 严格读取布尔。@param v 节点。@param p 路径。@pre 无。
             * @return 布尔值；不接受整数。@code auto b=boolean(v,p); @endcode */
            bool boolean(const Json& v, const std::string& p) {
                if (!v.is<bool>()) {
                    fail("INVALID_TYPE", p, "必须为布尔值");
                }
                return v.get<bool>();
            }
            /** @brief 读取物理标量数值。@param v 节点。@param p 路径。@pre 无。
             * @return double；整数物理量允许转换，身份字段不使用此接口。@code auto x=number(v,p); @endcode */
            double number(const Json& v, const std::string& p) {
                if (v.is<std::int64_t>()) {
                    return static_cast<double>(v.get<std::int64_t>());
                }
                if (!v.is<double>()) {
                    fail("INVALID_TYPE", p, "必须为数值");
                }
                return v.get<double>();
            }
            /** @brief 精确读取整数字面量。@param v 节点。@param p 路径。@pre 无。
             * @return int64；浮点或错类型抛InputFailure。@code auto tag=integer(v,p); @endcode */
            std::int64_t integer(const Json& v, const std::string& p) {
                if (!v.is<std::int64_t>()) {
                    fail("INVALID_TYPE", p, "必须为未经过浮点转换的整数");
                }
                return v.get<std::int64_t>();
            }
            /** @brief 读取可表示的正计数。@param v 节点。@param p 路径。@pre 无。
             * @return size_t；非正/越界抛InputFailure。@code auto n=positive(v,p); @endcode */
            std::size_t positive(const Json& v, const std::string& p) {
                auto value = integer(v, p);
                if (value <= 0 || static_cast<std::uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
                    fail("INVALID_VALUE", p, "必须为可表示的正整数");
                }
                return static_cast<std::size_t>(value);
            }
            /** @brief 严格借用数组。@param v 节点。@param p 路径。@pre 无。
             * @return 借用数组，生命周期受v限制；错类型抛InputFailure。@code auto& a=array(v,p); @endcode */
            const picojson::array& array(const Json& v, const std::string& p) {
                if (!v.is<picojson::array>()) {
                    fail("INVALID_TYPE", p, "必须为数组");
                }
                return v.get<picojson::array>();
            }
            /** @brief 检查封闭类型名。@param v 节点。@param p 路径。@param expected 合法值。
             * @pre 无。@note 无返回值，不匹配抛InputFailure。@code literal(v,p,"constant"); @endcode */
            void literal(const Json& v, const std::string& p, std::string_view expected) {
                if (text(v, p) != expected) {
                    fail("UNSUPPORTED_CAPABILITY", p, "不支持的类型或选项，预期" + std::string(expected));
                }
            }
            /** @brief 解码常量场。@param v 节点。@param p 路径。@pre 无。
             * @return 独立场值；非法抛InputFailure。@code auto f=field(v,p); @endcode */
            ScalarField field(const Json& v, const std::string& p) {
                object(v, p, {"kind", "value"});
                literal(required(v, "kind", p), p + "/kind", "constant");
                return ConstantScalar{number(required(v, "value", p), p + "/value")};
            }
            /** @brief 递归解码类型化选择器。@param v 节点。@param p 路径。
             * @pre JSON递归已经受限，随后仍需静态检查选择器深度。
             * @return 拥有子树的选择器；非法抛InputFailure。@code auto r=region(v,p); @endcode */
            RegionSelector region(const Json& v, const std::string& p) {
                if (!v.is<picojson::object>()) {
                    fail("INVALID_TYPE", p, "选择器必须为对象");
                }
                const auto kind = text(required(v, "kind", p), p + "/kind");
                RegionSelector r;
                if (kind == "physicalGroup") {
                    object(v, p, {"kind", "dimension", "tag", "name"});
                    const auto dim = integer(required(v, "dimension", p), p + "/dimension");
                    if (dim < 0 || dim > 3) {
                        fail("INVALID_SELECTOR", p + "/dimension", "维数必须为0至3");
                    }
                    r.dimension_ = static_cast<int>(dim);
                    if (has(v, "tag")) {
                        r.tag_ = integer(v.get("tag"), p + "/tag");
                    }
                    if (has(v, "name")) {
                        r.name_ = text(v.get("name"), p + "/name");
                    }
                }
                else if (kind == "union" || kind == "intersection") {
                    object(v, p, {"kind", "items"});
                    r.kind_ = kind == "union" ? RegionKind::Union : RegionKind::Intersection;
                    const auto& items = array(required(v, "items", p), p + "/items");
                    for (std::size_t i = 0; i < items.size(); ++i) {
                        r.operands_.push_back(region(items[i], p + "/items/" + std::to_string(i)));
                    }
                }
                else if (kind == "difference") {
                    object(v, p, {"kind", "base", "remove"});
                    r.kind_ = RegionKind::Difference;
                    r.operands_.push_back(region(required(v, "base", p), p + "/base"));
                    r.operands_.push_back(region(required(v, "remove", p), p + "/remove"));
                }
                else {
                    fail("UNSUPPORTED_CAPABILITY", p + "/kind", "不支持的选择器类型");
                }
                return r;
            }
        };
        CaseDefinition Decoder::decode(const Json& root) {
            object(root, "",
                   {"schemaVersion", "name", "description", "mesh", "analysis", "materials", "materialAssignments",
                    "sources", "boundaries", "boundaryPolicy", "initialState", "time", "discretization", "linearSolver",
                    "output"});
            CaseDefinition c;
            if (integer(required(root, "schemaVersion", ""), "/schemaVersion") != 1) {
                fail("UNSUPPORTED_VERSION", "/schemaVersion", "只支持协议版本1");
            }
            if (has(root, "name")) {
                c.name_ = text(root.get("name"), "/name");
            }
            if (has(root, "description")) {
                c.description_ = text(root.get("description"), "/description");
            }
            const auto& mesh = required(root, "mesh", "");
            object(mesh, "/mesh", {"path", "format", "coordinateScale"});
            c.mesh_.path_ = text(required(mesh, "path", "/mesh"), "/mesh/path");
            if (c.mesh_.path_.empty()) {
                fail("INVALID_VALUE", "/mesh/path", "文件路径不能为空");
            }
            if (has(mesh, "format")) {
                literal(mesh.get("format"), "/mesh/format", "msh");
            }
            if (has(mesh, "coordinateScale")) {
                c.mesh_.coordinateScale_ = number(mesh.get("coordinateScale"), "/mesh/coordinateScale");
            }
            const auto& a = required(root, "analysis", "");
            object(a, "/analysis", {"type", "space", "domain", "thickness", "geometryTolerance"});
            const auto type = text(required(a, "type", "/analysis"), "/analysis/type");
            if (type == "steady") {
                c.analysis_.type_ = AnalysisType::Steady;
            }
            else if (type == "transient") {
                c.analysis_.type_ = AnalysisType::Transient;
            }
            else {
                fail("UNSUPPORTED_CAPABILITY", "/analysis/type", "不支持的分析类型");
            }
            const auto space = text(required(a, "space", "/analysis"), "/analysis/space");
            if (space == "planar2d") {
                c.analysis_.space_ = SpaceModel::Planar2d;
            }
            else if (space == "solid3d") {
                c.analysis_.space_ = SpaceModel::Solid3d;
            }
            else {
                fail("UNSUPPORTED_CAPABILITY", "/analysis/space", "不支持的空间模型");
            }
            c.analysis_.domain_ = region(required(a, "domain", "/analysis"), "/analysis/domain");
            if (has(a, "thickness")) {
                c.analysis_.thickness_ = number(a.get("thickness"), "/analysis/thickness");
            }
            if (has(a, "geometryTolerance")) {
                c.analysis_.geometryTolerance_ = number(a.get("geometryTolerance"), "/analysis/geometryTolerance");
            }
            const auto& materials = array(required(root, "materials", ""), "/materials");
            for (std::size_t i = 0; i < materials.size(); ++i) {
                const auto& m = materials[i];
                const auto p = "/materials/" + std::to_string(i);
                object(m, p,
                       {"id", "displayName", "conductivity", "density", "specificHeat", "volumetricHeatCapacity"});
                Material value;
                value.id_ = text(required(m, "id", p), p + "/id");
                if (has(m, "displayName")) {
                    value.displayName_ = text(m.get("displayName"), p + "/displayName");
                }
                value.conductivity_ = field(required(m, "conductivity", p), p + "/conductivity");
                if (has(m, "density")) {
                    value.density_ = field(m.get("density"), p + "/density");
                }
                if (has(m, "specificHeat")) {
                    value.specificHeat_ = field(m.get("specificHeat"), p + "/specificHeat");
                }
                if (has(m, "volumetricHeatCapacity")) {
                    value.volumetricHeatCapacity_ =
                        field(m.get("volumetricHeatCapacity"), p + "/volumetricHeatCapacity");
                }
                c.materials_.push_back(std::move(value));
            }
            const auto& assignments = array(required(root, "materialAssignments", ""), "/materialAssignments");
            for (std::size_t i = 0; i < assignments.size(); ++i) {
                const auto& v = assignments[i];
                const auto p = "/materialAssignments/" + std::to_string(i);
                object(v, p, {"id", "region", "material"});
                c.materialAssignments_.push_back({text(required(v, "id", p), p + "/id"),
                                                  region(required(v, "region", p), p + "/region"),
                                                  text(required(v, "material", p), p + "/material")});
            }
            if (has(root, "sources")) {
                const auto& values = array(root.get("sources"), "/sources");
                for (std::size_t i = 0; i < values.size(); ++i) {
                    const auto& v = values[i];
                    const auto p = "/sources/" + std::to_string(i);
                    object(v, p, {"id", "region", "kind", "value"});
                    literal(required(v, "kind", p), p + "/kind", "volumetric");
                    c.sources_.push_back({text(required(v, "id", p), p + "/id"),
                                          region(required(v, "region", p), p + "/region"),
                                          field(required(v, "value", p), p + "/value")});
                }
            }
            if (has(root, "boundaries")) {
                const auto& values = array(root.get("boundaries"), "/boundaries");
                for (std::size_t i = 0; i < values.size(); ++i) {
                    const auto& v = values[i];
                    const auto p = "/boundaries/" + std::to_string(i);
                    if (!v.is<picojson::object>()) {
                        fail("INVALID_TYPE", p, "边界必须为对象");
                    }
                    BoundaryCondition b;
                    b.id_ = text(required(v, "id", p), p + "/id");
                    b.region_ = region(required(v, "region", p), p + "/region");
                    const auto kind = text(required(v, "kind", p), p + "/kind");
                    if (kind == "temperature") {
                        object(v, p, {"id", "region", "kind", "temperature"});
                        b.value_ = TemperatureBoundary{field(required(v, "temperature", p), p + "/temperature")};
                    }
                    else if (kind == "outwardFlux") {
                        object(v, p, {"id", "region", "kind", "flux"});
                        b.value_ = FluxBoundary{field(required(v, "flux", p), p + "/flux")};
                    }
                    else if (kind == "convection") {
                        object(v, p, {"id", "region", "kind", "coefficient", "ambientTemperature"});
                        b.value_ =
                            ConvectionBoundary{field(required(v, "coefficient", p), p + "/coefficient"),
                                               field(required(v, "ambientTemperature", p), p + "/ambientTemperature")};
                    }
                    else {
                        fail("UNSUPPORTED_CAPABILITY", p + "/kind", "不支持的边界类型");
                    }
                    c.boundaries_.push_back(std::move(b));
                }
            }
            if (has(root, "boundaryPolicy")) {
                const auto& v = root.get("boundaryPolicy");
                object(v, "/boundaryPolicy", {"defaultBoundary", "naturalOverlap"});
                if (has(v, "defaultBoundary")) {
                    literal(v.get("defaultBoundary"), "/boundaryPolicy/defaultBoundary", "adiabatic");
                }
                if (has(v, "naturalOverlap")) {
                    const auto s = text(v.get("naturalOverlap"), "/boundaryPolicy/naturalOverlap");
                    if (s == "add") {
                        c.naturalOverlap_ = NaturalOverlap::Add;
                    }
                    else if (s != "error") {
                        fail("INVALID_VALUE", "/boundaryPolicy/naturalOverlap", "必须为add或error");
                    }
                }
            }
            if (has(root, "initialState")) {
                const auto& v = root.get("initialState");
                const std::string p = "/initialState";
                object(v, p, {"defaultTemperature", "overrides"});
                InitialState state;
                state.defaultTemperature_ = field(required(v, "defaultTemperature", p), p + "/defaultTemperature");
                if (has(v, "overrides")) {
                    const auto& items = array(v.get("overrides"), p + "/overrides");
                    for (std::size_t i = 0; i < items.size(); ++i) {
                        const auto& o = items[i];
                        const auto q = p + "/overrides/" + std::to_string(i);
                        object(o, q, {"id", "region", "temperature"});
                        state.overrides_.push_back({text(required(o, "id", q), q + "/id"),
                                                    region(required(o, "region", q), q + "/region"),
                                                    field(required(o, "temperature", q), q + "/temperature")});
                    }
                }
                c.initialState_ = std::move(state);
            }
            if (has(root, "time")) {
                const auto& v = root.get("time");
                const std::string p = "/time";
                object(v, p, {"start", "end", "step", "maxSteps", "method"});
                TimeOptions t;
                t.start_ = number(required(v, "start", p), p + "/start");
                t.end_ = number(required(v, "end", p), p + "/end");
                t.step_ = number(required(v, "step", p), p + "/step");
                if (has(v, "maxSteps")) {
                    t.maxSteps_ = positive(v.get("maxSteps"), p + "/maxSteps");
                }
                if (has(v, "method")) {
                    literal(v.get("method"), p + "/method", "backwardEuler");
                }
                c.time_ = t;
            }
            if (has(root, "discretization")) {
                const auto& v = root.get("discretization");
                const std::string p = "/discretization";
                object(v, p, {"family", "order", "quadrature"});
                if (has(v, "family")) {
                    literal(v.get("family"), p + "/family", "lagrangeContinuous");
                }
                if (has(v, "quadrature")) {
                    literal(v.get("quadrature"), p + "/quadrature", "auto");
                }
                if (has(v, "order")) {
                    const auto n = positive(v.get("order"), p + "/order");
                    if (n > std::numeric_limits<unsigned>::max()) {
                        fail("INTEGER_RANGE", p + "/order", "阶次超过unsigned范围");
                    }
                    c.discretization_.order_ = static_cast<unsigned>(n);
                }
            }
            if (has(root, "linearSolver")) {
                const auto& v = root.get("linearSolver");
                const std::string p = "/linearSolver";
                object(v, p,
                       {"method", "preconditioner", "relativeTolerance", "absoluteTolerance", "maxIterations",
                        "initialGuess"});
                if (has(v, "method")) {
                    const auto s = text(v.get("method"), p + "/method");
                    if (s == "cg") {
                        c.linearSolver_.method_ = LinearMethod::Cg;
                    }
                    else if (s != "bicgstab") {
                        fail("UNSUPPORTED_CAPABILITY", p + "/method", "不支持的线性方法");
                    }
                }
                if (has(v, "preconditioner")) {
                    const auto& pc = v.get("preconditioner");
                    if (!pc.is<picojson::object>()) {
                        fail("INVALID_TYPE", p + "/preconditioner", "必须为对象");
                    }
                    literal(required(pc, "kind", p + "/preconditioner"), p + "/preconditioner/kind", "identity");
                    object(pc, p + "/preconditioner", {"kind"});
                }
                if (has(v, "relativeTolerance")) {
                    c.linearSolver_.relativeTolerance_ = number(v.get("relativeTolerance"), p + "/relativeTolerance");
                }
                if (has(v, "absoluteTolerance")) {
                    c.linearSolver_.absoluteTolerance_ = number(v.get("absoluteTolerance"), p + "/absoluteTolerance");
                }
                if (has(v, "maxIterations")) {
                    c.linearSolver_.maxIterations_ = positive(v.get("maxIterations"), p + "/maxIterations");
                }
                if (has(v, "initialGuess")) {
                    const auto s = text(v.get("initialGuess"), p + "/initialGuess");
                    if (s == "previousStep") {
                        c.linearSolver_.initialGuess_ = InitialGuess::PreviousStep;
                    }
                    else if (s != "zero") {
                        fail("INVALID_VALUE", p + "/initialGuess", "未知初猜策略");
                    }
                }
            }
            if (has(root, "output")) {
                const auto& v = root.get("output");
                const std::string p = "/output";
                object(v, p,
                       {"directory", "format", "fields", "overwrite", "schedule", "includeInitial", "includeFinal"});
                if (has(v, "directory")) {
                    c.output_.directory_ = text(v.get("directory"), p + "/directory");
                }
                if (has(v, "format")) {
                    literal(v.get("format"), p + "/format", "vtu");
                }
                if (has(v, "fields")) {
                    const auto& fields = array(v.get("fields"), p + "/fields");
                    if (fields.size() != 1) {
                        fail("UNSUPPORTED_CAPABILITY", p + "/fields", "首期仅支持单个temperature字段");
                    }
                    literal(fields[0], p + "/fields/0", "temperature");
                }
                if (has(v, "overwrite")) {
                    c.output_.overwrite_ = boolean(v.get("overwrite"), p + "/overwrite");
                }
                if (has(v, "includeInitial")) {
                    c.output_.includeInitial_ = boolean(v.get("includeInitial"), p + "/includeInitial");
                }
                if (has(v, "includeFinal")) {
                    c.output_.includeFinal_ = boolean(v.get("includeFinal"), p + "/includeFinal");
                }
                if (has(v, "schedule")) {
                    const auto& s = v.get("schedule");
                    object(s, p + "/schedule", {"kind", "interval"});
                    literal(required(s, "kind", p + "/schedule"), p + "/schedule/kind", "everySteps");
                    c.output_.everySteps_ =
                        positive(required(s, "interval", p + "/schedule"), p + "/schedule/interval");
                }
            }
            return c;
        }
    } // namespace
    CaseReadResult readCase(std::istream& input, const std::filesystem::path& baseDirectory, std::string source,
                            problem::CaseLimits limits) {
        problem::validateLimits(limits);
        if (!baseDirectory.is_absolute()) {
            throw std::invalid_argument("配置基目录必须为绝对路径");
        }
        CaseReadResult result{{}, problem::Diagnostics(limits.maxDiagnostics_), {}};
        std::string text;
        try {
            char buffer[4096];
            while (input) {
                input.read(buffer, sizeof(buffer));
                const auto count = static_cast<std::size_t>(input.gcount());
                if (count > limits.maxBytes_ - text.size()) {
                    throw InputFailure{"RESOURCE_LIMIT", "", "配置字节数超过限额", {}};
                }
                text.append(buffer, count);
            }
            if (input.bad() || (input.fail() && !input.eof())) {
                throw InputFailure{"IO_ERROR", "", "读取配置流失败", {}};
            }
            lexicalCheck(text);
            Json root;
            ParseState state{limits, text, source, result.locations_};
            state.enter("", 1, 0);
            StrictContext context(root, state, "", 1);
            picojson::input<std::string::const_iterator> parsed(text.cbegin(), text.cend());
            if (!picojson::_parse(context, parsed)) {
                throw InputFailure{"JSON_SYNTAX", "", "JSON语法错误",
                                   static_cast<std::size_t>(parsed.cur() - text.cbegin())};
            }
            parsed.skip_ws();
            if (parsed.getc() != -1) {
                throw InputFailure{"JSON_SYNTAX", "", "根值之后存在多余内容",
                                   static_cast<std::size_t>(parsed.cur() - text.cbegin()) - 1};
            }
            Decoder decoder;
            auto definition = decoder.decode(root);
            if (definition.analysis_.type_ == problem::AnalysisType::Transient) {
                if (!definition.output_.everySteps_) {
                    definition.output_.everySteps_ = 1;
                }
                if (!definition.output_.includeInitial_) {
                    definition.output_.includeInitial_ = true;
                }
            }
            definition.mesh_.path_ = (baseDirectory / definition.mesh_.path_).lexically_normal();
            definition.output_.directory_ = (baseDirectory / definition.output_.directory_).lexically_normal();
            auto report = problem::validateCase(definition, limits);
            for (auto diagnostic : report.entries()) {
                auto p = diagnostic.path_;
                auto it = result.locations_.find(p);
                while (it == result.locations_.end() && !p.empty()) {
                    p = p.substr(0, p.find_last_of('/'));
                    it = result.locations_.find(p);
                }
                if (it != result.locations_.end()) {
                    diagnostic.source_ = source;
                    diagnostic.line_ = it->second.line_;
                    diagnostic.column_ = it->second.column_;
                }
                result.diagnostics_.add(std::move(diagnostic));
            }
            // 聚合集合可能截断；合并原始状态以免截断错误被当成成功。
            if (report.hasErrors()) {
                if (!result.diagnostics_.hasErrors()) {
                    result.diagnostics_.append(report);
                }
            }
            else {
                result.definition_ = std::move(definition);
            }
        }
        catch (const InputFailure& failure) {
            auto p = failure.path_;
            auto it = result.locations_.find(p);
            while (it == result.locations_.end() && !p.empty()) {
                p = p.substr(0, p.find_last_of('/'));
                it = result.locations_.find(p);
            }
            auto pos = failure.offset_ ? location(text, *failure.offset_, source)
                                       : (it != result.locations_.end() ? it->second : SourceLocation{source, 1, 1});
            result.diagnostics_.add({problem::Severity::Error,
                                     failure.code_,
                                     failure.path_,
                                     {},
                                     {},
                                     failure.message_,
                                     "按算例配置协议修正输入",
                                     source,
                                     pos.line_,
                                     pos.column_});
        }
        return result;
    }
    CaseReadResult loadCase(const std::filesystem::path& path, problem::CaseLimits limits) {
        problem::validateLimits(limits);
        std::error_code ec;
        const auto absolute = std::filesystem::absolute(path, ec);
        std::ifstream input(absolute, std::ios::binary);
        if (ec || !input) {
            CaseReadResult result{{}, problem::Diagnostics(limits.maxDiagnostics_), {}};
            result.diagnostics_.add({problem::Severity::Error,
                                     "IO_ERROR",
                                     "/",
                                     {},
                                     {},
                                     "无法打开算例配置",
                                     "检查路径及读取权限",
                                     path.string(),
                                     {},
                                     {}});
            return result;
        }
        return readCase(input, absolute.parent_path(), absolute.string(), limits);
    }
} // namespace thermofem::io
