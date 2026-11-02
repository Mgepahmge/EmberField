/** @file Msh.cu
 * @brief 实现有界MSH解析与ASCII/二进制编码，保持标签和网格索引分离。
 */
#include <IO/Mesh/Msh.cuh>
#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace thermofem::io {
    namespace {
        using namespace mesh;
        /** @brief 报告带格式上下文的错误。@param message 具体原因。@pre 无。
         * @note 不返回，抛runtime_error。@code fail("invalid header"); @endcode */
        [[noreturn]] void fail(const std::string& message) { throw std::runtime_error("MSH: " + message); }

        /** @brief 有界读取器；仅在格式声明的数值区读取二进制。
         * @code Reader reader(bytes,limits); @endcode */
        class Reader {
        public:
            std::istringstream stream_; ///< 已受总字节限额约束的源数据。
            MshLimits limits_; ///< 记录和字节上限。
            bool binary_{}; ///< 当前文件数值编码。
            bool swap_{}; ///< 源字节序是否与本机相反。
            int width_{8}; ///< size_t编码宽度，支持4或8。
            std::string version_; ///< 仅支持2.2或4.1。
            /** @brief 创建读取器。@param bytes 输入副本。@param limits 限额。@pre 已检查大小。
             * @note 无返回值。@code Reader r(bytes,limits); @endcode */
            Reader(const std::string& bytes, MshLimits limits) :
                stream_(bytes, std::ios::in | std::ios::binary), limits_(limits) {}
            /** @brief 读取并去除CR的完整文本行，无参数。@pre 流有效。
             * @return 文本；截断抛异常。@code auto line=r.line(); @endcode */
            std::string line() {
                std::string s;
                if (!std::getline(stream_, s)) {
                    fail("truncated text record");
                }
                if (!s.empty() && s.back() == '\r') {
                    s.pop_back();
                }
                return s;
            }
            /** @brief 跳过段间空白并读取标记。@pre 段外调用。@return 段标记或EOF空串。
             * @code auto marker=r.marker(); @endcode */
            std::string marker() {
                std::string s;
                while (std::getline(stream_, s)) {
                    if (!s.empty() && s.back() == '\r') {
                        s.pop_back();
                    }
                    if (s.find_first_not_of(" \t") != std::string::npos) {
                        return s;
                    }
                }
                return {};
            }
            /** @brief 消费数值区后的唯一换行。@pre 数值区刚结束。@note 无返回值。
             * @code r.newline(); @endcode */
            void newline() {
                if (!binary_) {
                    auto s = line();
                    if (s.find_first_not_of(" \t") != std::string::npos) {
                        fail("extra numeric fields");
                    }
                }
                else {
                    char c{};
                    if (!stream_.get(c)) {
                        fail("missing binary newline");
                    }
                    if (c == '\r' && !stream_.get(c)) {
                        fail("missing LF");
                    }
                    if (c != '\n') {
                        fail("invalid binary boundary");
                    }
                }
            }
            /** @brief 读取可选换字节序的原始数值。@tparam T 平凡算术类型。@pre 容量足够。
             * @return 读取值。@code auto i=r.raw<int>(); @endcode */
            template <class T>
                requires std::is_arithmetic_v<T>
            T raw() {
                std::array<char, sizeof(T)> bytes{};
                if (!stream_.read(bytes.data(), bytes.size())) {
                    fail("truncated binary payload");
                }
                if (swap_) {
                    std::reverse(bytes.begin(), bytes.end());
                }
                T value;
                std::memcpy(&value, bytes.data(), sizeof(T));
                return value;
            }
            /** @brief 读取文本token或二进制数值。@tparam T 算术类型。@pre 编码正确。
             * @return 数值；非法和非有限浮点被拒绝。@code auto n=r.number<int>(); @endcode */
            template <class T>
                requires std::is_arithmetic_v<T>
            T number() {
                T value{};
                if (binary_) {
                    value = raw<T>();
                }
                else {
                    std::string token;
                    if (!(stream_ >> token)) {
                        fail("truncated numeric token");
                    }
                    if constexpr (std::is_integral_v<T>) {
                        auto result = std::from_chars(token.data(), token.data() + token.size(), value);
                        if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
                            fail("invalid integer");
                        }
                    }
                    else {
                        std::istringstream parse(token);
                        parse.imbue(std::locale::classic());
                        char extra;
                        if (!(parse >> value) || parse >> extra) {
                            fail("invalid floating point");
                        }
                    }
                }
                if constexpr (std::is_floating_point_v<T>) {
                    if (!std::isfinite(value)) {
                        fail("nonfinite value");
                    }
                }
                return value;
            }
            /** @brief 读取文件size_t宽度的无符号数。@pre header已读。
             * @return 64位值。@code auto tag=r.size(); @endcode */
            Tag size() {
                if (!binary_) {
                    return number<Tag>();
                }
                return width_ == 8 ? raw<std::uint64_t>() : raw<std::uint32_t>();
            }
            /** @brief 限制一个记录数量。@param n 待检查数量。@pre 无。
             * @return 宿主size_t。@code auto count=r.count(r.size()); @endcode */
            Index count(Tag n) {
                if (n > limits_.maxRecords_ || n > std::numeric_limits<Index>::max()) {
                    fail("record limit exceeded");
                }
                return static_cast<Index>(n);
            }
            /** @brief 读取size_t列表长度。@pre header已读。@return 有界长度。
             * @code auto n=r.count(); @endcode */
            Index count() { return count(size()); }
            /** @brief 检查段末标记。@param name 段名。@pre 仍位于数值段末。
             * @note 无返回值。@code r.end("Nodes"); @endcode */
            void end(const char* name) {
                newline();
                if (marker() != std::string("$End") + name) {
                    fail(std::string("missing end of ") + name);
                }
            }
        };

        /** @brief 追加一维可变长读取值。@tparam T 算术类型。@param r 读取器。
         * @param n 数量。@pre n受限额约束。@return 读取数组。@code auto v=values<int>(r,n); @endcode */
        template <class T>
            requires std::is_arithmetic_v<T>
        std::vector<T> values(Reader& r, Index n) {
            std::vector<T> out;
            out.reserve(r.count(n));
            for (Index i = 0; i < n; ++i) {
                out.push_back(r.number<T>());
            }
            return out;
        }

        /** @brief 读取原始或分区实体。@param r 读取器。@param d 构建数据。@param partitioned 分区段。
         * @pre 在段开始之后。@note 无返回值。@code entities(r,d,false); @endcode */
        void entities(Reader& r, MeshData& d, bool partitioned) {
            if (partitioned) {
                const auto partitions = r.count();
                if (d.partitionCount_ && d.partitionCount_ != partitions) {
                    fail("conflicting partition count");
                }
                d.partitionCount_ = partitions;
                auto n = r.count();
                for (Index i = 0; i < n; ++i) {
                    auto tag = r.number<int>();
                    auto p = r.number<int>();
                    d.ghostEntities_.emplace_back(tag, p);
                }
            }
            std::array<Index, 4> counts;
            for (auto& n : counts) {
                n = r.count();
            }
            auto& out = partitioned ? d.partitionedEntities_ : d.entities_;
            std::map<std::pair<int, std::int64_t>, Index> indices;
            for (Index i = 0; i < out.size(); ++i) {
                indices.emplace(std::pair{out[i].key_.dimension_, out[i].key_.tag_}, i);
            }
            for (int dimension = 0; dimension < 4; ++dimension) {
                for (Index i = 0; i < counts[dimension]; ++i) {
                    Entity e;
                    e.key_ = {dimension, r.number<int>()};
                    if (partitioned) {
                        int dim = r.number<int>(), tag = r.number<int>();
                        if (dim >= 0 && tag != 0) {
                            e.parent_ = EntityKey{dim, tag};
                        }
                        e.partitions_ = values<int>(r, r.count());
                    }
                    for (int j = 0; j < (dimension ? 6 : 3); ++j) {
                        e.bounds_[j] = r.number<double>();
                    }
                    for (Index j = 0, n = r.count(); j < n; ++j) {
                        e.physicalTags_.push_back(r.number<int>());
                    }
                    if (dimension) {
                        for (Index j = 0, n = r.count(); j < n; ++j) {
                            auto tag = r.number<int>();
                            if (tag == std::numeric_limits<int>::min()) {
                                fail("invalid oriented boundary");
                            }
                            e.boundary_.push_back(
                                {{dimension - 1, std::abs(static_cast<std::int64_t>(tag))}, tag < 0 ? -1 : 1});
                        }
                    }
                    const auto [found, inserted] =
                        indices.emplace(std::pair{e.key_.dimension_, e.key_.tag_}, out.size());
                    if (inserted) {
                        out.push_back(std::move(e));
                    }
                    else if (out[found->second] != e) {
                        fail("conflicting repeated entity");
                    }
                }
            }
            r.end(partitioned ? "PartitionedEntities" : "Entities");
        }
        /** @brief 读取完整物理组命名段。@param r 读取器。@param d 数据。
         * @pre 此段始终ASCII。@note 无返回值。@code physical(r,d); @endcode */
        void physical(Reader& r, MeshData& d) {
            std::istringstream count(r.line());
            Tag n;
            char extra;
            if (!(count >> n) || count >> extra) {
                fail("invalid physical names count");
            }
            r.count(n);
            for (Tag i = 0; i < n; ++i) {
                std::istringstream record(r.line());
                PhysicalGroup group;
                if (!(record >> group.key_.dimension_ >> group.key_.tag_)) {
                    fail("invalid physical name");
                }
                record >> std::ws;
                if (record.peek() != '"' || !(record >> std::quoted(group.name_)) || record >> extra) {
                    fail("invalid physical name");
                }
                const auto found = std::find_if(d.physicalGroups_.begin(), d.physicalGroups_.end(),
                                                [&](const PhysicalGroup& old) { return old.key_ == group.key_; });
                if (found == d.physicalGroups_.end()) {
                    d.physicalGroups_.push_back(std::move(group));
                }
                else if (*found != group) {
                    fail("conflicting repeated physical group");
                }
            }
            if (r.line() != "$EndPhysicalNames") {
                fail("missing physical names end");
            }
        }
        /** @brief 检查头部声明标签极值。@param actual 已读标签。@param minimum 声明最小。
         * @param maximum 声明最大。@pre 无。@note 无返回值。@code range(tags,1,3); @endcode */
        void range(const std::vector<Tag>& actual, Tag minimum, Tag maximum) {
            if (actual.empty()) {
                if (minimum || maximum) {
                    fail("empty section has nonzero tag range");
                }
            }
            else {
                auto [lo, hi] = std::minmax_element(actual.begin(), actual.end());
                if (*lo != minimum || *hi != maximum) {
                    fail("incorrect tag range");
                }
            }
        }
        /** @brief 读取节点段并保存标签，尚不解析单元引用。@param r 读取器。@param d 数据。
         * @pre 在Nodes标记之后。@note 无返回值。@code nodes(r,d); @endcode */
        void nodes(Reader& r, MeshData& d) {
            if (r.version_ == "2.2") {
                // 旧版段的计数为ASCII，二进制节点记录使用int标签。
                std::istringstream header(r.line());
                Tag n;
                if (!(header >> n)) {
                    fail("invalid legacy node count");
                }
                r.count(n);
                for (Tag i = 0; i < n; ++i) {
                    Node node;
                    if (r.binary_) {
                        int tag = r.raw<int>();
                        if (tag <= 0) {
                            fail("invalid legacy node tag");
                        }
                        node.tag_ = tag;
                    }
                    else {
                        node.tag_ = r.number<Tag>();
                    }
                    for (auto& x : node.position_) {
                        x = r.number<double>();
                    }
                    d.nodes_.push_back(std::move(node));
                }
                if (n) {
                    r.newline();
                }
                else if (r.binary_) {
                    r.newline();
                }
                if (r.line() != "$EndNodes") {
                    fail("missing legacy node end");
                }
                return;
            }
            auto blocks = r.count(), total = r.count();
            auto minimum = r.size(), maximum = r.size();
            std::vector<Tag> tags;
            const auto firstSection = d.nodes_.size();
            r.count(firstSection + total);
            for (Index b = 0; b < blocks; ++b) {
                EntityKey entity{r.number<int>(), r.number<int>()};
                int parametric = r.number<int>();
                auto n = r.count();
                if (entity.dimension_ < 0 || entity.dimension_ > 3 || (parametric != 0 && parametric != 1) ||
                    n > total - tags.size()) {
                    fail("invalid node block");
                }
                Index first = d.nodes_.size();
                for (Index j = 0; j < n; ++j) {
                    Node node;
                    node.tag_ = r.size();
                    node.classification_ = entity;
                    tags.push_back(node.tag_);
                    d.nodes_.push_back(std::move(node));
                }
                for (Index j = 0; j < n; ++j) {
                    auto& node = d.nodes_[first + j];
                    for (auto& x : node.position_) {
                        x = r.number<double>();
                    }
                    if (parametric) {
                        node.parameters_ = values<double>(r, entity.dimension_);
                    }
                }
            }
            if (d.nodes_.size() - firstSection != total) {
                fail("node total mismatch");
            }
            range(tags, minimum, maximum);
            r.end("Nodes");
        }
        /** @brief 根据2.2标签建立仅存在于旧格式的分类实体。@param d 构建数据。
         * @param block 当前单元块。@param tags 旧标签。@pre 类型已解析。
         * @note 无返回值。@code legacyEntity(d,b,tags); @endcode */
        void legacyEntity(MeshData& d, ElementBlock& block, const std::vector<int>& tags) {
            const int entity = tags.size() > 1 ? tags[1] : 0;
            block.classification_ = EntityKey{block.type_.dimension_, entity};
            block.physicalTags_ =
                !tags.empty() && tags[0] > 0 ? std::vector<std::int64_t>{tags[0]} : std::vector<std::int64_t>{};
            auto found = std::find_if(d.entities_.begin(), d.entities_.end(),
                                      [&](const Entity& e) { return e.key_ == *block.classification_; });
            if (found == d.entities_.end()) {
                Entity e;
                e.key_ = *block.classification_;
                d.entities_.push_back(std::move(e));
                found = std::prev(d.entities_.end());
            }
            if (!tags.empty() && tags[0] > 0 &&
                std::find(found->physicalTags_.begin(), found->physicalTags_.end(), tags[0]) ==
                    found->physicalTags_.end()) {
                found->physicalTags_.push_back(tags[0]);
            }
        }
        /** @brief 单元段读取，连接暂时保存标签，末尾统一解析。@param r 读取器。
         * @param d 数据。@param document 文件元数据。@pre 节点可在之后出现。
         * @note 无返回值。@code elements(r,d,doc); @endcode */
        void elements(Reader& r, MeshData& d, MshDocument& document) {
            if (r.version_ == "2.2") {
                std::istringstream header(r.line());
                Tag total;
                if (!(header >> total)) {
                    fail("invalid legacy element count");
                }
                r.count(total);
                Tag count = 0;
                while (count < total) {
                    int code = 0, numTags = 0;
                    Tag n = 1, tag = 0;
                    if (r.binary_) {
                        code = r.raw<int>();
                        int blockSize = r.raw<int>();
                        numTags = r.raw<int>();
                        if (blockSize <= 0 || numTags < 0) {
                            fail("invalid legacy element block");
                        }
                        n = blockSize;
                    }
                    else {
                        tag = r.number<Tag>();
                        code = r.number<int>();
                        numTags = r.number<int>();
                        if (numTags < 0) {
                            fail("negative legacy tag count");
                        }
                    }
                    if (n > total - count) {
                        fail("legacy element total exceeded");
                    }
                    r.count(numTags);
                    for (Tag j = 0; j < n; ++j) {
                        if (r.binary_) {
                            int value = r.raw<int>();
                            if (value <= 0) {
                                fail("invalid legacy element tag");
                            }
                            tag = value;
                        }
                        auto tags = values<int>(r, numTags);
                        if (tags.size() > 2) {
                            const int n = tags[2];
                            if (n < 0 || static_cast<Index>(n) > tags.size() - 3) {
                                fail("invalid legacy partition tag count");
                            }
                            if (n) {
                                ElementPartition membership;
                                membership.elementTag_ = tag;
                                for (int k = 0; k < n; ++k) {
                                    const auto p = static_cast<std::int64_t>(tags[3 + k]);
                                    if (p == 0 || p == std::numeric_limits<int>::min()) {
                                        fail("invalid legacy partition identifier");
                                    }
                                    const int partition = static_cast<int>(std::abs(p));
                                    d.partitionCount_ = std::max(d.partitionCount_, static_cast<Index>(partition));
                                    (p > 0 ? membership.partitions_ : membership.ghostPartitions_).push_back(partition);
                                }
                                d.elementPartitions_.push_back(std::move(membership));
                            }
                        }
                        ElementBlock block;
                        block.type_ = mshElementType(code);
                        block.tags_.push_back(tag);
                        legacyEntity(d, block, tags);
                        for (Index k = 0; k < block.type_.nodeCount_; ++k) {
                            auto value = r.binary_ ? static_cast<Tag>(r.raw<int>()) : r.number<Tag>();
                            block.nodes_.push_back(value);
                        }
                        block.offsets_.push_back(block.nodes_.size());
                        d.elements_.push_back(std::move(block));
                        document.legacyElementTags_.emplace(tag, std::move(tags));
                    }
                    count += n;
                }
                if (total || r.binary_) {
                    r.newline();
                }
                if (r.line() != "$EndElements") {
                    fail("missing legacy element end");
                }
                return;
            }
            auto blocks = r.count(), total = r.count();
            auto minimum = r.size(), maximum = r.size();
            std::vector<Tag> tags;
            for (Index b = 0; b < blocks; ++b) {
                ElementBlock block;
                block.classification_ = EntityKey{r.number<int>(), r.number<int>()};
                block.type_ = mshElementType(r.number<int>());
                auto n = r.count();
                if (n > total - tags.size() ||
                    (block.type_.nodeCount_ && n > r.limits_.maxRecords_ / block.type_.nodeCount_)) {
                    fail("element connectivity limit");
                }
                for (Index j = 0; j < n; ++j) {
                    Tag tag = r.size();
                    tags.push_back(tag);
                    block.tags_.push_back(tag);
                    for (Index k = 0; k < block.type_.nodeCount_; ++k) {
                        block.nodes_.push_back(r.size());
                    }
                    block.offsets_.push_back(block.nodes_.size());
                }
                d.elements_.push_back(std::move(block));
            }
            if (tags.size() != total) {
                fail("element total mismatch");
            }
            range(tags, minimum, maximum);
            r.end("Elements");
        }
        /** @brief 读取周期关系，保留节点标签直到统一解析。@param r 读取器。@param d 数据。
         * @pre 节点段可尚未读取。@note 无返回值。@code periodic(r,d); @endcode */
        void periodic(Reader& r, MeshData& d) {
            if (r.version_ == "2.2") {
                const bool binary = r.binary_;
                r.binary_ = false;
                auto n = r.count();
                for (Index i = 0; i < n; ++i) {
                    PeriodicLink link;
                    int dim = r.number<int>();
                    link.slave_ = {dim, r.number<int>()};
                    link.master_ = {dim, r.number<int>()};
                    std::string token;
                    if (!(r.stream_ >> token)) {
                        fail("truncated legacy periodic link");
                    }
                    Tag count{};
                    if (token == "Affine") {
                        link.affine_ = values<double>(r, 16);
                        count = r.size();
                    }
                    else {
                        auto parsed = std::from_chars(token.data(), token.data() + token.size(), count);
                        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size()) {
                            fail("invalid legacy periodic count");
                        }
                    }
                    r.count(count);
                    for (Tag j = 0; j < count; ++j) {
                        auto a = r.size(), b = r.size();
                        link.nodes_.emplace_back(a, b);
                    }
                    d.periodic_.push_back(std::move(link));
                }
                r.end("Periodic");
                r.binary_ = binary;
                return;
            }
            auto n = r.count();
            for (Index i = 0; i < n; ++i) {
                PeriodicLink link;
                int dim = r.number<int>();
                link.slave_ = {dim, r.number<int>()};
                link.master_ = {dim, r.number<int>()};
                link.affine_ = values<double>(r, r.count());
                auto count = r.count();
                for (Index j = 0; j < count; ++j) {
                    auto a = r.size();
                    auto b = r.size();
                    link.nodes_.emplace_back(a, b);
                }
                d.periodic_.push_back(std::move(link));
            }
            r.end("Periodic");
        }
        /** @brief 读取幽灵单元。@param r 读取器。@param d 数据。@pre 4.1编码。
         * @note 无返回值。@code ghosts(r,d); @endcode */
        void ghosts(Reader& r, MeshData& d) {
            auto n = r.count();
            for (Index i = 0; i < n; ++i) {
                GhostElement g;
                g.elementTag_ = r.size();
                g.owner_ = r.number<int>();
                g.replicas_ = values<int>(r, r.count());
                d.ghostElements_.push_back(std::move(g));
            }
            r.end("GhostElements");
        }
        /** @brief 读取实体参数化采样。@param r 读取器。@param d 数据。@pre 4.1编码。
         * @note 无返回值。@code parametrizations(r,d); @endcode */
        void parametrizations(Reader& r, MeshData& d) {
            auto curves = r.count(), surfaces = r.count();
            for (int dim = 1; dim <= 2; ++dim) {
                for (Index i = 0; i < (dim == 1 ? curves : surfaces); ++i) {
                    Parametrization p;
                    p.entity_ = {dim, r.number<int>()};
                    auto n = r.count(), triangles = dim == 2 ? r.count() : 0;
                    Index stride = dim == 1 ? 4 : 11;
                    if (n > r.limits_.maxRecords_ / stride) {
                        fail("parametrization limit");
                    }
                    p.samples_ = values<double>(r, n * stride);
                    for (Index j = 0; j < triangles; ++j) {
                        std::array<int, 3> t;
                        for (int& index : t) {
                            index = r.number<int>();
                        }
                        p.triangles_.push_back(t);
                    }
                    d.parametrizations_.push_back(std::move(p));
                }
            }
            r.end("Parametrizations");
        }
    } // namespace

    MshDocument readMsh(std::istream& input, MshLimits limits) {
        std::string bytes;
        std::array<char, 8192> chunk;
        while (input.read(chunk.data(), chunk.size()) || input.gcount()) {
            auto n = static_cast<Index>(input.gcount());
            if (n > limits.maxBytes_ - std::min(limits.maxBytes_, bytes.size())) {
                fail("input byte limit");
            }
            bytes.append(chunk.data(), n);
        }
        if (input.bad()) {
            fail("input stream failure");
        }
        Reader r(bytes, limits);
        if (r.line() != "$MeshFormat") {
            fail("missing MeshFormat");
        }
        std::istringstream header(r.line());
        int binary;
        char extra;
        if (!(header >> r.version_ >> binary >> r.width_) || header >> extra ||
            (r.version_ != "4.1" && r.version_ != "2.2") || (binary != 0 && binary != 1) ||
            (r.width_ != 4 && r.width_ != 8)) {
            fail("unsupported header");
        }
        if (r.version_ == "2.2" && r.width_ != 8) {
            fail("legacy double width must be 8");
        }
        r.binary_ = binary;
        if (r.binary_) {
            auto sentinel = r.raw<std::uint32_t>();
            if (sentinel == 0x01000000) {
                r.swap_ = true;
            }
            else if (sentinel != 1) {
                fail("invalid endian sentinel");
            }
            r.newline();
        }
        if (r.line() != "$EndMeshFormat") {
            fail("missing MeshFormat end");
        }
        MshDocument document;
        document.sourceVersion_ = r.version_;
        MeshData data;
        std::set<std::string> seen;
        for (auto marker = r.marker(); !marker.empty(); marker = r.marker()) {
            if (marker.size() < 2 || marker[0] != '$' || marker.starts_with("$End")) {
                fail("invalid section marker");
            }
            auto name = marker.substr(1);
            if (name == "NodeData" || name == "ElementData" || name == "ElementNodeData" ||
                name == "InterpolationScheme") {
                fail("result fields are not part of a mesh document");
            }
            bool known = name == "PhysicalNames" || name == "Entities" || name == "PartitionedEntities" ||
                name == "Nodes" || name == "Elements" || name == "Periodic" || name == "GhostElements" ||
                name == "Parametrizations";
            if (name == "MeshFormat") {
                fail("repeated MeshFormat");
            }
            if (known) {
                seen.insert(name);
            }
            if (name == "PhysicalNames") {
                physical(r, data);
            }
            else if (name == "Entities") {
                entities(r, data, false);
            }
            else if (name == "PartitionedEntities") {
                entities(r, data, true);
            }
            else if (name == "Nodes") {
                nodes(r, data);
            }
            else if (name == "Elements") {
                elements(r, data, document);
            }
            else if (name == "Periodic") {
                periodic(r, data);
            }
            else if (name == "GhostElements") {
                ghosts(r, data);
            }
            else if (name == "Parametrizations") {
                parametrizations(r, data);
            }
            else {
                if (r.binary_) {
                    fail("unknown binary section: " + name);
                }
                MshSection section{name, {}};
                auto end = "$End" + name;
                while (true) {
                    auto line = r.line();
                    if (line == end) {
                        break;
                    }
                    section.text_ += line + '\n';
                }
                document.extensions_.push_back(std::move(section));
            }
        }
        if (!seen.contains("Nodes") || !seen.contains("Elements")) {
            fail("missing required mesh section");
        }
        std::unordered_map<Tag, Index> indices;
        for (Index i = 0; i < data.nodes_.size(); ++i) {
            if (!indices.emplace(data.nodes_[i].tag_, i).second) {
                fail("duplicate node tag");
            }
        }
        for (auto& block : data.elements_) {
            for (auto& tag : block.nodes_) {
                auto found = indices.find(tag);
                if (found == indices.end()) {
                    fail("dangling element node tag");
                }
                tag = found->second;
            }
        }
        for (auto& link : data.periodic_) {
            for (auto& pair : link.nodes_) {
                if (!indices.contains(pair.first) || !indices.contains(pair.second)) {
                    fail("dangling periodic node tag");
                }
                pair = {indices.at(pair.first), indices.at(pair.second)};
            }
        }
        try {
            document.mesh_ = Mesh(std::move(data));
        }
        catch (const std::invalid_argument& error) {
            fail(error.what());
        }
        return document;
    }

    mesh::ElementType mshElementType(int code) {
        using F = mesh::CellFamily;
        struct Entry {
            int code;
            F family;
            int dimension;
            int nodes;
            int order;
            const char* layout;
        };
        static constexpr Entry entries[] = {
            {1, F::Segment, 1, 2, 1, "complete"},          {2, F::Triangle, 2, 3, 1, "complete"},
            {3, F::Quadrilateral, 2, 4, 1, "complete"},    {4, F::Tetrahedron, 3, 4, 1, "complete"},
            {5, F::Hexahedron, 3, 8, 1, "complete"},       {6, F::Prism, 3, 6, 1, "complete"},
            {7, F::Pyramid, 3, 5, 1, "complete"},          {8, F::Segment, 1, 3, 2, "complete"},
            {9, F::Triangle, 2, 6, 2, "complete"},         {10, F::Quadrilateral, 2, 9, 2, "complete"},
            {11, F::Tetrahedron, 3, 10, 2, "complete"},    {12, F::Hexahedron, 3, 27, 2, "complete"},
            {13, F::Prism, 3, 18, 2, "complete"},          {14, F::Pyramid, 3, 14, 2, "complete"},
            {15, F::Point, 0, 1, 0, "complete"},           {16, F::Quadrilateral, 2, 8, 2, "edge-only"},
            {17, F::Hexahedron, 3, 20, 2, "edge-only"},    {18, F::Prism, 3, 15, 2, "edge-only"},
            {19, F::Pyramid, 3, 13, 2, "edge-only"},       {20, F::Triangle, 2, 9, 3, "edge-only"},
            {21, F::Triangle, 2, 10, 3, "complete"},       {22, F::Triangle, 2, 12, 4, "edge-only"},
            {23, F::Triangle, 2, 15, 4, "complete"},       {24, F::Triangle, 2, 15, 5, "edge-only"},
            {25, F::Triangle, 2, 21, 5, "complete"},       {26, F::Segment, 1, 4, 3, "complete"},
            {27, F::Segment, 1, 5, 4, "complete"},         {28, F::Segment, 1, 6, 5, "complete"},
            {29, F::Tetrahedron, 3, 20, 3, "complete"},    {30, F::Tetrahedron, 3, 35, 4, "complete"},
            {31, F::Tetrahedron, 3, 56, 5, "complete"},    {32, F::Tetrahedron, 3, 22, 4, "edge-only"},
            {33, F::Tetrahedron, 3, 28, 5, "edge-only"},   {36, F::Quadrilateral, 2, 16, 3, "complete"},
            {37, F::Quadrilateral, 2, 25, 4, "complete"},  {38, F::Quadrilateral, 2, 36, 5, "complete"},
            {39, F::Quadrilateral, 2, 12, 3, "edge-only"}, {40, F::Quadrilateral, 2, 16, 4, "edge-only"},
            {41, F::Quadrilateral, 2, 20, 5, "edge-only"}, {67, F::Segment, 1, 2, 1, "border"},
            {68, F::Triangle, 2, 3, 1, "border"},          {70, F::Segment, 1, 2, 1, "child"},
            {84, F::Segment, 1, 1, 0, "complete"},         {85, F::Triangle, 2, 1, 0, "complete"},
            {86, F::Quadrilateral, 2, 1, 0, "complete"},   {87, F::Tetrahedron, 3, 1, 0, "complete"},
            {88, F::Hexahedron, 3, 1, 0, "complete"},      {89, F::Prism, 3, 1, 0, "complete"},
            {90, F::Prism, 3, 40, 3, "complete"},          {91, F::Prism, 3, 75, 4, "complete"},
            {132, F::Pyramid, 3, 1, 0, "complete"},        {133, F::Point, 0, 1, 0, "subdivision"},
            {134, F::Segment, 1, 2, 1, "subdivision"},     {135, F::Triangle, 2, 3, 1, "subdivision"},
            {136, F::Tetrahedron, 3, 4, 1, "subdivision"}, {137, F::Tetrahedron, 3, 16, 3, "edge-only"},
            {138, F::Triangle, 2, 4, 1, "mini"},           {139, F::Tetrahedron, 3, 5, 1, "mini"},
            {140, F::Trihedron, 3, 4, 1, "complete"}};
        for (const auto& e : entries) {
            if (e.code == code) {
                return {e.family, e.dimension, static_cast<std::size_t>(e.nodes), e.order, e.layout};
            }
        }
        if (code >= 42 && code <= 46) {
            int p = code - 36;
            return {F::Triangle, 2, static_cast<Index>((p + 1) * (p + 2) / 2), p, "complete"};
        }
        if (code >= 47 && code <= 51) {
            int p = code - 41;
            return {F::Quadrilateral, 2, static_cast<Index>((p + 1) * (p + 1)), p, "complete"};
        }
        if (code >= 52 && code <= 56) {
            int p = code - 46;
            return {F::Triangle, 2, static_cast<Index>(3 * p), p, "edge-only"};
        }
        if (code >= 57 && code <= 61) {
            int p = code - 51;
            return {F::Quadrilateral, 2, static_cast<Index>(4 * p), p, "edge-only"};
        }
        if (code >= 62 && code <= 66) {
            int p = code - 56;
            return {F::Segment, 1, static_cast<Index>(p + 1), p, "complete"};
        }
        if (code >= 71 && code <= 75) {
            int p = code - 65;
            return {F::Tetrahedron, 3, static_cast<Index>((p + 1) * (p + 2) * (p + 3) / 6), p, "complete"};
        }
        if (code >= 79 && code <= 83) {
            int p = code - 73;
            return {F::Tetrahedron, 3, static_cast<Index>(4 + 6 * (p - 1)), p, "edge-only"};
        }
        if (code >= 92 && code <= 98) {
            int p = code - 89;
            return {F::Hexahedron, 3, static_cast<Index>((p + 1) * (p + 1) * (p + 1)), p, "complete"};
        }
        if (code >= 99 && code <= 105) {
            int p = code - 96;
            return {F::Hexahedron, 3, static_cast<Index>(8 + 12 * (p - 1)), p, "edge-only"};
        }
        if (code >= 106 && code <= 110) {
            int p = code - 101;
            return {F::Prism, 3, static_cast<Index>((p + 1) * (p + 1) * (p + 2) / 2), p, "complete"};
        }
        if (code >= 111 && code <= 117) {
            int p = code - 108;
            return {F::Prism, 3, static_cast<Index>(6 + 9 * (p - 1)), p, "edge-only"};
        }
        if (code >= 118 && code <= 124) {
            int p = code - 115;
            return {F::Pyramid, 3, static_cast<Index>((p + 1) * (p + 2) * (2 * p + 3) / 6), p, "complete"};
        }
        if (code >= 125 && code <= 131) {
            int p = code - 122;
            return {F::Pyramid, 3, static_cast<Index>(5 + 8 * (p - 1)), p, "edge-only"};
        }
        fail("unsupported or variable-layout element type " + std::to_string(code));
    }

    namespace {
        /** @brief 将中立参考单元映射为文件编号。@param type 描述符。@pre 已登记布局。
         * @return MSH编号；无法表示抛异常。@code auto code=elementCode(type); @endcode */
        int elementCode(const ElementType& type) {
            for (int code = 1; code <= 140; ++code) {
                if (code == 34 || code == 35 || code == 69 || (code >= 76 && code <= 78)) {
                    continue;
                }
                if (mshElementType(code) == type) {
                    return code;
                }
            }
            fail("reference layout has no MSH4.1 encoding");
        }
        /** @brief 检查写出所需的32位实体标签。@param value 中立整数。@pre 无。
         * @return 32位整数；无法表示抛异常。@code auto tag=integer(key.tag_); @endcode */
        int integer(std::int64_t value) {
            if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
                fail("entity metadata exceeds int32");
            }
            return static_cast<int>(value);
        }

        /** @brief 数值编码器，文本仅用max_digits10，二进制使用明确64位size_t。
         * @code Writer writer(output,encoding); @endcode */
        class Writer {
        public:
            std::ostream& stream_; ///< 借用目标流。
            bool binary_; ///< 数值编码。
            /** @brief 创建写出器。@param stream 流。@param encoding 编码。@pre 流有效。
             * @note 无返回值。@code Writer w(out,MshEncoding::Ascii); @endcode */
            Writer(std::ostream& stream, MshEncoding encoding) :
                stream_(stream), binary_(encoding == MshEncoding::Binary) {}
            /** @brief 写一个算术值。@tparam T 算术类型。@param value 数值。@pre 类型与格式一致。
             * @note 无返回值。@code w.number(3); @endcode */
            template <class T>
                requires std::is_arithmetic_v<T>
            void number(T value) {
                if (binary_) {
                    stream_.write(reinterpret_cast<const char*>(&value), sizeof(T));
                }
                else {
                    stream_ << value << ' ';
                }
            }
            /** @brief 写64位无符号字段。@param value 数量或标签。@pre 可表示。
             * @note 无返回值。@code w.size(3); @endcode */
            void size(Tag value) { number<std::uint64_t>(value); }
            /** @brief 结束ASCII记录；二进制不插入字节。无参数、无前提、无返回值。
             * @code w.line(); @endcode */
            void line() {
                if (!binary_) {
                    stream_ << '\n';
                }
            }
            /** @brief 写段开始。@param name 段名。@pre 段外。@note 无返回值。
             * @code w.begin("Nodes"); @endcode */
            void begin(const char* name) { stream_ << '$' << name << '\n'; }
            /** @brief 写段结束。@param name 段名。@pre 已完成数值payload。@note 无返回值。
             * @code w.end("Nodes"); @endcode */
            void end(const char* name) { stream_ << '\n' << "$End" << name << '\n'; }
        };
        /** @brief 编码前检验可表示性，避免打开文件后才发现结构降级。
         * @param document 文档。@pre Mesh已验证。@note 无返回值，失败抛异常。
         * @code writable(document); @endcode */
        void writable(const MshDocument& document) {
            const auto& d = document.mesh_.data();
            if (!d.elementPartitions_.empty()) {
                fail("per-cell partition memberships require explicit MSH4.1 partition-entity conversion");
            }
            if (!d.cellBoundaries_.empty()) {
                fail("explicit cell boundary graph has no independent MSH4.1 section");
            }
            for (const auto& node : d.nodes_) {
                if (!node.classification_) {
                    fail("MSH4.1 requires explicit node classification");
                }
                integer(node.classification_->tag_);
            }
            for (const auto& block : d.elements_) {
                if (!block.classification_) {
                    fail("MSH4.1 requires explicit element classification");
                }
                integer(block.classification_->tag_);
                (void)elementCode(block.type_);
                if (block.physicalTags_) {
                    auto e = std::find_if(d.entities_.begin(), d.entities_.end(),
                                          [&](const Entity& entity) { return entity.key_ == *block.classification_; });
                    if (e == d.entities_.end() || e->physicalTags_ != *block.physicalTags_) {
                        fail("per-cell groups require explicit entity conversion for MSH4.1 export");
                    }
                }
            }
            for (const auto* es : {&d.entities_, &d.partitionedEntities_}) {
                for (const auto& e : *es) {
                    integer(e.key_.tag_);
                    if (e.key_.dimension_ == 0 && (e.bounds_[3] != 0 || e.bounds_[4] != 0 || e.bounds_[5] != 0)) {
                        fail("point bounds have no trailing MSH fields");
                    }
                    if (es == &d.entities_ && (e.parent_ || !e.partitions_.empty())) {
                        fail("original entity contains partition-only metadata");
                    }
                    for (const auto& b : e.boundary_) {
                        if (b.entity_.dimension_ != e.key_.dimension_ - 1) {
                            fail("MSH boundary dimension mismatch");
                        }
                    }
                    if (!e.embedded_.empty()) {
                        fail("explicit embedded incidence cannot be represented independently in MSH4.1");
                    }
                    if (e.parent_) {
                        integer(e.parent_->tag_);
                    }
                    for (auto p : e.physicalTags_) {
                        integer(p);
                    }
                    for (const auto& b : e.boundary_) {
                        integer(b.entity_.tag_);
                    }
                }
            }
            for (const auto& group : d.physicalGroups_) {
                integer(group.key_.tag_);
                if (group.name_.size() > 127) {
                    fail("physical name exceeds MSH limit");
                }
            }
            for (const auto& link : d.periodic_) {
                integer(link.slave_.tag_);
                integer(link.master_.tag_);
            }
            for (auto [tag, p] : d.ghostEntities_) {
                (void)integer(tag);
            }
            for (const auto& p : d.parametrizations_) {
                integer(p.entity_.tag_);
            }
            // 2.2物理标签属于单元而4.1属于实体，不能将不同成员无声扩大为并集。
            for (const auto& block : d.elements_) {
                for (Tag tag : block.tags_) {
                    if (auto old = document.legacyElementTags_.find(tag); old != document.legacyElementTags_.end()) {
                        const auto& values = old->second;
                        if (values.size() > 2) {
                            fail("legacy element attributes need explicit conversion before MSH4.1 export");
                        }
                        auto entity = std::find_if(d.entities_.begin(), d.entities_.end(), [&](const Entity& e) {
                            return block.classification_ && e.key_ == *block.classification_;
                        });
                        const std::vector<std::int64_t> expected = !values.empty() && values[0] > 0
                            ? std::vector<std::int64_t>{values[0]}
                            : std::vector<std::int64_t>{};
                        if (entity == d.entities_.end() || entity->physicalTags_ != expected) {
                            fail("legacy per-element physical groups cannot be widened on export");
                        }
                    }
                }
            }
            for (const auto& section : document.extensions_) {
                if (section.name_.empty() || section.name_.find_first_of("$\r\n \t") != std::string::npos ||
                    section.text_.find("$End" + section.name_) != std::string::npos) {
                    fail("invalid extension section");
                }
                static const std::set<std::string> reserved = {
                    "MeshFormat",  "PhysicalNames",   "Entities",           "PartitionedEntities", "Nodes",
                    "Elements",    "Periodic",        "GhostElements",      "Parametrizations",    "NodeData",
                    "ElementData", "ElementNodeData", "InterpolationScheme"};
                if (reserved.contains(section.name_)) {
                    fail("extension shadows structural or result section");
                }
            }
        }
        /** @brief 写原始或分区实体。@param w 编码器。@param d 网格。@param partitioned 分区段。
         * @pre 可表示性已验证。@note 无返回值。@code writeEntities(w,d,false); @endcode */
        void writeEntities(Writer& w, const MeshData& d, bool partitioned) {
            auto name = partitioned ? "PartitionedEntities" : "Entities";
            w.begin(name);
            if (partitioned) {
                w.size(d.partitionCount_);
                w.size(d.ghostEntities_.size());
                for (auto [tag, p] : d.ghostEntities_) {
                    w.number(integer(tag));
                    w.number(p);
                }
            }
            const auto& es = partitioned ? d.partitionedEntities_ : d.entities_;
            for (int dim = 0; dim < 4; ++dim) {
                w.size(std::count_if(es.begin(), es.end(), [&](const Entity& e) { return e.key_.dimension_ == dim; }));
            }
            for (int dim = 0; dim < 4; ++dim) {
                for (const auto& e : es) {
                    if (e.key_.dimension_ == dim) {
                        w.number(integer(e.key_.tag_));
                        if (partitioned) {
                            w.number(e.parent_ ? e.parent_->dimension_ : 0);
                            w.number(e.parent_ ? integer(e.parent_->tag_) : 0);
                            w.size(e.partitions_.size());
                            for (int p : e.partitions_) {
                                w.number(p);
                            }
                        }
                        for (int i = 0; i < (dim ? 6 : 3); ++i) {
                            w.number(e.bounds_[i]);
                        }
                        w.size(e.physicalTags_.size());
                        for (auto p : e.physicalTags_) {
                            w.number(integer(p));
                        }
                        if (dim) {
                            w.size(e.boundary_.size());
                            for (const auto& b : e.boundary_) {
                                w.number(integer(b.entity_.tag_) * b.orientation_);
                            }
                        }
                    }
                }
            }
            w.end(name);
        }
    } // namespace

    void writeMsh(std::ostream& output, const MshDocument& document, MshEncoding encoding, MshVersion version) {
        if (encoding != MshEncoding::Ascii && encoding != MshEncoding::Binary) {
            fail("invalid output encoding");
        }
        if (version != MshVersion::V41 && version != MshVersion::V22) {
            fail("invalid output version");
        }
        if (encoding == MshEncoding::Binary && !document.extensions_.empty()) {
            fail("unknown extensions require ASCII output");
        }
        if (version == MshVersion::V22) {
            const auto& d = document.mesh_.data();
            if (!d.partitionedEntities_.empty() || !d.ghostElements_.empty() || !d.ghostEntities_.empty() ||
                !d.parametrizations_.empty() || !d.cellBoundaries_.empty()) {
                fail("MSH2.2 cannot preserve rich mesh metadata");
            }
            for (const auto& n : d.nodes_) {
                if (n.classification_ || !n.parameters_.empty()) {
                    fail("MSH2.2 cannot preserve node classification");
                }
                if (n.tag_ > std::numeric_limits<int>::max()) {
                    fail("MSH2.2 node tag exceeds int32");
                }
            }
            for (const auto& e : d.entities_) {
                if (!e.boundary_.empty() || !e.embedded_.empty() || e.parent_ || !e.partitions_.empty() ||
                    e.bounds_ != std::array<double, 6>{}) {
                    fail("MSH2.2 cannot preserve entity structure");
                }
            }
            std::ostringstream bytes(std::ios::out | std::ios::binary);
            bytes.imbue(std::locale::classic());
            bytes << std::setprecision(std::numeric_limits<double>::max_digits10);
            const bool binary = encoding == MshEncoding::Binary;
            bytes << "$MeshFormat\n2.2 " << (binary ? 1 : 0) << " 8\n";
            Writer w(bytes, encoding);
            if (binary) {
                w.number<int>(1);
                bytes << '\n';
            }
            bytes << "$EndMeshFormat\n";
            if (!d.physicalGroups_.empty()) {
                bytes << "$PhysicalNames\n" << d.physicalGroups_.size() << '\n';
                for (const auto& p : d.physicalGroups_) {
                    bytes << p.key_.dimension_ << ' ' << integer(p.key_.tag_) << ' ' << std::quoted(p.name_) << '\n';
                }
                bytes << "$EndPhysicalNames\n";
            }
            bytes << "$Nodes\n" << d.nodes_.size() << '\n';
            for (const auto& n : d.nodes_) {
                w.number<int>(static_cast<int>(n.tag_));
                for (double x : n.position_) {
                    w.number(x);
                }
                if (!binary) {
                    bytes << '\n';
                }
            }
            if (binary) {
                bytes << '\n';
            }
            bytes << "$EndNodes\n";
            Index count = 0;
            for (const auto& b : d.elements_) {
                count += b.tags_.size();
            }
            bytes << "$Elements\n" << count << '\n';
            for (const auto& b : d.elements_) {
                for (Index i = 0; i < b.tags_.size(); ++i) {
                    Tag tag = b.tags_[i];
                    if (tag > std::numeric_limits<int>::max()) {
                        fail("MSH2.2 element tag exceeds int32");
                    }
                    std::vector<int> tags;
                    if (auto found = document.legacyElementTags_.find(tag);
                        found != document.legacyElementTags_.end()) {
                        tags = found->second;
                    }
                    else {
                        const auto groups = b.physicalTags_.value_or(std::vector<std::int64_t>{});
                        if (groups.size() > 1) {
                            fail("MSH2.2 accepts one physical group per element");
                        }
                        tags = {groups.empty() ? 0 : integer(groups.front()),
                                b.classification_ ? integer(b.classification_->tag_) : 0};
                    }
                    if (tags.size() > static_cast<Index>(std::numeric_limits<int>::max())) {
                        fail("too many legacy tags");
                    }
                    const int code = elementCode(b.type_);
                    if (binary) {
                        w.number(code);
                        w.number<int>(1);
                        w.number<int>(tags.size());
                        w.number<int>(tag);
                    }
                    else {
                        w.number<int>(tag);
                        w.number(code);
                        w.number<int>(tags.size());
                    }
                    for (int value : tags) {
                        w.number(value);
                    }
                    for (Index j = b.offsets_[i]; j < b.offsets_[i + 1]; ++j) {
                        w.number<int>(d.nodes_[b.nodes_[j]].tag_);
                    }
                    if (!binary) {
                        bytes << '\n';
                    }
                }
            }
            if (binary) {
                bytes << '\n';
            }
            bytes << "$EndElements\n";
            if (!d.periodic_.empty()) {
                bytes << "$Periodic\n" << d.periodic_.size() << '\n';
                for (const auto& p : d.periodic_) {
                    bytes << p.slave_.dimension_ << ' ' << integer(p.slave_.tag_) << ' ' << integer(p.master_.tag_)
                          << '\n';
                    if (!p.affine_.empty()) {
                        bytes << "Affine";
                        for (double x : p.affine_) {
                            bytes << ' ' << x;
                        }
                        bytes << '\n';
                    }
                    bytes << p.nodes_.size() << '\n';
                    for (auto [a, b] : p.nodes_) {
                        bytes << d.nodes_[a].tag_ << ' ' << d.nodes_[b].tag_ << '\n';
                    }
                }
                bytes << "$EndPeriodic\n";
            }
            for (const auto& s : document.extensions_) {
                if (s.name_.empty() || s.name_.find_first_of("$\r\n \t") != std::string::npos ||
                    s.text_.find('$') != std::string::npos) {
                    fail("invalid legacy extension");
                }
                bytes << '$' << s.name_ << '\n'
                      << s.text_ << (s.text_.empty() || s.text_.back() == '\n' ? "" : "\n") << "$End" << s.name_
                      << '\n';
            }
            const auto content = bytes.str();
            // 旧格式不能表达分类节点和一般实体图，交叉解析验证拒绝隐式信息丢失。
            std::istringstream verification(content, std::ios::in | std::ios::binary);
            const auto decoded = readMsh(verification);
            if (decoded.mesh_.data() != d ||
                (!document.legacyElementTags_.empty() && decoded.legacyElementTags_ != document.legacyElementTags_)) {
                fail("MSH2.2 export would lose or alter mesh metadata");
            }
            output.write(content.data(), content.size());
            if (!output) {
                fail("output stream failure");
            }
            return;
        }
        writable(document);
        // 编码到独立流，不改变调用者locale、精度或格式标志。
        std::ostringstream storage(std::ios::out | std::ios::binary);
        storage.imbue(std::locale::classic());
        storage << std::setprecision(std::numeric_limits<double>::max_digits10);
        Writer w(storage, encoding);
        const auto& d = document.mesh_.data();
        storage << "$MeshFormat\n4.1 " << (w.binary_ ? 1 : 0) << " 8\n";
        if (w.binary_) {
            w.number<std::int32_t>(1);
            storage << '\n';
        }
        storage << "$EndMeshFormat\n";
        if (!d.physicalGroups_.empty()) {
            w.begin("PhysicalNames");
            storage << d.physicalGroups_.size() << '\n';
            for (const auto& p : d.physicalGroups_) {
                storage << p.key_.dimension_ << ' ' << p.key_.tag_ << ' ' << std::quoted(p.name_) << '\n';
            }
            storage << "$EndPhysicalNames\n";
        }
        if (!d.entities_.empty()) {
            writeEntities(w, d, false);
        }
        if (d.partitionCount_ || !d.partitionedEntities_.empty()) {
            writeEntities(w, d, true);
        }
        // 按相邻同分类节点分块，保持节点数组顺序及连续索引。
        std::vector<std::pair<Index, Index>> blocks;
        for (Index i = 0; i < d.nodes_.size();) {
            Index end = i + 1;
            while (end < d.nodes_.size() && d.nodes_[end].classification_ == d.nodes_[i].classification_ &&
                   d.nodes_[end].parameters_.empty() == d.nodes_[i].parameters_.empty()) {
                ++end;
            }
            blocks.emplace_back(i, end);
            i = end;
        }
        auto nodeRange = std::minmax_element(d.nodes_.begin(), d.nodes_.end(),
                                             [](const Node& a, const Node& b) { return a.tag_ < b.tag_; });
        w.begin("Nodes");
        w.size(blocks.size());
        w.size(d.nodes_.size());
        w.size(d.nodes_.empty() ? 0 : nodeRange.first->tag_);
        w.size(d.nodes_.empty() ? 0 : nodeRange.second->tag_);
        w.line();
        for (auto [first, end] : blocks) {
            const auto& key = *d.nodes_[first].classification_;
            w.number(key.dimension_);
            w.number(integer(key.tag_));
            w.number(d.nodes_[first].parameters_.empty() ? 0 : 1);
            w.size(end - first);
            w.line();
            for (Index i = first; i < end; ++i) {
                w.size(d.nodes_[i].tag_);
                w.line();
            }
            for (Index i = first; i < end; ++i) {
                for (double x : d.nodes_[i].position_) {
                    w.number(x);
                }
                for (double x : d.nodes_[i].parameters_) {
                    w.number(x);
                }
                w.line();
            }
        }
        w.end("Nodes");
        Index count = 0;
        Tag minimum = std::numeric_limits<Tag>::max(), maximum = 0;
        for (const auto& block : d.elements_) {
            for (Tag tag : block.tags_) {
                ++count;
                minimum = std::min(minimum, tag);
                maximum = std::max(maximum, tag);
            }
        }
        w.begin("Elements");
        w.size(d.elements_.size());
        w.size(count);
        w.size(count ? minimum : 0);
        w.size(maximum);
        w.line();
        for (const auto& block : d.elements_) {
            w.number(block.classification_->dimension_);
            w.number(integer(block.classification_->tag_));
            w.number(elementCode(block.type_));
            w.size(block.tags_.size());
            w.line();
            for (Index e = 0; e < block.tags_.size(); ++e) {
                w.size(block.tags_[e]);
                for (Index j = block.offsets_[e]; j < block.offsets_[e + 1]; ++j) {
                    w.size(d.nodes_[block.nodes_[j]].tag_);
                }
                w.line();
            }
        }
        w.end("Elements");
        if (!d.periodic_.empty()) {
            w.begin("Periodic");
            w.size(d.periodic_.size());
            w.line();
            for (const auto& p : d.periodic_) {
                w.number(p.slave_.dimension_);
                w.number(integer(p.slave_.tag_));
                w.number(integer(p.master_.tag_));
                w.line();
                w.size(p.affine_.size());
                for (double x : p.affine_) {
                    w.number(x);
                }
                w.line();
                w.size(p.nodes_.size());
                w.line();
                for (auto [a, b] : p.nodes_) {
                    w.size(d.nodes_[a].tag_);
                    w.size(d.nodes_[b].tag_);
                    w.line();
                }
            }
            w.end("Periodic");
        }
        if (!d.ghostElements_.empty()) {
            w.begin("GhostElements");
            w.size(d.ghostElements_.size());
            for (const auto& g : d.ghostElements_) {
                w.size(g.elementTag_);
                w.number(g.owner_);
                w.size(g.replicas_.size());
                for (int p : g.replicas_) {
                    w.number(p);
                }
            }
            w.end("GhostElements");
        }
        if (!d.parametrizations_.empty()) {
            w.begin("Parametrizations");
            for (int dim = 1; dim <= 2; ++dim) {
                w.size(std::count_if(d.parametrizations_.begin(), d.parametrizations_.end(),
                                     [&](const Parametrization& p) { return p.entity_.dimension_ == dim; }));
            }
            for (int dim = 1; dim <= 2; ++dim) {
                for (const auto& p : d.parametrizations_) {
                    if (p.entity_.dimension_ == dim) {
                        w.number(integer(p.entity_.tag_));
                        w.size(p.samples_.size() / (dim == 1 ? 4 : 11));
                        if (dim == 2) {
                            w.size(p.triangles_.size());
                        }
                        for (double x : p.samples_) {
                            w.number(x);
                        }
                        for (auto t : p.triangles_) {
                            for (int i : t) {
                                w.number(i);
                            }
                        }
                    }
                }
            }
            w.end("Parametrizations");
        }
        for (const auto& section : document.extensions_) {
            storage << '$' << section.name_ << '\n'
                    << section.text_ << (section.text_.empty() || section.text_.back() == '\n' ? "" : "\n") << "$End"
                    << section.name_ << '\n';
        }
        if (!storage) {
            fail("encoding failed");
        }
        const auto bytes = storage.str();
        output.write(bytes.data(), bytes.size());
        if (!output) {
            fail("output stream failure");
        }
    }
    MshDocument loadMsh(const std::filesystem::path& path, MshLimits limits) {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            fail("cannot open input " + path.string());
        }
        return readMsh(input, limits);
    }
    void saveMsh(const std::filesystem::path& path, const MshDocument& document, MshEncoding encoding,
                 MshVersion version) {
        std::ostringstream bytes(std::ios::out | std::ios::binary);
        writeMsh(bytes, document, encoding, version);
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) {
            fail("cannot open output " + path.string());
        }
        const auto content = bytes.str();
        output.write(content.data(), content.size());
        output.close();
        if (!output) {
            fail("file write failure " + path.string());
        }
    }
} // namespace thermofem::io
