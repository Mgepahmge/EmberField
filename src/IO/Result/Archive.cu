/** @file Archive.cu
 * @brief 独占目录、受限文件写入、完整CSV记录与摘要最后提交。
 */
#define PICOJSON_USE_INT64
#include <IO/Case/Write.cuh>
#include <IO/Result/Archive.cuh>
#include <IO/Result/Pvd.cuh>
#include <IO/Result/Vtu.cuh>
#include <Simulation/Case/Region.cuh>
#include <chrono>
#include <cmath>
#include <fstream>
#include <map>
#include <picojson.h>
#include <random>
#include <sstream>
#include <version.cuh>
#include "Encoding.cuh"
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif
namespace emberfield::io {
    namespace {
        namespace fs = std::filesystem;
        using J = picojson::value;
        using O = picojson::object;
        using A = picojson::array;
        /** @brief UTC时间。无参数无前提。@return ISO8601。@code auto t=now(); @endcode */
        std::string now() {
            auto time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            std::tm tm{};
#ifdef _WIN32
            gmtime_s(&tm, &time);
#else
            gmtime_r(&time, &tm);
#endif
            std::ostringstream s;
            s << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
            return s.str();
        }
        /** @brief 非凭据运行标识。无参数无前提。@return ASCII字符串。@code auto id=runId(); @endcode */
        std::string runId() {
            std::random_device random;
            std::ostringstream s;
            s << std::hex << std::chrono::high_resolution_clock::now().time_since_epoch().count() << '-' << random()
              << random();
            return s.str();
        }
        /** @brief 有限数值字符串。@param x 可空数值。@pre 无。@return 空或17位十进制。@code auto s=number(x); @endcode
         */
        std::string number(std::optional<double> x) {
            if (!x || !std::isfinite(*x)) {
                return {};
            }
            std::ostringstream s;
            s.imbue(std::locale::classic());
            s << std::setprecision(17) << *x;
            return s.str();
        }
        /** @brief JSON数值空值适配。@param x 可空值。@pre 无。@return 有限数或null。@code auto j=numeric(x); @endcode
         */
        J numeric(std::optional<double> x) { return x && std::isfinite(*x) ? J(*x) : J(); }
        /** @brief RFC4180字段。@param s 原文。@pre UTF-8。@return 转义字段。@code auto f=csv(id); @endcode */
        std::string csv(const std::string& s) {
            if (s.find_first_of(",\"\r\n") == std::string::npos) {
                return s;
            }
            std::string out = "\"";
            for (char c : s) {
                out += c;
                if (c == '"') {
                    out += c;
                }
            }
            return out + '"';
        }
        /** @brief 一条完整CSV记录。@param fields 字段。@pre 无。@return CRLF结尾。@code auto row=record(f); @endcode */
        std::string record(std::initializer_list<std::string> fields) {
            std::string out;
            bool first = true;
            for (const auto& f : fields) {
                if (!first) {
                    out += ',';
                }
                first = false;
                out += csv(f);
            }
            return out + "\r\n";
        }
        /** @brief 路径前缀按分量比较。@param parent 绝对路径。@param child 绝对路径。@pre 已规范化。
         * @return child位于parent或与其相同。@code auto b=contains(p,c); @endcode */
        bool contains(const fs::path& parent, const fs::path& child) {
            auto a = parent.begin(), b = child.begin();
            for (; a != parent.end() && b != child.end(); ++a, ++b) {
                if (*a != *b) {
                    return false;
                }
            }
            return a == parent.end();
        }
        /** @brief 禁止路径任意现有分量经过符号链接。@param path 绝对路径。@pre 无。
         * @note 无返回值，非法invalid_argument。@code safePath(p); @endcode */
        void safePath(const fs::path& path) {
            fs::path prefix;
            for (const auto& part : path) {
                prefix /= part;
                std::error_code ec;
                const auto st = fs::symlink_status(prefix, ec);
                if (ec && ec != std::errc::no_such_file_or_directory) {
                    throw fs::filesystem_error("检查输出路径", prefix, ec);
                }
                if (fs::is_symlink(st)) {
                    throw std::invalid_argument("输出路径禁止符号链接");
                }
#ifdef _WIN32
                auto flags = GetFileAttributesW(prefix.c_str());
                if (flags != INVALID_FILE_ATTRIBUTES && (flags & FILE_ATTRIBUTE_REPARSE_POINT)) {
                    throw std::invalid_argument("输出路径禁止重解析点");
                }
#endif
            }
        }
        /** @brief 替换已完成文件。@param from 临时文件。@param to 正式文件。@pre 同文件系统。
         * @note 无返回值，系统错误传播。@code replace(a,b); @endcode */
        void replace(const fs::path& from, const fs::path& to) {
#ifdef _WIN32
            if (!MoveFileExW(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING)) {
                throw fs::filesystem_error("替换结果文件", from, to,
                                           std::error_code(GetLastError(), std::system_category()));
            }
#else
            fs::rename(from, to);
#endif
        }
        /** @brief 限额写出缓冲，转发到文件而不缓存整个VTU。@code LimitedBuffer b(file.rdbuf(),limit); @endcode */
        class LimitedBuffer : public std::streambuf {
        public:
            /** @brief 借用目标。@param target 文件缓冲。@param limit 允许字节。@pre target有效。@note 无返回值。@code
             * LimitedBuffer b(t,100); @endcode */
            LimitedBuffer(std::streambuf* target, std::uint64_t limit) : target_(target), remaining_(limit) {}

        protected:
            std::streamsize xsputn(const char* s, std::streamsize n) override {
                if (n < 0 || static_cast<std::uint64_t>(n) > remaining_) {
                    throw std::length_error("输出总字节预算不足");
                }
                auto written = target_->sputn(s, n);
                remaining_ -= static_cast<std::uint64_t>(written);
                return written;
            }
            int_type overflow(int_type c) override {
                if (traits_type::eq_int_type(c, traits_type::eof())) {
                    return traits_type::not_eof(c);
                }
                char value = traits_type::to_char_type(c);
                return xsputn(&value, 1) == 1 ? c : traits_type::eof();
            }
            int sync() override { return target_->pubsync(); }

        private:
            std::streambuf* target_; ///< 借用文件。
            std::uint64_t remaining_; ///< 剩余字节。
        };
        /** @brief 表格已提交前缀。@code Table t; @endcode */
        struct Table {
            std::string path_;
            std::uint64_t rows_{}, bytes_{};
        }; // ///< 相对路径、记录数和字节。
        /** @brief 为JSON树和编码文本保守预估内存，不分配中间树。
         * @param value 借用JSON。@param budget 允许估计值。@pre 有限深度协议树。
         * @return 估计字节，超限length_error。@code auto n=jsonBudget(j,limit); @endcode */
        std::size_t jsonBudget(const J& value, std::size_t budget) {
            std::size_t used = 512;
            if (used > budget) {
                throw std::length_error("摘要内存预算不足");
            }
            const auto add = [&](std::size_t n) {
                if (n > budget - used) {
                    throw std::length_error("摘要内存预算不足");
                }
                used += n;
            };
            if (value.is<std::string>()) {
                const auto n = value.get<std::string>().size();
                if (n > (budget - used) / 16) {
                    throw std::length_error("摘要字符串预算不足");
                }
                add(n * 16);
            }
            else if (value.is<A>()) {
                for (const auto& x : value.get<A>()) {
                    add(jsonBudget(x, budget - used));
                }
            }
            else if (value.is<O>()) {
                for (const auto& [key, x] : value.get<O>()) {
                    if (key.size() > (budget - used) / 16) {
                        throw std::length_error("摘要键预算不足");
                    }
                    add(key.size() * 16);
                    add(jsonBudget(x, budget - used));
                }
            }
            return used;
        }
        /** @brief 不复制根对象的摘要预算检查。@param object 根对象。@param budget 限额。
         * @pre 无。@note 无返回值，超限length_error。@code summaryBudget(o,limit); @endcode */
        void summaryBudget(const O& object, std::size_t budget) {
            std::size_t used = 0;
            for (const auto& [key, value] : object) {
                if (key.size() > (budget - used) / 16) {
                    throw std::length_error("摘要预算不足");
                }
                used += key.size() * 16;
                used += jsonBudget(value, budget - used);
            }
        }
        constexpr auto stepHeader =
            "record,step,time_s,dt_s,kind,status,iterations,solver_residual_l2,solver_relative_residual,accepted_"
            "residual_l2,temperature_min_k,temperature_max_k,stored_energy_j,storage_power_w,source_power_w,prescribed_"
            "flux_out_w,convection_out_w,dirichlet_out_w,balance_error_w,balance_tolerance_w,balance_status,setup_s,"
            "solve_s,postprocess_s,frame_path,error_code\r\n";
        constexpr auto monitorHeader =
            "record,step,time_s,monitor_id,monitor_kind,quantity,value,unit,status,error_code\r\n";
        constexpr auto boundaryHeader =
            "record,step,time_s,boundary_id,boundary_kind,aggregation,power_out_w,status\r\n";
    } // namespace
    struct ResultArchive::Impl {
        fs::path directory_, lock_; ///< 绝对会话与旁路独占目录。
        bool ownsLock_{}, finished_{}, broken_{}; ///< 生命周期状态。
        O summary_; ///< 当前候选摘要；仅写成功才为磁盘权威。
        std::map<std::string, std::uint64_t> sizes_; ///< 本会话正式及临时文件大小。
        std::uint64_t limit_{}, maxFrames_{}, warnings_{}, errors_{}; ///< 输出限额与诊断计数。
        std::size_t hostLimit_{}; ///< 摘要和文本的增量预算。
        Table steps_{"steps.csv"}, monitors_{"monitors.csv"}, boundary_{"boundary.csv"}; ///< 表格提交前缀。
        bool hasMonitors_{}, hasEnergy_{}, transient_{}; ///< 选定输出。
        VtuEncoding encoding_{}; ///< 字段编码。
        std::vector<PvdEntry> frames_; ///< 已提交时间序列。
        std::optional<std::uint64_t> acceptedStep_, publishedStep_; ///< 数值与发布游标，禁止倒序或重复。
        /** @brief 清理本实例持有的空锁，无参数无前提。@note 无返回值，不抛。@code impl.reset(); @endcode */
        ~Impl() noexcept {
            if (ownsLock_) {
                std::error_code ec;
                fs::remove(lock_, ec);
            }
        }
        /** @brief 查询剩余逻辑字节，无参数。@pre 当前计数可信。@return 剩余；溢出length_error。@code auto
         * n=remaining(); @endcode */
        std::uint64_t remaining() const {
            std::uint64_t total = 0;
            for (const auto& [path, n] : sizes_) {
                if (n > limit_ - total) {
                    throw std::length_error("结果预算已耗尽");
                }
                total += n;
            }
            return limit_ - total;
        }
        /** @brief 有界完整文件写出，不修改正式文件。@tparam F 可调用编码器。@param path 相对临时路径。
         * @param writer 编码器。@pre 独占目录。@note 无返回值；任何失败保留文件供诊断。
         * @code file("x.tmp",[](auto& s){s<<"x";}); @endcode */
        template <class F>
            requires std::invocable<F, std::ostream&>
        void file(const std::string& path, F&& writer) {
            const auto allowance = remaining() + sizes_[path];
            std::ofstream f(directory_ / path, std::ios::binary | std::ios::trunc);
            if (!f) {
                throw std::ios_base::failure("无法创建结果文件: " + path);
            }
            LimitedBuffer buffer(f.rdbuf(), allowance);
            std::ostream out(&buffer);
            out.exceptions(std::ios::badbit | std::ios::failbit);
            writer(out);
            out.flush();
            f.close();
            if (!f) {
                throw std::ios_base::failure("关闭结果文件失败");
            }
            sizes_[path] = fs::file_size(directory_ / path);
        }
        /** @brief 安全替换文本文件。@param path 相对正式路径。@param text 文本。@pre 独占目录。
         * @note 无返回值，IO异常传播。@code text("run-summary.json",j); @endcode */
        void text(const std::string& path, const std::string& text) {
            const auto temp = path + ".tmp";
            file(temp, [&](std::ostream& out) { out << text; });
            replace(directory_ / temp, directory_ / path);
            sizes_[path] = sizes_[temp];
            sizes_.erase(temp);
        }
        /** @brief 同步表格清单后最后提交摘要，无参数。@pre 前面文件已完成。
         * @note 无返回值，失败保留旧摘要。@code commit(); @endcode */
        void commit() {
            auto table = [](const Table& t) {
                return J(O{{"path", J(t.path_)},
                           {"committedRows", J(std::to_string(t.rows_))},
                           {"committedBytes", J(std::to_string(t.bytes_))}});
            };
            summary_["tables"] = J(O{{"steps", table(steps_)},
                                     {"monitors", hasMonitors_ ? table(monitors_) : J()},
                                     {"boundary", hasEnergy_ ? table(boundary_) : J()}});
            summaryBudget(summary_, hostLimit_ / 4);
            text("run-summary.json", J(summary_).serialize(true) + "\n");
        }
        /** @brief 完整记录追加并flush。@param table 表。@param rows 已编码文本。@param count 记录数。
         * @pre 尚未提交摘要。@note 无返回值，失败不更新前缀。
         * @code append(steps,row,1); @endcode */
        void append(Table& table, const std::string& rows, std::uint64_t count) {
            if (rows.size() > remaining()) {
                throw std::length_error("CSV输出预算不足");
            }
            std::ofstream out(directory_ / table.path_, std::ios::binary | std::ios::app);
            out.exceptions(std::ios::badbit | std::ios::failbit);
            out << rows;
            out.flush();
            out.close();
            table.rows_ += count;
            table.bytes_ += rows.size();
            sizes_[table.path_] = table.bytes_;
        }
        /** @brief 日志追加。@param level 级别。@param code 代码。@param message 单消息。
         * @param step 可选步。@pre 无。@note 无返回值，错误传播。@code log("INFO","RUN","开始",{}); @endcode */
        void log(const std::string& level, const std::string& code, const std::string& message,
                 std::optional<std::uint64_t> step) {
            std::string clean;
            for (char c : message) {
                if (c == '\r') {
                    clean += "\\r";
                }
                else if (c == '\n') {
                    clean += "\\n";
                }
                else {
                    clean += c;
                }
            }
            const auto line = now() + " " + level + " run " + code + " step=" + (step ? std::to_string(*step) : "-") +
                " " + clean + "\n";
            if (line.size() > remaining()) {
                throw std::length_error("日志输出预算不足");
            }
            std::ofstream out(directory_ / "run.log", std::ios::binary | std::ios::app);
            out.exceptions(std::ios::badbit | std::ios::failbit);
            out << line;
            out.flush();
            out.close();
            sizes_["run.log"] += line.size();
        }
    };
    ResultArchive::ResultArchive(const std::filesystem::path& directory, const std::filesystem::path& input,
                                 const problem::AssembledThermalModel& model,
                                 const problem::ThermalPostprocessPlan& plan) : impl_(std::make_unique<Impl>()) {
        auto& s = *impl_;
        const auto& prepared = *model.plan()->prepared();
        const auto& d = prepared.bound()->data();
        const auto& c = d.definition_;
        const auto& options = c.output_;
        s.directory_ = fs::absolute(directory).lexically_normal();
        if (s.directory_ == s.directory_.root_path()) {
            throw std::invalid_argument("禁止将文件系统根作为结果目录");
        }
        safePath(s.directory_);
        for (const auto& source : {fs::weakly_canonical(input), fs::weakly_canonical(c.mesh_.path_)}) {
            if (contains(fs::weakly_canonical(s.directory_), source)) {
                throw std::invalid_argument("输出目录包含配置或网格来源");
            }
        }
        fs::create_directories(s.directory_.parent_path());
        safePath(s.directory_.parent_path());
        s.lock_ = s.directory_;
        s.lock_ += ".emberfield-lock";
        if (!fs::create_directory(s.lock_)) {
            throw std::runtime_error("结果目录已被锁定: " + s.lock_.string());
        }
        s.ownsLock_ = true;
        const auto id = runId();
        J backup;
        if (fs::exists(s.directory_) && !fs::is_empty(s.directory_)) {
            if (!options.overwrite_) {
                throw std::invalid_argument("结果目录非空，默认禁止覆盖");
            }
            safePath(s.directory_ / "run-summary.json");
            std::ifstream old(s.directory_ / "run-summary.json", std::ios::binary);
            J manifest;
            std::string data;
            if (old) {
                const auto size = fs::file_size(s.directory_ / "run-summary.json");
                if (size > options.limits_.maxHostBytes_) {
                    throw std::length_error("旧清单超出读取预算");
                }
                data.assign(std::istreambuf_iterator<char>(old), {});
            }
            if (!picojson::parse(manifest, data).empty() || !manifest.is<O>() || !manifest.contains("schemaVersion") ||
                !manifest.get("schemaVersion").is<std::int64_t>() ||
                manifest.get("schemaVersion").get<std::int64_t>() != 1 || !manifest.contains("program") ||
                !manifest.get("program").is<O>() || !manifest.get("program").contains("name") ||
                manifest.get("program").get("name") != J("EmberField") || !manifest.contains("runId") ||
                !manifest.get("runId").is<std::string>() || manifest.get("runId").get<std::string>().empty() ||
                !manifest.contains("tables") || !manifest.get("tables").is<O>() || !manifest.contains("frames") ||
                !manifest.get("frames").is<A>()) {
                throw std::invalid_argument("仅允许备份带有效EmberField清单的旧结果");
            }
            auto path = s.directory_;
            path += ".backup-" + id;
            fs::rename(s.directory_, path);
            backup = J(path.filename().generic_string());
        }
        fs::create_directories(s.directory_ / "fields");
        s.limit_ = options.limits_.maxOutputBytes_;
        s.hostLimit_ = options.limits_.maxHostBytes_;
        s.maxFrames_ = options.limits_.maxFrames_;
        s.hasEnergy_ = options.energy_.enabled_;
        s.hasMonitors_ = !options.monitors_.empty();
        s.transient_ = c.time_.has_value();
        s.encoding_ =
            options.encoding_ == problem::OutputEncoding::Ascii ? VtuEncoding::Ascii : VtuEncoding::AppendedRaw;
        J effective;
        if (!picojson::parse(effective, serializeCase(c)).empty()) {
            throw std::logic_error("有效配置序列化失败");
        }
        J monitorDictionary;
        if (!picojson::parse(monitorDictionary, plan.monitorDictionary()).empty()) {
            throw std::logic_error("监测字典序列化失败");
        }
        const auto& definitions = effective.get("output").get("monitors").get<A>();
        for (std::size_t i = 0; i < definitions.size(); ++i) {
            monitorDictionary.get<A>()[i].get<O>()["definition"] = definitions[i];
        }
        A materials, boundaries, groups, fieldDefinitions;
        for (std::size_t i = 0; i < c.materials_.size(); ++i) {
            materials.emplace_back(O{{"index", J(std::to_string(i))},
                                     {"id", J(c.materials_[i].id_)},
                                     {"displayName", J(c.materials_[i].displayName_)}});
        }
        for (std::size_t i = 0; i < c.boundaries_.size(); ++i) {
            boundaries.emplace_back(O{{"index", J(std::to_string(i))},
                                      {"id", J(c.boundaries_[i].id_)},
                                      {"kind",
                                       J(c.boundaries_[i].value_.index() == 0       ? "temperature"
                                             : c.boundaries_[i].value_.index() == 1 ? "flux"
                                                                                    : "convection")}});
        }
        const auto membership =
            problem::buildPhysicalMembership(*d.mesh_, std::max<std::size_t>(1000000, plan.grid()->cellCount() * 16));
        for (const auto& [key, members] : membership.members_) {
            J name;
            for (const auto& g : d.mesh_->data().physicalGroups_) {
                if (g.key_.dimension_ == key.first && g.key_.tag_ == key.second) {
                    name = J(g.name_);
                }
            }
            groups.emplace_back(O{{"dimension", J(static_cast<std::int64_t>(key.first))},
                                  {"tag", J(std::to_string(key.second))},
                                  {"name", name}});
        }
        for (const auto field : options.fields_) {
            const auto f = std::string(problem::outputFieldName(field));
            fieldDefinitions.emplace_back(O{{"name", J(f)},
                                            {"association", J(f == "temperature" ? "point" : "cell")},
                                            {"components", J(static_cast<std::int64_t>(f == "temperature" ? 1 : 3))},
                                            {"unit",
                                             J(f == "temperature"    ? "K"
                                                   : f == "heatFlux" ? "W/m^2"
                                                                     : "K/m")},
                                            {"representation", J(f == "temperature" ? "nodal" : "cellConstant")}});
        }
        std::size_t sourceCells = 0;
        for (const auto& block : d.mesh_->data().elements_) {
            sourceCells += block.tags_.size();
        }
        const auto v = version();
        s.summary_ =
            O{{"schemaVersion", J(std::int64_t{1})},
              {"runId", J(id)},
              {"program",
               J(O{{"name", J("EmberField")},
                   {"version",
                    J(std::to_string(v.major_) + "." + std::to_string(v.minor_) + "." + std::to_string(v.patch_))},
                   {"gitCommit", J()}})},
              {"startedAt", J(now())},
              {"finishedAt", J()},
              {"analysis", J(s.transient_ ? "transient" : "steady")},
              {"status", J("running")},
              {"solveStatus", J("running")},
              {"outputStatus", J("running")},
              {"lastAcceptedStep", J()},
              {"lastPersistedStep", J()},
              {"lastAcceptedTimeS", J()},
              {"completedSteps", J("0")},
              {"input",
               J(O{{"caseFile", J(fs::absolute(input).generic_string())},
                   {"meshFile", J(c.mesh_.path_.generic_string())},
                   {"effectiveCase", effective}})},
              {"mesh",
               J(O{{"dimension", J(static_cast<std::int64_t>(prepared.space()->dimension()))},
                   {"sourceNodeCount", J(std::to_string(d.mesh_->data().nodes_.size()))},
                   {"sourceCellCount", J(std::to_string(sourceCells))},
                   {"pointCount", J(std::to_string(plan.grid()->pointCount()))},
                   {"cellCount", J(std::to_string(plan.grid()->cellCount()))},
                   {"dofCount", J(std::to_string(prepared.space()->dofCount()))},
                   {"nnz", J(std::to_string(model.spatial().nonZeroCount()))},
                   {"coordinateScale", J(c.mesh_.coordinateScale_)},
                   {"thicknessM", numeric(c.analysis_.thickness_)},
                   {"elementFamilies", J(A{J(prepared.space()->dimension() == 2 ? "triangle3" : "tet4")})}})},
              {"dictionaries",
               J(O{{"materials", J(materials)},
                   {"boundaries", J(boundaries)},
                   {"physicalGroups", J(groups)},
                   {"monitors", monitorDictionary}})},
              {"fieldDefinitions", J(fieldDefinitions)},
              {"frames", J(A{})},
              {"tables", J(O{})},
              {"timingS",
               J(O{{"input", J()},
                   {"preparation", J()},
                   {"assembly", J()},
                   {"solverSetup", J()},
                   {"solve", J()},
                   {"postprocess", J()},
                   {"output", J()},
                   {"total", J()}})},
              {"diagnostics",
               J(O{{"warningCount", J("0")}, {"errorCount", J("0")}, {"truncated", J(false)}, {"lastError", J()}})},
              {"backupDirectory", backup}};
        s.text(s.steps_.path_, stepHeader);
        s.steps_.bytes_ = std::string_view(stepHeader).size();
        if (s.hasMonitors_) {
            s.text(s.monitors_.path_, monitorHeader);
            s.monitors_.bytes_ = std::string_view(monitorHeader).size();
        }
        if (s.hasEnergy_) {
            s.text(s.boundary_.path_, boundaryHeader);
            s.boundary_.bytes_ = std::string_view(boundaryHeader).size();
        }
        s.log("INFO", "RUN_STARTED", "求解与结果输出开始", {});
        s.commit();
    }
    ResultArchive::~ResultArchive() noexcept = default;
    void ResultArchive::accepted(std::uint64_t step, std::optional<double> time) {
        auto& s = *impl_;
        if (s.finished_ || s.broken_) {
            throw std::logic_error("输出会话不可接受状态");
        }
        if (s.acceptedStep_ && step <= *s.acceptedStep_) {
            throw std::invalid_argument("数值接受步必须递增");
        }
        if (s.transient_ != time.has_value() || (time && !std::isfinite(*time))) {
            throw std::invalid_argument("数值接受时间不符");
        }
        s.acceptedStep_ = step;
        s.summary_["lastAcceptedStep"] = J(std::to_string(step));
        s.summary_["lastAcceptedTimeS"] = numeric(time);
        s.summary_["completedSteps"] = J(std::to_string(step));
    }
    void ResultArchive::timing(const std::string& stage, double seconds) {
        auto& times = impl_->summary_["timingS"].get<O>();
        if (!times.contains(stage) || !std::isfinite(seconds) || seconds < 0) {
            throw std::invalid_argument("非法运行计时");
        }
        times[stage] = J(seconds);
    }
    void ResultArchive::addDiagnostic(const problem::Diagnostic& diagnostic, const std::string& stage) {
        auto& s = *impl_;
        if (s.finished_ || s.broken_) {
            throw std::logic_error("输出会话不可追加诊断");
        }
        const bool warning = diagnostic.severity_ == problem::Severity::Warning;
        if (warning) {
            ++s.warnings_;
        }
        else {
            ++s.errors_;
        }
        auto& d = s.summary_["diagnostics"].get<O>();
        d[warning ? "warningCount" : "errorCount"] = J(std::to_string(warning ? s.warnings_ : s.errors_));
        if (!warning) {
            d["lastError"] = J(O{{"code", J(diagnostic.code_)},
                                 {"stage", J(stage)},
                                 {"message", J(diagnostic.message_)},
                                 {"step", J()}});
        }
        s.log(warning ? "WARNING" : "ERROR", diagnostic.code_, stage + ": " + diagnostic.message_, {});
    }
    void ResultArchive::publishStep(const StepRecord& r, const problem::ThermalFields* fields, bool saveFrame) {
        auto& s = *impl_;
        if (s.finished_ || s.broken_) {
            throw std::logic_error("输出会话已终结或失败");
        }
        if ((saveFrame && (!fields || !fields->snapshot_)) ||
            (fields && fields->snapshot_ &&
             (fields->snapshot_->step() != r.step_ || fields->snapshot_->time() != r.time_))) {
            throw std::invalid_argument("结果记录与快照不相容");
        }
        if (s.publishedStep_ && r.step_ <= *s.publishedStep_) {
            throw std::invalid_argument("禁止重复或倒序发布状态");
        }
        if (s.transient_ != r.time_.has_value() || (r.time_ && !std::isfinite(*r.time_)) ||
            (r.dt_ && (!std::isfinite(*r.dt_) || *r.dt_ <= 0))) {
            throw std::invalid_argument("发布时刻或步长非法");
        }
        if (fields) {
            if (!s.acceptedStep_ || *s.acceptedStep_ != r.step_) {
                throw std::invalid_argument("物理场未通过数值接受");
            }
            if (r.kind_ == "initial") {
                if (r.step_ != 0 || !s.transient_ || r.solve_ || r.acceptedResidual_ || r.dt_ ||
                    r.status_ != "provided" ||
                    (fields->snapshot_ && fields->snapshot_->kind() != field::SampleKind::Initial)) {
                    throw std::invalid_argument("初态记录非法");
                }
            }
            else if (r.kind_ != "accepted" || r.status_ != "converged" || !r.solve_ ||
                     r.solve_->status_ != SolveStatus::Converged || !r.acceptedResidual_ ||
                     !std::isfinite(*r.acceptedResidual_) || *r.acceptedResidual_ < 0 ||
                     (fields->snapshot_ && fields->snapshot_->kind() != field::SampleKind::Accepted)) {
                throw std::invalid_argument("接受态求解记录非法");
            }
        }
        else if (r.kind_ != "failedAttempt") {
            throw std::invalid_argument("缺少后处理仅允许失败尝试");
        }
        summaryBudget(s.summary_, s.hostLimit_ / 4);
        const auto oldSummary = s.summary_;
        const auto oldSteps = s.steps_, oldMonitors = s.monitors_, oldBoundary = s.boundary_;
        const auto oldFrames = s.frames_;
        try {
            std::string path;
            if (saveFrame) {
                auto& frames = s.summary_["frames"].get<A>();
                if (frames.size() >= s.maxFrames_) {
                    throw std::length_error("输出帧数超限");
                }
                std::ostringstream name;
                name << "fields/state_" << std::setw(10) << std::setfill('0') << r.step_ << ".vtu";
                path = name.str();
                if (fs::exists(s.directory_ / path)) {
                    throw std::logic_error("禁止重复发布结果帧");
                }
                s.file(path + ".tmp", [&](std::ostream& out) { writeVtu(out, *fields->snapshot_, s.encoding_); });
                replace(s.directory_ / (path + ".tmp"), s.directory_ / path);
                s.sizes_[path] = s.sizes_[path + ".tmp"];
                s.sizes_.erase(path + ".tmp");
                frames.emplace_back(O{{"step", J(std::to_string(r.step_))},
                                      {"timeS", numeric(r.time_)},
                                      {"kind", J(r.kind_)},
                                      {"path", J(path)},
                                      {"bytes", J(std::to_string(s.sizes_[path]))}});
                if (s.transient_) {
                    s.frames_.push_back({*r.time_, path});
                    std::ostringstream pvd;
                    writePvd(pvd, s.frames_);
                    s.text("temperature.pvd", pvd.str());
                }
            }
            problem::HeatBalanceReport empty;
            const auto& e = fields ? fields->energy_ : empty;
            const auto row = record({std::to_string(s.steps_.rows_ + 1),
                                     std::to_string(r.step_),
                                     number(r.time_),
                                     number(r.dt_),
                                     r.kind_,
                                     r.status_,
                                     r.solve_ ? std::to_string(r.solve_->iterations_) : "",
                                     r.solve_ ? number(r.solve_->residualNorm_) : "",
                                     r.solve_ ? number(r.solve_->relativeResidual_) : "",
                                     number(r.acceptedResidual_),
                                     fields ? number(fields->minimumK_) : "",
                                     fields ? number(fields->maximumK_) : "",
                                     number(e.storedEnergyJ_),
                                     number(e.storagePowerW_),
                                     number(e.sourcePowerW_),
                                     number(e.prescribedFluxOutW_),
                                     number(e.convectionOutW_),
                                     number(e.dirichletOutW_),
                                     number(e.errorW_),
                                     number(e.toleranceW_),
                                     e.status_,
                                     r.kind_ == "initial" ? "" : number(r.setupS_),
                                     r.kind_ == "initial" ? "" : number(r.solveS_),
                                     number(r.postprocessS_),
                                     path,
                                     r.errorCode_});
            s.append(s.steps_, row, 1);
            if (fields && s.hasMonitors_) {
                std::string rows;
                std::uint64_t count = 0;
                for (const auto& m : fields->monitors_) {
                    rows +=
                        record({std::to_string(s.monitors_.rows_ + (++count)), std::to_string(r.step_), number(r.time_),
                                m.id_, m.kind_, m.quantity_, number(m.value_), m.unit_, "valid", ""});
                }
                s.append(s.monitors_, rows, count);
            }
            if (fields && s.hasEnergy_) {
                std::string rows;
                std::uint64_t count = 0;
                for (const auto& b : e.boundaries_) {
                    rows +=
                        record({std::to_string(s.boundary_.rows_ + (++count)), std::to_string(r.step_), number(r.time_),
                                b.id_, b.kind_, b.kind_ == "temperature" ? "allConstrainedDofs" : "definition",
                                number(b.powerOutW_), "valid"});
                }
                s.append(s.boundary_, rows, count);
            }
            if (e.status_ == "warning") {
                ++s.warnings_;
                s.summary_["diagnostics"].get<O>()["warningCount"] = J(std::to_string(s.warnings_));
                s.log("WARNING", "ENERGY_IMBALANCE", "热量平衡误差超过容差", r.step_);
            }
            s.log("INFO", "STEP_PUBLISHED", r.status_, r.step_);
            if (fields) {
                s.summary_["lastPersistedStep"] = J(std::to_string(r.step_));
            }
            s.commit();
            s.publishedStep_ = r.step_;
        }
        catch (...) {
            s.summary_ = oldSummary;
            s.steps_ = oldSteps;
            s.monitors_ = oldMonitors;
            s.boundary_ = oldBoundary;
            s.frames_ = oldFrames;
            s.broken_ = true;
            // 仅回退本会话表格到已提交完整记录；孤立帧由旧摘要排除。
            for (const auto* table : {&s.steps_, &s.monitors_, &s.boundary_}) {
                std::error_code ec;
                if (fs::exists(s.directory_ / table->path_, ec)) {
                    fs::resize_file(s.directory_ / table->path_, table->bytes_, ec);
                    if (!ec) {
                        s.sizes_[table->path_] = table->bytes_;
                    }
                }
            }
            if (s.transient_) {
                try {
                    if (s.frames_.empty()) {
                        std::error_code ec;
                        fs::remove(s.directory_ / "temperature.pvd", ec);
                        if (!ec) {
                            s.sizes_.erase("temperature.pvd");
                        }
                    }
                    else {
                        std::ostringstream pvd;
                        writePvd(pvd, s.frames_);
                        s.text("temperature.pvd", pvd.str());
                    }
                }
                catch (...) {
                }
            }
            throw;
        }
    }
    void ResultArchive::finish(bool success, std::string code, std::string message, bool outputFailure,
                               bool interrupted) {
        auto& s = *impl_;
        if (s.finished_) {
            throw std::logic_error("重复终结结果会话");
        }
        outputFailure = outputFailure || s.broken_;
        const bool completed = success && !outputFailure;
        s.summary_["status"] = J(completed ? "completed" : interrupted && !outputFailure ? "interrupted" : "failed");
        s.summary_["solveStatus"] = J(success             ? "completed"
                                          : interrupted   ? "interrupted"
                                          : outputFailure ? "running"
                                                          : "failed");
        s.summary_["outputStatus"] = J(outputFailure ? "failed" : "completed");
        s.summary_["finishedAt"] = J(now());
        if (!completed) {
            ++s.errors_;
            s.summary_["diagnostics"].get<O>()["errorCount"] = J(std::to_string(s.errors_));
            s.summary_["diagnostics"].get<O>()["lastError"] = J(O{{"code", J(code)},
                                                                  {"stage", J(outputFailure ? "output" : "solve")},
                                                                  {"message", J(message)},
                                                                  {"step", s.summary_["lastAcceptedStep"]}});
        }
        s.log(completed ? "INFO" : "ERROR", completed ? "RUN_COMPLETED" : code, message, {});
        s.commit();
        s.finished_ = true;
    }
    bool shouldSave(const problem::OutputOptions& o, bool transient, std::uint64_t step, bool final, bool initial) {
        if (o.everySteps_ && !*o.everySteps_) {
            throw std::invalid_argument("输出间隔必须为正");
        }
        if (!transient) {
            return true;
        }
        if (initial) {
            return o.includeInitial_.value_or(true);
        }
        return step % o.everySteps_.value_or(1) == 0 || (final && o.includeFinal_);
    }
} // namespace emberfield::io
