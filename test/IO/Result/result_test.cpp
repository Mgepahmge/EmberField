/** @file result_test.cpp @brief 输出编码、配置、后处理与会话故障的分层验收。 */
#include <IO/Case/Case.cuh>
#include <IO/Case/CommandLine.cuh>
#include <IO/Case/Write.cuh>
#include <IO/Result/Archive.cuh>
#include <IO/Result/Pvd.cuh>
#include <IO/Result/Vtu.cuh>
#include <Simulation/Case/System.cuh>
#include <Simulation/Thermal/PostprocessOperations.cuh>
#include <chrono>
#include <cmath>
#include <fstream>
#include <regex>
#include <sstream>
#include <support/AssemblyFixture.cuh>
#include <support/CaseFixture.cuh>
#include <support/IntegrationFixture.cuh>
namespace {
    using namespace emberfield;
    using namespace emberfield::problem;
    namespace fs = std::filesystem;
    /** @brief 唯一临时目录；析构仅清理本测试创建的目录。@code Temporary t; @endcode */
    struct Temporary {
        fs::path path_ = fs::temp_directory_path() /
            ("emberfield-result-test-" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())); ///< 测试私有根。
        Temporary() { fs::create_directory(path_); } ///< 获取唯一目录。
        ~Temporary() {
            std::error_code ec;
            fs::remove_all(path_, ec);
        } ///< 删除私有测试内容。
    };
    /** @brief 完整准备夹具。@param c 定义。@pre 合法。@return 成功上下文。@code auto p=prepare(c); @endcode */
    std::shared_ptr<const PreparedThermalCase> prepare(CaseDefinition c,
                                                       mesh::MeshData data = casefixture::meshData()) {
        auto b = casefixture::bind(c, std::move(data));
        if (!b.bound_) {
            throw std::runtime_error("binding");
        }
        auto space = buildTemperatureSpace(*b.bound_);
        if (!space.space_) {
            throw std::runtime_error("space");
        }
        auto p = prepareThermalCase(b.bound_, space.space_);
        if (!p.ok()) {
            throw std::runtime_error("prepare");
        }
        return p.prepared();
    }
    /** @brief 构造恒温global状态。@param n 维数。@param value 温度。@pre CUDA可用。
     * @return 拥有设备数据。@code auto t=constant(4,300); @endcode */
    Vector<double> constant(std::size_t n, double value) {
        return Vector<double>(std::span<const double>(std::vector<double>(n, value)));
    }
    using ResultGpu = assemblyfixture::GpuAssembly;
    TEST(ResultPaths, SingleFolderAndExecutableRoot) {
        EXPECT_TRUE(io::executableDirectory().is_absolute());
        const auto root = fs::temp_directory_path();
        EXPECT_EQ(io::outputDirectory("a.b.json", root), root / "a.b");
        for (const auto& name : {"../escape", ".", "..", "a/b", "a\\b", "", "CON", "x:", "tail."}) {
            EXPECT_THROW(io::outputDirectory("x.json", root, std::string(name)), std::invalid_argument) << name;
        }
        EXPECT_EQ(io::outputDirectory("x.json", root, std::string("plate")), root / "plate");
    }
    TEST(ResultSchedule, InitialIntervalAndFinalUnion) {
        OutputOptions o;
        o.everySteps_ = 3;
        o.includeInitial_ = false;
        o.includeFinal_ = false;
        EXPECT_FALSE(io::shouldSave(o, true, 0, false, true));
        EXPECT_FALSE(io::shouldSave(o, true, 2, true, false));
        EXPECT_TRUE(io::shouldSave(o, true, 3, true, false));
        o.includeFinal_ = true;
        EXPECT_TRUE(io::shouldSave(o, true, 2, true, false));
        EXPECT_TRUE(io::shouldSave(o, false, 0, true, false));
    }
    TEST(ResultArrays, StrictShapeUtf8AndFiniteValues) {
        field::NumericArray a{"t", 1, 2, "K", std::vector<double>{300, 301}};
        EXPECT_NO_THROW(a.validate());
        a.components_ = 2;
        EXPECT_THROW(a.validate(), std::invalid_argument);
        a.components_ = 1;
        std::get<std::vector<double>>(a.values_)[0] = INFINITY;
        EXPECT_THROW(a.validate(), std::invalid_argument);
        EXPECT_THROW(field::validateText(std::string("\xc0\xaf", 2)), std::invalid_argument);
        EXPECT_THROW(field::validateText(std::string("\0", 1)), std::invalid_argument);
    }
    TEST(ResultPvd, RelativePathsIncreasingTimesAndEscaping) {
        std::ostringstream out;
        const std::vector<io::PvdEntry> entries{{0, "fields/a&b.vtu"}, {.25, "fields/c.vtu"}};
        io::writePvd(out, entries);
        EXPECT_NE(out.str().find("a&amp;b.vtu"), std::string::npos);
        const std::vector<io::PvdEntry> bad{{1, "../a.vtu"}};
        EXPECT_THROW(io::writePvd(out, bad), std::invalid_argument);
        const std::vector<io::PvdEntry> same{{1, "a.vtu"}, {1, "b.vtu"}};
        EXPECT_THROW(io::writePvd(out, same), std::invalid_argument);
    }
    TEST(ResultConfiguration, NormalizedRoundTripAndProgrammaticValidation) {
        auto c = casefixture::definition();
        c.mesh_.path_ = fs::absolute("mesh.msh");
        c.output_.fields_ = {OutputField::Temperature, OutputField::HeatFlux, OutputField::TemperatureGradient};
        c.output_.monitors_ = {{"n", NodeMonitor{UINT64_MAX}},
                               {"p", PointMonitor{{.2, .3, 0}, 1e-10}},
                               {"r", RegionMonitor{casefixture::group(2, 11), {RegionStatistic::TemperatureMean}}}};
        auto json = io::serializeCase(c);
        std::istringstream input(json);
        auto r = io::readCase(input, fs::current_path());
        ASSERT_TRUE(r.definition_);
        EXPECT_EQ(r.definition_->output_.fields_, c.output_.fields_);
        EXPECT_EQ(std::get<NodeMonitor>(r.definition_->output_.monitors_[0].value_).nodeTag_, UINT64_MAX);
        c.output_.fields_.push_back(OutputField::Temperature);
        EXPECT_TRUE(validateCase(c).hasErrors());
        c.output_.fields_ = {OutputField::Temperature};
        c.output_.limits_.batchCells_ = 0;
        EXPECT_TRUE(validateCase(c).hasErrors());
    }
    TEST_F(ResultGpu, AffineGradientFourierAndIndependentReduction) {
        thermal::P1CellSample cell{};
        cell.dimension_ = 2;
        cell.volume_ = .5;
        cell.capacity_ = 2;
        cell.dofs_[0] = 0;
        cell.dofs_[1] = 1;
        cell.dofs_[2] = 2;
        cell.gradients_[0] = -1;
        cell.gradients_[1] = -1;
        cell.gradients_[3] = 1;
        cell.gradients_[7] = 1;
        Vector<double> t(std::span<const double>(std::vector<double>{300, 302, 303}));
        const auto values = thermal::evaluateP1Gradient(std::span(&cell, 1), t);
        EXPECT_DOUBLE_EQ(values[0].gradient_[0], 2);
        EXPECT_DOUBLE_EQ(values[0].gradient_[1], 3);
        EXPECT_DOUBLE_EQ(values[0].gradient_[2], 0);
        const auto flux =
            thermal::applyFourierLaw(std::vector<std::array<double, 3>>{{2, 3, 0}}, std::vector<double>{5});
        EXPECT_DOUBLE_EQ(flux[0][0], -10);
        EXPECT_DOUBLE_EQ(flux[0][1], -15);
        const auto stats = thermal::reduceRegion(values, std::vector<double>{.5});
        EXPECT_DOUBLE_EQ(stats.minimum_, 300);
        EXPECT_DOUBLE_EQ(stats.maximum_, 303);
        EXPECT_NEAR(stats.integral_, .5 * 905 / 3, 1e-12);
        EXPECT_NEAR(stats.energy_, 905 / 3., 1e-12);
        EXPECT_DOUBLE_EQ(thermal::sumValues(t), 905);
        EXPECT_DOUBLE_EQ(thermal::sumValues(t, std::vector<std::uint64_t>{0, 2}), 603);
    }
    TEST_F(ResultGpu, IdentityMappingFieldsAndBinaryBlockOffsets) {
        auto c = casefixture::definition();
        c.output_.fields_ = {OutputField::Temperature, OutputField::HeatFlux, OutputField::TemperatureGradient};
        auto p = prepare(c);
        auto model = assembleThermalModel(buildThermalAssemblyPlan(p));
        ThermalPostprocessPlan plan(p);
        auto fields = plan.evaluate(model, constant(4, 300), nullptr, {}, 0, {});
        EXPECT_EQ(plan.grid()->pointCount(), 4);
        EXPECT_EQ(plan.grid()->cellCount(), 2);
        EXPECT_EQ(fields.energy_.status_, "passed");
        const auto statisticsOnly = plan.evaluate(model, constant(4, 300), nullptr, {}, 0, {}, false, false);
        EXPECT_FALSE(statisticsOnly.snapshot_.has_value());
        EXPECT_DOUBLE_EQ(statisticsOnly.minimumK_, fields.minimumK_);
        EXPECT_DOUBLE_EQ(statisticsOnly.maximumK_, fields.maximumK_);
        EXPECT_EQ(statisticsOnly.energy_.status_, fields.energy_.status_);
        EXPECT_EQ(statisticsOnly.energy_.errorW_, fields.energy_.errorW_);
        const auto& tags = std::get<std::vector<std::uint64_t>>(plan.grid()->pointIdentity()[0].values_);
        EXPECT_EQ(tags[0], 5000000001ULL);
        std::ostringstream ascii;
        ascii << std::hex << std::showpos;
        io::writeVtu(ascii, *fields.snapshot_, io::VtuEncoding::Ascii);
        EXPECT_NE(ascii.str().find("9007199254740993"), std::string::npos);
        EXPECT_NE(ascii.str().find("type=\"Float64\""), std::string::npos);
        std::ostringstream binary;
        io::writeVtu(binary, *fields.snapshot_);
        auto data = binary.str();
        const auto marker = data.find("<AppendedData encoding=\"raw\">");
        ASSERT_NE(marker, std::string::npos);
        auto start = data.find('_', marker) + 1;
        std::regex offset("offset=\"([0-9]+)\"");
        const auto header = data.substr(0, marker);
        std::uint64_t expected = 0;
        unsigned blocks = 0;
        for (std::sregex_iterator i(header.begin(), header.end(), offset), end; i != end; ++i) {
            const auto position = std::stoull((*i)[1]);
            EXPECT_EQ(position, expected);
            ASSERT_LE(start + position + 8, data.size());
            std::uint64_t bytes = 0;
            for (unsigned k = 0; k < 8; ++k) {
                bytes |= static_cast<std::uint64_t>(static_cast<unsigned char>(data[start + position + k])) << (k * 8);
            }
            expected += 8 + bytes;
            ++blocks;
        }
        EXPECT_GT(blocks, 15);
        EXPECT_EQ(data.substr(start + expected, 15).find("</AppendedData>"), 0);
        EXPECT_NE(data.substr(start + expected).find("</AppendedData>"), std::string::npos);
    }
    TEST_F(ResultGpu, PointTieRegionStatisticsAndInitialEnergy) {
        auto c = casefixture::definition();
        casefixture::makeTransient(c);
        c.output_.monitors_ = {{"node", NodeMonitor{5000000001ULL}},
                               {"point", PointMonitor{{.5, .5, 0}, 0}},
                               {"region",
                                RegionMonitor{casefixture::group(2, 11),
                                              {RegionStatistic::TemperatureMin, RegionStatistic::TemperatureMax,
                                               RegionStatistic::TemperatureMean, RegionStatistic::StoredEnergy}}}};
        auto p = prepare(c);
        auto model = assembleThermalModel(buildThermalAssemblyPlan(p));
        ThermalPostprocessPlan plan(p);
        auto t = constant(4, 300);
        auto initial = plan.evaluate(model, t, nullptr, {}, 0, 0, true);
        EXPECT_EQ(initial.energy_.status_, "unavailable");
        ASSERT_TRUE(initial.energy_.storedEnergyJ_);
        EXPECT_NEAR(*initial.energy_.storedEnergyJ_, 11700000, 1e-6);
        EXPECT_FALSE(initial.energy_.dirichletOutW_);
        EXPECT_EQ(initial.monitors_.size(), 6);
        EXPECT_NE(plan.monitorDictionary().find("\"cellRow\":\"0\""), std::string::npos);
        auto next = plan.evaluate(model, t, &t, .1, 1, .1);
        EXPECT_NEAR(*next.energy_.storagePowerW_, 0, 1e-12);
        EXPECT_EQ(next.energy_.status_, "passed");
        EXPECT_NEAR(next.monitors_[4].value_, 300, 1e-10);
        c.output_.monitors_ = {{"outside", PointMonitor{{2, 2, 0}, 1e-10}}};
        EXPECT_THROW(ThermalPostprocessPlan invalid(prepare(c)), std::invalid_argument);
    }
    TEST_F(ResultGpu, ArchiveLocksBackupAndCsvEscaping) {
        Temporary temp;
        auto c = casefixture::definition();
        c.mesh_.path_ = temp.path_ / "mesh.msh";
        c.output_.overwrite_ = true;
        c.output_.monitors_ = {{"a,\"b", NodeMonitor{8}}};
        auto p = prepare(c);
        auto model = assembleThermalModel(buildThermalAssemblyPlan(p));
        ThermalPostprocessPlan plan(p);
        auto fields = plan.evaluate(model, constant(4, 300), nullptr, {}, 0, {});
        const auto directory = temp.path_ / "result";
        {
            io::ResultArchive archive(directory, temp.path_ / "case.json", model, plan);
            EXPECT_THROW(io::ResultArchive second(directory, temp.path_ / "case.json", model, plan),
                         std::runtime_error);
            archive.accepted(0, {});
            io::StepRecord record;
            record.solve_ = SolveResult<double>{SolveStatus::Converged, 0, 0, 0};
            record.acceptedResidual_ = 0;
            archive.publishStep(record, &fields, true);
            archive.finish(true);
        }
        EXPECT_TRUE(fs::exists(directory / "fields/state_0000000000.vtu"));
        EXPECT_FALSE(fs::exists(directory / "temperature.pvd"));
        std::ifstream csvFile(directory / "monitors.csv");
        const std::string csv((std::istreambuf_iterator<char>(csvFile)), {});
        EXPECT_NE(csv.find("\"a,\"\"b\""), std::string::npos);
        {
            io::ResultArchive second(directory, temp.path_ / "case.json", model, plan);
            second.finish(false, "TEST", "失败");
        }
        bool backup = false;
        for (const auto& entry : fs::directory_iterator(temp.path_)) {
            if (entry.path().filename().string().starts_with("result.backup-")) {
                backup = true;
            }
        }
        EXPECT_TRUE(backup);
        const auto unrelated = temp.path_ / "unrelated";
        fs::create_directory(unrelated);
        std::ofstream(unrelated / "keep") << "important";
        EXPECT_THROW(io::ResultArchive bad(unrelated, temp.path_ / "case.json", model, plan), std::invalid_argument);
        EXPECT_TRUE(fs::exists(unrelated / "keep"));
        EXPECT_THROW(io::ResultArchive bad(temp.path_, temp.path_ / "case.json", model, plan), std::invalid_argument);
    }
    TEST_F(ResultGpu, FailedAttemptDoesNotCreateNormalFrame) {
        Temporary temp;
        auto p = prepare(casefixture::definition());
        auto model = assembleThermalModel(buildThermalAssemblyPlan(p));
        ThermalPostprocessPlan plan(p);
        io::ResultArchive archive(temp.path_ / "run", temp.path_ / "case.json", model, plan);
        io::StepRecord r;
        r.kind_ = "failedAttempt";
        r.status_ = "nonFinite";
        r.solve_ = SolveResult<double>{SolveStatus::NonFinite, 2, INFINITY, INFINITY};
        archive.publishStep(r, nullptr, false);
        archive.finish(false, "SOLVER_FAILED", "未收敛");
        EXPECT_TRUE(fs::is_empty(temp.path_ / "run/fields"));
    }
    TEST_F(ResultGpu, TetrahedronHeatFluxAndPointProjection) {
        auto data = integrationfixture::data(3);
        data.elements_[0].physicalTags_ = std::vector<std::int64_t>{11};
        auto c = casefixture::definition();
        c.analysis_.space_ = SpaceModel::Solid3d;
        c.analysis_.thickness_.reset();
        c.analysis_.domain_ = casefixture::group(3, 11);
        c.materialAssignments_[0].region_ = casefixture::group(3, 11);
        c.boundaries_.clear();
        c.output_.fields_ = {OutputField::Temperature, OutputField::TemperatureGradient, OutputField::HeatFlux};
        c.output_.monitors_ = {{"inside", PointMonitor{{.2, .3, .1}, 0}},
                               {"projected", PointMonitor{{-.000001, .3, .1}, .000002}}};
        auto p = prepare(c, std::move(data));
        ThermalPostprocessPlan plan(p);
        Vector<double> t(std::span<const double>(std::vector<double>{300, 302, 303, 304}));
        auto fields = plan.evaluateThermalFields(t, 0, {});
        const auto& g = std::get<std::vector<double>>(fields.cellFields()[0].values_);
        ASSERT_EQ(g.size(), 3);
        EXPECT_DOUBLE_EQ(g[0], 2);
        EXPECT_DOUBLE_EQ(g[1], 3);
        EXPECT_DOUBLE_EQ(g[2], 4);
        const auto& q = std::get<std::vector<double>>(fields.cellFields()[1].values_);
        EXPECT_DOUBLE_EQ(q[0], -30);
        EXPECT_DOUBLE_EQ(q[1], -45);
        EXPECT_DOUBLE_EQ(q[2], -60);
        const auto monitors = plan.evaluateMonitors(t);
        EXPECT_NEAR(monitors[0].value_, 301.7, 1e-12);
        EXPECT_NEAR(monitors[1].value_, 301.3, 1e-12);
        std::ostringstream ascii;
        io::writeVtu(ascii, fields, io::VtuEncoding::Ascii);
        EXPECT_NE(ascii.str().find("Name=\"types\" NumberOfComponents=\"1\" format=\"ascii\">10 "), std::string::npos);
    }
    TEST_F(ResultGpu, FileFailureKeepsCommittedPrefixAndDistinctCursors) {
        Temporary temp;
        auto c = casefixture::definition();
        casefixture::makeTransient(c);
        auto p = prepare(c);
        auto model = assembleThermalModel(buildThermalAssemblyPlan(p));
        ThermalPostprocessPlan plan(p);
        auto t = constant(4, 300);
        auto fields = plan.evaluate(model, t, nullptr, {}, 0, 0, true);
        io::ResultArchive archive(temp.path_ / "run", temp.path_ / "case.json", model, plan);
        archive.accepted(0, 0);
        io::StepRecord r;
        r.kind_ = "initial";
        r.status_ = "provided";
        r.time_ = 0;
        archive.publishStep(r, &fields, true);
        std::ifstream before(temp.path_ / "run/run-summary.json");
        const std::string manifest((std::istreambuf_iterator<char>(before)), {});
        archive.accepted(1, .1);
        auto next = plan.evaluate(model, t, &t, .1, 1, .1);
        r.step_ = 1;
        r.kind_ = "accepted";
        r.status_ = "converged";
        r.time_ = .1;
        r.dt_ = .1;
        r.solve_ = SolveResult<double>{SolveStatus::Converged, 0, 0, 0};
        r.acceptedResidual_ = 0;
        fs::create_directory(temp.path_ / "run/fields/state_0000000001.vtu.tmp");
        EXPECT_THROW(archive.publishStep(r, &next, true), std::ios_base::failure);
        std::ifstream after(temp.path_ / "run/run-summary.json");
        EXPECT_EQ(std::string((std::istreambuf_iterator<char>(after)), {}), manifest);
        archive.finish(false, "OUTPUT_IO", "故障注入", true);
        std::ifstream finished(temp.path_ / "run/run-summary.json");
        const std::string summary((std::istreambuf_iterator<char>(finished)), {});
        EXPECT_NE(summary.find("\"lastAcceptedStep\": \"1\""), std::string::npos);
        EXPECT_NE(summary.find("\"lastPersistedStep\": \"0\""), std::string::npos);
        EXPECT_FALSE(fs::exists(temp.path_ / "run/fields/state_0000000001.vtu"));
    }
    TEST_F(ResultGpu, OutputLimitsAreCheckedBeforeAllocationOrFrameCommit) {
        auto c = casefixture::definition();
        c.output_.limits_.maxHostBytes_ = 1;
        EXPECT_THROW(ThermalPostprocessPlan plan(prepare(c)), std::length_error);
        c = casefixture::definition();
        c.output_.limits_.maxDeviceBytes_ = 1;
        EXPECT_THROW(ThermalPostprocessPlan plan(prepare(c)), std::length_error);
        Temporary temp;
        c = casefixture::definition();
        c.output_.limits_.maxOutputBytes_ = 16;
        auto p = prepare(c);
        auto model = assembleThermalModel(buildThermalAssemblyPlan(p));
        ThermalPostprocessPlan plan(p);
        EXPECT_THROW(io::ResultArchive archive(temp.path_ / "run", temp.path_ / "case.json", model, plan),
                     std::exception);
        EXPECT_FALSE(fs::exists(temp.path_ / "run.emberfield-lock"));
    }
} // namespace
