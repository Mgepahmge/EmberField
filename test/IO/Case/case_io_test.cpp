/** @file case_io_test.cpp
 * @brief 验证完整协议、picojson严格适配、精确整数、来源位置与资源限额。
 */
#include <IO/Case/Case.cuh>
#include <IO/Case/ResourceLimits.cuh>
#include <clocale>
#include <gtest/gtest.h>
#include <sstream>
#include "support/CaseFixture.cuh"
namespace {
    using namespace emberfield;
    using namespace casefixture;
    /** @brief 创建含全部主要字段的完整JSON。@pre 无。
     * @return 合法稳态配置。@code auto text=steady(); @endcode */
    std::string steady() {
        return R"({
          "schemaVersion":1,"name":"中文算例","description":"单位为SI",
          "mesh":{"path":"meshes/plate.msh","format":"msh","coordinateScale":0.001},
          "analysis":{"type":"steady","space":"planar2d","thickness":0.01,"geometryTolerance":1e-12,
            "domain":{"kind":"physicalGroup","dimension":2,"tag":9007199254740993}},
          "materials":[{"id":"solid","displayName":"钢","conductivity":{"kind":"constant","value":15}}],
          "materialAssignments":[{"id":"all","region":{"kind":"physicalGroup","dimension":2,"name":"body"},"material":"solid"}],
          "sources":[{"id":"heating","region":{"kind":"physicalGroup","dimension":2,"tag":11},"kind":"volumetric","value":{"kind":"constant","value":100000}}],
          "boundaries":[
            {"id":"cold","region":{"kind":"physicalGroup","dimension":1,"tag":24},"kind":"temperature","temperature":{"kind":"constant","value":300}},
            {"id":"cool","region":{"kind":"physicalGroup","dimension":1,"tag":22},"kind":"convection","coefficient":{"kind":"constant","value":20},"ambientTemperature":{"kind":"constant","value":293.15}},
            {"id":"flux","region":{"kind":"physicalGroup","dimension":1,"tag":23},"kind":"outwardFlux","flux":{"kind":"constant","value":-10}}
          ],
          "boundaryPolicy":{"defaultBoundary":"adiabatic","naturalOverlap":"error"},
          "discretization":{"family":"lagrangeContinuous","order":1,"quadrature":"auto"},
          "linearSolver":{"method":"bicgstab","preconditioner":{"kind":"identity"},"relativeTolerance":1e-8,"absoluteTolerance":1e-12,"maxIterations":1000,"initialGuess":"zero"},
          "output":{"directory":"out","format":"vtu","fields":["temperature"],"overwrite":false,"includeFinal":true}
        })";
    }
    /** @brief 在测试字符串中恰好替换一个片段。@param text 字符串。@param from 原片段。
     * @param to 新片段。@pre 片段存在。@note 无返回值。@code replace(text,"1","2"); @endcode */
    void replace(std::string& text, const std::string& from, const std::string& to) {
        const auto at = text.find(from);
        ASSERT_NE(at, std::string::npos);
        text.replace(at, from.size(), to);
    }
    /** @brief 解析测试字符串。@param text 原文。@param limits 限额。@pre 限额为正。
     * @return IO结果。@code auto r=parse(steady()); @endcode */
    io::CaseReadResult parse(const std::string& text, problem::CaseLimits limits = {}) {
        std::istringstream input(text);
        return io::readCase(input, std::filesystem::path("/cases/plate"), "case.json", limits);
    }
    TEST(CaseIO, ParsesFullSteadyWithExactLargeTagsAndBindsWithoutGpu) {
        auto r = parse(steady());
        ASSERT_TRUE(r.definition_);
        EXPECT_FALSE(r.diagnostics_.hasErrors());
        const auto& c = *r.definition_;
        EXPECT_EQ(c.analysis_.domain_.tag_, 9007199254740993LL);
        EXPECT_EQ(c.mesh_.path_, "/cases/plate/meshes/plate.msh");
        EXPECT_EQ(c.output_.directory_, "/cases/plate/out");
        EXPECT_EQ(c.name_, "中文算例");
        EXPECT_DOUBLE_EQ(c.analysis_.thickness_.value(), 0.01);
        EXPECT_EQ(c.boundaries_.size(), 3);
        EXPECT_TRUE(bind(c).bound_);
        EXPECT_GT(r.locations_.at("/materials/0/conductivity").line_, 1);
    }
    TEST(CaseIO, ParsesFullTransientDefaultsAndOverrides) {
        auto text = steady();
        replace(text, "\"type\":\"steady\"", "\"type\":\"transient\"");
        replace(text, "\"displayName\":\"钢\"",
                "\"displayName\":\"钢\",\"volumetricHeatCapacity\":{\"kind\":\"constant\",\"value\":3900000}");
        replace(
            text, "\"schemaVersion\":1",
            R"("schemaVersion":1,"time":{"start":0,"end":1,"step":0.3,"maxSteps":4,"method":"backwardEuler"},"initialState":{"defaultTemperature":{"kind":"constant","value":300},"overrides":[{"id":"start","region":{"kind":"physicalGroup","dimension":2,"tag":11},"temperature":{"kind":"constant","value":300}}]})");
        replace(text, "\"initialGuess\":\"zero\"", "\"initialGuess\":\"previousStep\"");
        auto r = parse(text);
        ASSERT_TRUE(r.definition_);
        EXPECT_EQ(r.definition_->output_.everySteps_, 1);
        EXPECT_EQ(r.definition_->output_.includeInitial_, true);
        EXPECT_TRUE(bind(*r.definition_).bound_);
        replace(
            text, "\"includeFinal\":true",
            "\"includeFinal\":false,\"includeInitial\":false,\"schedule\":{\"kind\":\"everySteps\",\"interval\":2}");
        r = parse(text);
        ASSERT_TRUE(r.definition_);
        EXPECT_EQ(r.definition_->output_.includeInitial_, false);
        EXPECT_EQ(r.definition_->output_.everySteps_, 2);
    }
    TEST(CaseIO, RejectsDuplicateKeysIncludingEscapedAliases) {
        auto text = steady();
        replace(text, "\"schemaVersion\":1", "\"schemaVersion\":1,\"schemaVersion\":1");
        auto r = parse(text);
        EXPECT_FALSE(r.definition_);
        EXPECT_TRUE(hasCode(r.diagnostics_, "DUPLICATE_KEY"));
        text = steady();
        replace(text, "\"value\":15", "\"value\":15,\"v\\u0061lue\":16");
        EXPECT_TRUE(hasCode(parse(text).diagnostics_, "DUPLICATE_KEY"));
    }
    TEST(CaseIO, RejectsUnknownKeysAndCrossVariantParametersWithSourceLocations) {
        for (const auto& patch : std::vector<std::pair<std::string, std::string>>{
                 {"\"schemaVersion\":1", "\"typo\":1,\"schemaVersion\":1"},
                 {"\"thickness\":0.01", "\"thickness\":0.01,\"foo\":0"},
                 {"\"kind\":\"identity\"", "\"kind\":\"identity\",\"cycles\":1"},
                 {"\"kind\":\"outwardFlux\"",
                  "\"kind\":\"outwardFlux\",\"temperature\":{\"kind\":\"constant\",\"value\":300}"}}) {
            auto text = steady();
            replace(text, patch.first, patch.second);
            auto r = parse(text);
            EXPECT_FALSE(r.definition_);
            ASSERT_TRUE(hasCode(r.diagnostics_, "UNKNOWN_FIELD"));
            EXPECT_TRUE(r.diagnostics_.entries()[0].line_);
            EXPECT_EQ(r.diagnostics_.entries()[0].source_, "case.json");
        }
    }
    TEST(CaseIO, RejectsInvalidJsonNumericGrammarTrailingDataAndOverflow) {
        for (const auto& number :
             {"01", "1.", "-01", "1e", "1e+", "1e999", "9223372036854775808", "-9223372036854775809"}) {
            auto text = steady();
            replace(text, "\"value\":15", std::string("\"value\":") + number);
            auto r = parse(text);
            EXPECT_FALSE(r.definition_) << number;
            EXPECT_TRUE(r.diagnostics_.hasErrors()) << number;
        }
        EXPECT_FALSE(parse(steady() + " {}").definition_);
        EXPECT_FALSE(parse(steady().substr(0, 100)).definition_);
        EXPECT_FALSE(parse("// comment\n" + steady()).definition_);
    }
    TEST(CaseIO, RequiresActualIntegersAndRejectsBooleanOrNumericStrings) {
        for (const auto& value : {"1.0", "1e0", "true", "\"1\""}) {
            auto text = steady();
            replace(text, "\"order\":1", std::string("\"order\":") + value);
            EXPECT_FALSE(parse(text).definition_);
        }
        auto text = steady();
        replace(text, "\"overwrite\":false", "\"overwrite\":0");
        EXPECT_FALSE(parse(text).definition_);
        text = steady();
        replace(text, "\"value\":15", "\"value\":\"15\"");
        EXPECT_FALSE(parse(text).definition_);
    }
    TEST(CaseIO, EnforcesVersionAndExplicitUnsupportedCapabilities) {
        for (const auto& patch :
             std::vector<std::pair<std::string, std::string>>{{"\"schemaVersion\":1", "\"schemaVersion\":2"},
                                                              {"\"planar2d\"", "\"axisymmetric2d\""},
                                                              {"\"kind\":\"constant\"", "\"kind\":\"timeTable\""},
                                                              {"\"kind\":\"identity\"", "\"kind\":\"amg\""},
                                                              {"\"vtu\"", "\"csv\""},
                                                              {"\"auto\"", "\"onePoint\""}}) {
            auto text = steady();
            replace(text, patch.first, patch.second);
            auto r = parse(text);
            EXPECT_FALSE(r.definition_);
            EXPECT_TRUE(r.diagnostics_.hasErrors());
        }
    }
    TEST(CaseIO, EnforcesAllResourceLimitsAndUtf8) {
        problem::CaseLimits limits;
        limits.maxBytes_ = 10;
        EXPECT_TRUE(hasCode(parse(steady(), limits).diagnostics_, "RESOURCE_LIMIT"));
        limits = {};
        limits.maxValues_ = 5;
        EXPECT_FALSE(parse(steady(), limits).definition_);
        limits = {};
        limits.maxStringBytes_ = 2;
        EXPECT_FALSE(parse(steady(), limits).definition_);
        limits = {};
        limits.maxNesting_ = 2;
        EXPECT_FALSE(parse(steady(), limits).definition_);
        auto text = steady();
        replace(text, "中文算例", std::string("\xc0\x80", 2));
        EXPECT_TRUE(hasCode(parse(text).diagnostics_, "INVALID_UTF8"));
    }
    TEST(CaseIO, ParsesSetSelectorsAndRejectsEmptyOrMixedSets) {
        auto text = steady();
        replace(
            text, "{\"kind\":\"physicalGroup\",\"dimension\":2,\"tag\":9007199254740993}",
            R"({"kind":"difference","base":{"kind":"union","items":[{"kind":"physicalGroup","dimension":2,"tag":11},{"kind":"physicalGroup","dimension":2,"tag":11}]},"remove":{"kind":"intersection","items":[{"kind":"physicalGroup","dimension":2,"tag":11},{"kind":"physicalGroup","dimension":2,"tag":9007199254740993}]}})");
        auto r = parse(text);
        ASSERT_TRUE(r.definition_);
        EXPECT_TRUE(hasCode(bind(*r.definition_).diagnostics_, "EMPTY_SELECTION"));
        text = steady();
        replace(text, "{\"kind\":\"physicalGroup\",\"dimension\":2,\"tag\":9007199254740993}",
                R"({"kind":"union","items":[]})");
        EXPECT_FALSE(parse(text).definition_);
    }
    TEST(CaseIO, DistinguishesConfigurationSuccessFromExecutionReadiness) {
        auto text = steady();
        replace(text, "\"order\":1", "\"order\":2");
        auto r = parse(text);
        ASSERT_TRUE(r.definition_);
        auto bound = bind(*r.definition_);
        ASSERT_TRUE(bound.bound_);
        EXPECT_TRUE(bound.bound_->data().capability_.diagnostics_.hasErrors());
        EXPECT_FALSE(bound.bound_->data().capability_.executable());
    }
    TEST(CaseIO, RejectsNulPathsAndPreservesInt64Maximum) {
        auto text = steady();
        replace(text, "meshes/plate.msh", "meshes/plate\\u0000.msh");
        EXPECT_FALSE(parse(text).definition_);
        text = steady();
        replace(text, "9007199254740993", "9223372036854775807");
        auto r = parse(text);
        ASSERT_TRUE(r.definition_);
        EXPECT_EQ(r.definition_->analysis_.domain_.tag_, std::numeric_limits<std::int64_t>::max());
        text = steady();
        replace(text, "\"name\":\"中文算例\"", "\"name\":\"\\uD800\"");
        EXPECT_FALSE(parse(text).definition_);
    }
    TEST(CaseIO, PreservesJsonDecimalPointUnderCommaNumericLocale) {
        /** @brief 只在当前独立测试进程中暂存并恢复数值区域设置。
         * @code NumericLocaleGuard guard; @endcode */
        struct NumericLocaleGuard {
            std::string previous_{std::setlocale(LC_NUMERIC, nullptr)}; ///< 恢复用区域名称。
            /** @brief 恢复区域设置。@pre 无。@note 无参数无返回值，不抛异常。
             * @code { NumericLocaleGuard guard; } @endcode */
            ~NumericLocaleGuard() { std::setlocale(LC_NUMERIC, previous_.c_str()); }
        } guard;
        if (!std::setlocale(LC_NUMERIC, "en_DK.utf8")) {
            GTEST_SKIP() << "逗号数值区域不可用";
        }
        ASSERT_STREQ(std::localeconv()->decimal_point, ",");
        auto result = parse(steady());
        ASSERT_TRUE(result.definition_);
        EXPECT_DOUBLE_EQ(result.definition_->analysis_.thickness_.value(), 0.01);
    }
    TEST(ResourceLimitsIO, DecodesEveryBudgetWithoutFloatingPointConversion) {
        std::istringstream input(R"({"schemaVersion":1,
          "case":{"maxBytes":101,"maxValues":102,"maxStringBytes":103,"maxNesting":104,"maxSelectorDepth":105,"maxDiagnostics":106},
          "mesh":{"maxRecords":201,"maxBytes":202},
          "space":{"maxCells":301,"maxLocalDofs":302,"maxGlobalDofs":303,"maxDiagnostics":304},
          "preparation":{"maxCells":401,"maxDofs":402,"maxFacetUses":403,"maxAssociations":404,"maxDiagnostics":405},
          "assembly":{"maxDofs":501,"maxStencils":502,"maxLocalEntries":503,"maxNonzeros":504},
          "integration":{"maxElements":601,"maxScalars":602},
          "output":{"maxPoints":701,"maxCells":702,"maxHostBytes":703,"maxDeviceBytes":704,"maxFrames":705,"maxMonitors":706,"maxOutputBytes":9007199254740993}})");
        const auto r = io::readResourceLimits(input);
        EXPECT_EQ(r.case_.maxBytes_, 101);
        EXPECT_EQ(r.case_.maxValues_, 102);
        EXPECT_EQ(r.case_.maxStringBytes_, 103);
        EXPECT_EQ(r.case_.maxNesting_, 104);
        EXPECT_EQ(r.case_.maxSelectorDepth_, 105);
        EXPECT_EQ(r.case_.maxDiagnostics_, 106);
        EXPECT_EQ(r.mesh_.maxRecords_, 201);
        EXPECT_EQ(r.mesh_.maxBytes_, 202);
        EXPECT_EQ(r.space_.maxCells_, 301);
        EXPECT_EQ(r.space_.maxLocalDofs_, 302);
        EXPECT_EQ(r.space_.maxGlobalDofs_, 303);
        EXPECT_EQ(r.space_.maxDiagnostics_, 304);
        EXPECT_EQ(r.preparation_.maxCells_, 401);
        EXPECT_EQ(r.preparation_.maxDofs_, 402);
        EXPECT_EQ(r.preparation_.maxFacetUses_, 403);
        EXPECT_EQ(r.preparation_.maxAssociations_, 404);
        EXPECT_EQ(r.preparation_.maxDiagnostics_, 405);
        EXPECT_EQ(r.assembly_.limits_.maxDofs_, 501);
        EXPECT_EQ(r.assembly_.limits_.maxStencils_, 502);
        EXPECT_EQ(r.assembly_.limits_.maxLocalEntries_, 503);
        EXPECT_EQ(r.assembly_.limits_.maxNonzeros_, 504);
        EXPECT_EQ(r.assembly_.gpuLimits_.maxElements_, 601);
        EXPECT_EQ(r.assembly_.gpuLimits_.maxScalars_, 602);
        problem::OutputLimits output;
        r.output_.applyTo(output);
        EXPECT_EQ(output.maxPoints_, 701);
        EXPECT_EQ(output.maxCells_, 702);
        EXPECT_EQ(output.maxHostBytes_, 703);
        EXPECT_EQ(output.maxDeviceBytes_, 704);
        EXPECT_EQ(output.maxFrames_, 705);
        EXPECT_EQ(output.maxMonitors_, 706);
        EXPECT_EQ(output.maxOutputBytes_, 9007199254740993ULL);
        EXPECT_EQ(output.batchCells_, 4096);
        EXPECT_EQ(r.assembly_.batchElements_, 4096);
        EXPECT_DOUBLE_EQ(r.preparation_.maxCondition_, 1e12);
    }
    TEST(ResourceLimitsIO, DefaultsAndFixedParserBudgetRemainIndependent) {
        std::istringstream defaults(R"({"schemaVersion":1,"output":{}})");
        const auto r = io::readResourceLimits(defaults);
        EXPECT_EQ(r.case_.maxValues_, problem::CaseLimits{}.maxValues_);
        EXPECT_FALSE(r.output_.maxHostBytes_);
        std::istringstream deep(std::string(129, '[') + std::string(129, ']'));
        EXPECT_THROW(static_cast<void>(io::readResourceLimits(deep)), std::runtime_error);
        std::istringstream large(std::string(16 * 1024 * 1024 + 1, ' '));
        EXPECT_THROW(static_cast<void>(io::readResourceLimits(large)), std::runtime_error);
        std::istringstream bad;
        bad.setstate(std::ios::badbit);
        EXPECT_THROW(static_cast<void>(io::readResourceLimits(bad)), std::runtime_error);
        EXPECT_THROW(static_cast<void>(io::loadResourceLimits("/nonexistent/limits.json")), std::runtime_error);
    }
    TEST(CaseIO, ReportsMissingFilesBadStreamsAndInvalidCallLimits) {
        EXPECT_TRUE(hasCode(io::loadCase("/nonexistent/case.json").diagnostics_, "IO_ERROR"));
        std::istringstream input;
        input.setstate(std::ios::badbit);
        EXPECT_TRUE(hasCode(io::readCase(input, "/cases").diagnostics_, "IO_ERROR"));
        std::istringstream good(steady());
        EXPECT_THROW(io::readCase(good, "relative"), std::invalid_argument);
        problem::CaseLimits limits;
        limits.maxDiagnostics_ = 0;
        EXPECT_THROW(parse(steady(), limits), std::invalid_argument);
    }
} // namespace
