/** @file nvtx_profiler_test.cpp @brief 编译开关、参数消除、作用域与提前释放契约测试。 */
#include <NVTXProfiler.cuh>
#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>

#if defined(EMBERFIELD_ENABLE_NVTX)
TEST(NvtxProfiler, scopesSupportNestingAndExplicitRelease) {
    static_assert(!std::is_copy_constructible_v<emberfield::NvtxProfiler>);
    static_assert(!std::is_move_constructible_v<emberfield::NvtxProfiler>);
    emberfield::NvtxProfiler outer("nvtx-test-outer");
    EXPECT_TRUE(outer.isActive());
    {
        emberfield::NvtxProfiler inner("nvtx-test-inner");
        EXPECT_TRUE(inner.isActive());
        inner.release();
        inner.release();
        EXPECT_FALSE(inner.isActive());
    }
    EXPECT_TRUE(outer.isActive());
    outer.release();
    EXPECT_FALSE(outer.isActive());
}

TEST(NvtxProfiler, macrosEvaluateOnceAndUnwindOnException) {
    int evaluations = 0;
    auto name = [&]() {
        ++evaluations;
        return "nvtx-test-macro";
    };
    EXPECT_THROW(
        {
            EMBERFIELD_NVTX_SCOPE(name());
            EMBERFIELD_NVTX_SCOPE_COLOR(name(), emberfield::NvtxColor::Green);
            EMBERFIELD_NVTX_SCOPE_RANDOM(name());
            throw std::runtime_error("scope-unwind");
        },
        std::runtime_error);
    EXPECT_EQ(evaluations, 3);
    // 同行宏展开必须生成不同局部变量，即使外层测试宏把整个代码块放在同一逻辑行。
    EMBERFIELD_NVTX_SCOPE("nvtx-test-repeated");
    EMBERFIELD_NVTX_SCOPE("nvtx-test-repeated");
    emberfield::NvtxProfiler counted("nvtx-test-counted", emberfield::NvtxProfiler::ColorMode::None, 0, true);
    counted.release();
    emberfield::NvtxProfiler again("nvtx-test-counted", emberfield::NvtxProfiler::ColorMode::None, 0, true);
    EXPECT_TRUE(again.isActive());
}
#else
TEST(NvtxProfiler, disabledMacrosDiscardArgumentsAndNeedNoNvtxTypes) {
    int evaluations = 0;
    EMBERFIELD_NVTX_SCOPE((++evaluations, "unused"));
    EMBERFIELD_NVTX_SCOPE_COLOR((++evaluations, "unused"), ++evaluations);
    EMBERFIELD_NVTX_SCOPE_RANDOM((++evaluations, "unused"));
    EMBERFIELD_NVTX_SCOPE(notDeclaredAndMustNotBeCompiled());
    EMBERFIELD_NVTX_SCOPE_COLOR(alsoNotDeclared(), emberfield::NvtxColor::Green);
    EXPECT_EQ(evaluations, 0);
}
#endif
