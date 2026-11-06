/** @file tui_progress_test.cpp @brief 验证残差指标、观察契约及数值结果不变。 */
#include <Simulation/Case/Run.cuh>
#include <Solver/Iterative/BiCgStab.cuh>
#include <Solver/Iterative/Cg.cuh>
#include <array>
#include <cmath>
#include <filesystem>
#include <gtest/gtest.h>
#include <limits>
#include <unistd.h>
#include "../../../src/IO/Case/ConvergenceProgress.cuh"
namespace {
    using namespace emberfield;
    using cli::ConvergenceProgress;
    TEST(ConvergenceProgress, UsesFirstCompletedIterationAndAllowsRegression) {
        ConvergenceProgress p;
        p.update({0, 20, 2, IterationKind::Initial, ResidualKind::True, {}});
        EXPECT_FALSE(p.fraction());
        p.update({1, 10, 2, IterationKind::Iteration, ResidualKind::Recursive, {}});
        ASSERT_TRUE(p.fraction());
        EXPECT_DOUBLE_EQ(*p.fraction(), 0);
        p.update({2, 6, 2, IterationKind::Iteration, ResidualKind::Recursive, {}});
        EXPECT_DOUBLE_EQ(*p.fraction(), 0.5);
        p.update({3, 8, 2, IterationKind::Iteration, ResidualKind::True, {}});
        EXPECT_NEAR(*p.fraction(), 0.25, 1e-15);
        p.update({4, 30, 2, IterationKind::Iteration, ResidualKind::True, {}});
        EXPECT_DOUBLE_EQ(*p.fraction(), 0);
        p.update({5, 1, 2, IterationKind::Final, ResidualKind::True, SolveStatus::Converged});
        EXPECT_DOUBLE_EQ(*p.fraction(), 1);
        EXPECT_FALSE(p.accepted());
        p.accept();
        EXPECT_TRUE(p.accepted());
    }
    TEST(ConvergenceProgress, HandlesZeroIterationNonfiniteAndReset) {
        ConvergenceProgress p;
        p.update({0, 0, 0, IterationKind::Initial, ResidualKind::True, {}});
        EXPECT_EQ(p.fraction(), 1);
        p.update({0, 12, 2, IterationKind::Final, ResidualKind::True, SolveStatus::MaxIterations});
        EXPECT_FALSE(p.fraction());
        p.update(
            {1, std::numeric_limits<double>::infinity(), 2, IterationKind::Iteration, ResidualKind::Recursive, {}});
        EXPECT_FALSE(p.fraction());
        p.update({1, 2, 2, IterationKind::Iteration, ResidualKind::True, {}});
        EXPECT_EQ(p.fraction(), 1);
        p.accept();
        p.update({0, 10, 2, IterationKind::Initial, ResidualKind::True, {}});
        EXPECT_FALSE(p.accepted());
        EXPECT_FALSE(p.fraction());
        EXPECT_FALSE(p.status());
    }
    TEST(ConvergenceProgress, LargeResidualAndZeroToleranceRemainBounded) {
        ConvergenceProgress p;
        p.update({1, 1e308, 0, IterationKind::Iteration, ResidualKind::True, {}});
        p.update({2, 5e307, 0, IterationKind::Iteration, ResidualKind::True, {}});
        EXPECT_DOUBLE_EQ(*p.fraction(), 0.5);
        p.update({3, std::numeric_limits<double>::quiet_NaN(), 0, IterationKind::Final, ResidualKind::True,
                  SolveStatus::NonFinite});
        EXPECT_FALSE(p.fraction());
    }
    /** @brief 固定容量避免无异常回调内分配。@code Capture capture; @endcode */
    struct Capture final : IterationObserver<double> {
        std::array<IterationInfo<double>, 256> events_{}; ///< 有界事件。
        std::size_t size_{}; ///< 已记录事件数。
        /** @brief 保存事件。@param info 标量。@pre 无。@note 无返回值。@code onIteration(info); @endcode */
        void onIteration(const IterationInfo<double>& info) noexcept override {
            if (size_ < events_.size()) {
                events_[size_++] = info;
            }
        }
    };
    /** @brief 在同一个固定SPD系统上比较有无观察者。@param bicg 是否使用BiCGStab。
     * @pre CUDA可用。@note 无返回值。@code compareSolve(false); @endcode */
    void compareSolve(bool bicg) {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
            GTEST_SKIP();
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        const std::vector<int> offsets{0, 2, 5, 7}, columns{0, 1, 0, 1, 2, 1, 2};
        const std::vector<double> values{4, -1, -1, 4, -1, -1, 3}, b{1, 2, 3}, zeros(3);
        CsrMatrix<double> matrix(3, 3, offsets, columns, values);
        std::unique_ptr<Solver<double>> solver;
        if (bicg) {
            solver =
                std::make_unique<BiCgStab<double>>(BiCgStabOptions<double>{}, std::make_unique<Identity<double>>());
        }
        else {
            solver = std::make_unique<Cg<double>>();
        }
        solver->setup(matrix);
        Vector<double> rhs{std::span<const double>(b)}, x{std::span<const double>(zeros)},
            y{std::span<const double>(zeros)};
        const auto baseline = solver->apply(rhs, x);
        Capture capture, previous;
        solver->setObserver(&previous);
        {
            IterationSubscription<double> subscription(*solver, &capture);
            const auto actual = solver->apply(rhs, y);
            EXPECT_EQ(actual.status_, baseline.status_);
            EXPECT_EQ(actual.iterations_, baseline.iterations_);
            EXPECT_DOUBLE_EQ(actual.residualNorm_, baseline.residualNorm_);
        }
        EXPECT_EQ(solver->setObserver(nullptr), &previous);
        EXPECT_EQ(linalg::detail::hostCopy(x), linalg::detail::hostCopy(y));
        ASSERT_GE(capture.size_, 3u);
        EXPECT_EQ(capture.events_[0].kind_, IterationKind::Initial);
        EXPECT_EQ(capture.events_[capture.size_ - 1].kind_, IterationKind::Final);
        EXPECT_EQ(capture.events_[capture.size_ - 1].residualNorm_, baseline.residualNorm_);
        int expected = 1;
        for (std::size_t i = 0; i < capture.size_; ++i) {
            if (capture.events_[i].kind_ == IterationKind::Iteration) {
                EXPECT_EQ(capture.events_[i].iteration_, expected++);
            }
            EXPECT_NEAR(capture.events_[i].tolerance_, 1e-8 * std::sqrt(14.), 1e-20);
        }
        EXPECT_EQ(expected - 1, baseline.iterations_);
    }
    TEST(IterationObserver, CgDoesNotChangeNumericalResults) { compareSolve(false); }
    TEST(IterationObserver, BiCgStabDoesNotChangeNumericalResults) { compareSolve(true); }
    TEST(IterationObserver, SubscriptionRestoresAfterExceptionAndNestedScope) {
        Cg<double> solver;
        Capture original, outer, inner;
        solver.setObserver(&original);
        try {
            IterationSubscription<double> first(solver, &outer);
            {
                IterationSubscription<double> second(solver, &inner);
            }
            EXPECT_EQ(solver.setObserver(&outer), &outer);
            throw std::runtime_error("test");
        }
        catch (const std::runtime_error&) {
        }
        EXPECT_EQ(solver.setObserver(nullptr), &original);
    }
    TEST(IterationObserver, MovingSolverDoesNotTransferBorrowedObserver) {
        Cg<double> source;
        Capture capture;
        source.setObserver(&capture);
        Cg<double> destination(std::move(source));
        EXPECT_EQ(destination.setObserver(nullptr), nullptr);
        EXPECT_EQ(source.setObserver(nullptr), &capture);
    }
    /** @brief 实际运行事件的无分配检查器。@code RunCapture capture; @endcode */
    struct RunCapture final : problem::RunObserver {
        problem::RunControl* control_{}; ///< 可选在首个提交后取消。
        std::size_t total_{}, committed_{}, started_{}, accepted_{}, finished_{}; ///< 协议计数。
        std::array<std::size_t, 5> workTotals_{}; ///< 实际执行类别分母。
        /** @brief 校验批计数。@param info 批快照。@pre 无。@note 无返回值。@code onProgress(info); @endcode */
        void onProgress(const problem::ProgressInfo& info) noexcept override {
            valid_ &= !info.determinate_ || info.completed_ <= info.total_;
            if (info.determinate_) {
                workTotals_[static_cast<int>(info.work_)] =
                    std::max(workTotals_[static_cast<int>(info.work_)], info.total_);
            }
        }
        bool valid_{true}; ///< 回调内只累计判定，断言在调用后执行。
        int code_{-1}; ///< 最终退出码。
        /** @brief 检查事件。@param e 事件。@pre 无。@note 无返回值。@code onEvent(e); @endcode */
        void onEvent(const problem::RunEvent& e) noexcept override {
            using problem::RunEventKind;
            if (e.kind_ == RunEventKind::Metadata) {
                total_ = e.totalSteps_;
            }
            if (e.kind_ == RunEventKind::StepStarted) {
                ++started_;
            }
            if (e.kind_ == RunEventKind::SolveAccepted) {
                ++accepted_;
            }
            if (e.kind_ == RunEventKind::StepCommitted) {
                valid_ &= e.step_ == committed_ + 1 && e.totalSteps_ == total_ && accepted_ > committed_;
                committed_ = e.step_;
                if (control_) {
                    control_->requestCancel();
                }
            }
            if (e.kind_ == RunEventKind::Finished) {
                ++finished_;
                code_ = e.exitCode_;
            }
        }
    };
    TEST(RunObserver, CountsEventAlignedStepsAndCancelsAfterCommittedStep) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
            GTEST_SKIP();
        }
        const auto base =
            std::filesystem::temp_directory_path() / ("emberfield-time-observer-" + std::to_string(getpid()));
        std::filesystem::remove_all(base);
        const auto input = std::filesystem::path(EMBERFIELD_SOURCE_DIR) / "example/cases/time-varying.json";
        RunCapture full;
        EXPECT_EQ(problem::runCase(input, base / "full", nullptr, {}, &full, nullptr), 0);
        EXPECT_TRUE(full.valid_);
        EXPECT_EQ(full.total_, 9u);
        EXPECT_EQ(full.committed_, 9u);
        EXPECT_EQ(full.workTotals_[static_cast<int>(problem::ProgressWork::Flux)], 1u);
        EXPECT_EQ(full.workTotals_[static_cast<int>(problem::ProgressWork::Convection)], 1u);
        EXPECT_EQ(full.started_, 9u);
        EXPECT_EQ(full.accepted_, 9u);
        EXPECT_EQ(full.finished_, 1u);
        problem::RunControl control;
        RunCapture cancelled;
        cancelled.control_ = &control;
        EXPECT_EQ(problem::runCase(input, base / "cancelled", nullptr, {}, &cancelled, &control), 130);
        EXPECT_TRUE(cancelled.valid_);
        EXPECT_EQ(cancelled.committed_, 1u);
        EXPECT_EQ(cancelled.accepted_, 1u);
        EXPECT_EQ(cancelled.finished_, 1u);
        EXPECT_EQ(cancelled.code_, 130);
        std::filesystem::remove_all(base);
    }
    TEST(IterationObserver, ZeroAndFirstIterationAndBreakdownDoNotInventEvents) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
            GTEST_SKIP();
        }
        const std::vector<int> offsets{0, 1}, columns{0};
        const std::vector<double> one{1}, zero{0}, negative{-1};
        CsrMatrix<double> identity(1, 1, offsets, columns, one), invalid(1, 1, offsets, columns, negative);
        Cg<double> solver;
        Capture capture;
        solver.setup(identity);
        solver.setObserver(&capture);
        Vector<double> rhs{std::span<const double>(one)}, x{std::span<const double>(one)};
        EXPECT_EQ(solver.apply(rhs, x).iterations_, 0);
        ASSERT_EQ(capture.size_, 2u);
        EXPECT_EQ(capture.events_[0].kind_, IterationKind::Initial);
        EXPECT_EQ(capture.events_[1].kind_, IterationKind::Final);
        capture.size_ = 0;
        x = Vector<double>{std::span<const double>(zero)};
        EXPECT_EQ(solver.apply(rhs, x).iterations_, 1);
        ASSERT_EQ(capture.size_, 3u);
        EXPECT_EQ(capture.events_[1].kind_, IterationKind::Iteration);
        EXPECT_EQ(capture.events_[1].residualKind_, ResidualKind::True);
        solver.setup(invalid);
        capture.size_ = 0;
        x = Vector<double>{std::span<const double>(zero)};
        EXPECT_EQ(solver.apply(rhs, x).status_, SolveStatus::Breakdown);
        ASSERT_EQ(capture.size_, 2u);
        EXPECT_EQ(capture.events_[1].kind_, IterationKind::Final);
        ConvergenceProgress progress;
        for (std::size_t i = 0; i < capture.size_; ++i) {
            progress.update(capture.events_[i]);
        }
        EXPECT_FALSE(progress.fraction());
    }
    TEST(RunObserver, CommitAndFinishedFollowSuccessfulSteadyRun) {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || !count) {
            GTEST_SKIP();
        }
        const auto directory =
            std::filesystem::temp_directory_path() / ("emberfield-observer-" + std::to_string(getpid()));
        std::filesystem::remove_all(directory);
        RunCapture capture;
        const auto result = problem::runCase(std::filesystem::path(EMBERFIELD_SOURCE_DIR) / "example/cases/steady.json",
                                             directory, nullptr, {}, &capture, nullptr);
        EXPECT_EQ(result, 0);
        EXPECT_TRUE(capture.valid_);
        EXPECT_EQ(capture.total_, 1u);
        EXPECT_EQ(capture.committed_, 1u);
        EXPECT_EQ(capture.finished_, 1u);
        EXPECT_EQ(capture.code_, result);
        std::filesystem::remove_all(directory);
    }
} // namespace
