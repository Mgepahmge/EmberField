/** @file amg_move_test.cpp
 * @brief 验证 AMG 不透明层次的移动所有权和 apply 别名契约。
 */
#include <Solver/Preconditioner/Amg.cuh>
#include <gtest/gtest.h>

namespace {
    using namespace emberfield;

    /** @brief 构造供生命周期测试使用的短对角占优矩阵。
     * @tparam T 浮点精度。@tparam I CSR 索引。
     * @param n 矩阵阶数。@pre n>0。@return 三对角方阵。
     * @code auto a = makeMatrix<double, int>(4); @endcode */
    template <linalg::Scalar T, CsrIndex I>
    CsrMatrix<T, I> makeMatrix(int n) {
        std::vector<I> offsets{0}, columns;
        std::vector<T> values;
        for (int row = 0; row < n; ++row) {
            if (row > 0) {
                columns.push_back(static_cast<I>(row - 1));
                values.push_back(T(-1));
            }
            columns.push_back(static_cast<I>(row));
            values.push_back(T(2));
            if (row + 1 < n) {
                columns.push_back(static_cast<I>(row + 1));
                values.push_back(T(-1));
            }
            offsets.push_back(static_cast<I>(values.size()));
        }
        return {static_cast<std::size_t>(n), static_cast<std::size_t>(n), offsets, columns, values};
    }

    /** @brief 检查 AMGs 状态在两类移动中保持唯一并可使用。
     * @tparam T 精度。@tparam I 索引。@pre CUDA 设备可用。@note 无返回值。
     * @code checkMove<float, std::int64_t>(); @endcode */
    template <linalg::Scalar T, CsrIndex I>
    void checkMove() {
        const auto matrix = makeMatrix<T, I>(4);
        const std::vector<T> ones(4, T(1));
        Vector<T> exact{std::span<const T>(ones)};
        auto rhs = linalg::multiply(matrix, exact);

        Amg<T, I> source;
        source.setup(matrix);
        Amg<T, I> moved(std::move(source));
        EXPECT_EQ(source.levelCount(), 0u);
        EXPECT_THROW(source.apply(rhs, exact), std::logic_error);

        Amg<T, I> target;
        target.setup(matrix);
        target = std::move(moved);
        EXPECT_EQ(moved.levelCount(), 0u);
        ASSERT_EQ(target.levelCount(), 1u);

        auto* const address = rhs.data();
        target.apply(rhs, rhs);
        EXPECT_EQ(rhs.data(), address);
        for (const T value : linalg::detail::hostCopy(rhs)) {
            EXPECT_NEAR(value, T(1), 1e-5);
        }
    }

    TEST(AmgLifecycle, MovePreservesOpaqueStateAndAliasing) {
        int deviceCount{};
        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || !deviceCount) {
            GTEST_SKIP() << "CUDA unavailable";
        }
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        checkMove<float, int>();
        checkMove<double, int>();
        checkMove<float, std::int64_t>();
        checkMove<double, std::int64_t>();
    }
} // namespace
