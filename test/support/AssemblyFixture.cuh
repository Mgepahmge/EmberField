/** @file AssemblyFixture.cuh
 * @brief 全局组装和系统测试的独立主机稠密对照工具。
 */
#pragma once
#include <Assembly/Operations.cuh>
#include <gtest/gtest.h>
namespace assemblyfixture {
    /** @brief 下载缓冲区。@tparam T 元素。@param b 缓冲区。@pre 已同步。
     * @return 主机数组。@code auto x=host(buffer); @endcode */
    template <thermofem::BufferElement T>
    std::vector<T> host(const thermofem::CudaBuffer<T>& b) {
        std::vector<T> v(b.size());
        b.copyToHost(v);
        return v;
    }
    /** @brief 转成稠密行主序。@param a CSR。@pre 无。
     * @return 主机稠密值。@code auto d=dense(a); @endcode */
    inline std::vector<double> dense(const thermofem::assembly::Matrix& a) {
        auto r = host(a.rowOffsets()), c = host(a.columnIndices());
        auto v = host(a.values());
        std::vector<double> out(a.rowCount() * a.columnCount(), 0);
        for (std::size_t i = 0; i < a.rowCount(); ++i) {
            for (auto j = r[i]; j < r[i + 1]; ++j) {
                out[i * a.columnCount() + c[j]] += v[j];
            }
        }
        return out;
    }
    /** @brief 构造全存储规范矩阵。@param n 维度。@param v 行主序值。
     * @pre n平方等于值数。@return 设备CSR。@code auto a=matrix(2,{2,-1,-1,2}); @endcode */
    inline thermofem::assembly::Matrix matrix(std::size_t n, std::vector<double> v) {
        std::vector<thermofem::assembly::Index> r(n + 1), c(n * n);
        for (std::size_t i = 0; i <= n; ++i) {
            r[i] = i * n;
        }
        for (std::size_t i = 0; i < n * n; ++i) {
            c[i] = i % n;
        }
        return {n, n, r, c, v};
    }
    /** @brief 相对绝对混合逐项比较。@param a 结果。@param b 参照。@pre 无。
     * @note 无返回，测试断言。@code near(a,b); @endcode */
    inline void near(const std::vector<double>& a, const std::vector<double>& b) {
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            EXPECT_NEAR(a[i], b[i], 2e-11 * std::max(1., std::abs(b[i]))) << i;
        }
    }
    /** @brief 要求可用GPU的测试夹具。@code TEST_F(GpuAssembly,Name){} @endcode */
    class GpuAssembly : public testing::Test {
    protected:
        /** @brief 无GPU时跳过，无参数无返回值。@pre 无。@code SetUp(); @endcode */
        void SetUp() override {
            int n = 0;
            if (cudaGetDeviceCount(&n) != cudaSuccess || !n) {
                GTEST_SKIP() << "CUDA设备不可用";
            }
        }
    };
} // namespace assemblyfixture
