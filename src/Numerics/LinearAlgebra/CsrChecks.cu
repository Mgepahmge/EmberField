/** @file CsrChecks.cu @brief 规范CSR结构和有限数值的同步设备检查，不保存可变输入缓存。 */
#include <NVTXProfiler.cuh>
#include <Numerics/LinearAlgebra/Backend.cuh>
#include "DeviceChecks.cuh"
/** @brief CSR规范结构和有限值的设备检查。 */
namespace emberfield::linalg::detail {
    namespace {
        /** @brief 一线程检查一行严格列序。@tparam I CSR索引。
         * @param n 行数。@param rp 偏移。@param ci 列。@param bad 状态。
         * @pre 基础CSR合法。@return 无。@code ordered<<<grid,128>>>(n,rp,ci,bad); @endcode */
        template <CsrIndex I>
        __global__ void ordered(std::size_t n, const I* rp, const I* ci, int* bad) {
            for (auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < n;
                 row += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
                for (auto j = rp[row] + (rp[row] < rp[row + 1] ? 1 : 0); j < rp[row + 1]; ++j) {
                    if (ci[j] <= ci[j - 1]) {
                        atomicExch(bad, 1);
                    }
                }
            }
        }
        /** @brief 检查有限值。@tparam T 精度。@param n 数量。@param x 输入。@param bad 状态。
         * @pre 设备数组有效。@return 无。@code finiteKernel<<<grid,128>>>(n,x,bad); @endcode */
        template <Scalar T>
        __global__ void finiteKernel(std::size_t n, const T* x, int* bad) {
            for (auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
                 i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
                if (!isfinite(x[i])) {
                    atomicExch(bad, 1);
                }
            }
        }
    } // namespace
    template <CsrIndex I>
    bool canonicalPattern(std::size_t rows, const I* offsets, const I* columns) {
        EMBERFIELD_NVTX_SCOPE("emberfield::linalg::detail::canonicalPattern");
        if (!rows) {
            return true;
        }
        CudaBuffer<int> bad(1);
        device::check(cudaMemset(bad.deviceData(), 0, sizeof(int)));
        ordered<<<device::grid(rows), 128>>>(rows, offsets, columns, bad.deviceData());
        device::check(cudaGetLastError());
        int result{};
        device::check(cudaMemcpy(&result, bad.deviceData(), sizeof(int), cudaMemcpyDeviceToHost));
        return result == 0;
    }
    template <Scalar T>
    bool finiteValues(std::size_t count, const T* values) {
        EMBERFIELD_NVTX_SCOPE("emberfield::linalg::detail::finiteValues");
        if (!count) {
            return true;
        }
        CudaBuffer<int> bad(1);
        device::check(cudaMemset(bad.deviceData(), 0, sizeof(int)));
        finiteKernel<<<device::grid(count), 128>>>(count, values, bad.deviceData());
        device::check(cudaGetLastError());
        int result{};
        device::check(cudaMemcpy(&result, bad.deviceData(), sizeof(int), cudaMemcpyDeviceToHost));
        return result == 0;
    }
    template bool canonicalPattern<std::int32_t>(std::size_t, const std::int32_t*, const std::int32_t*);
    template bool canonicalPattern<std::int64_t>(std::size_t, const std::int64_t*, const std::int64_t*);
    template bool finiteValues<float>(std::size_t, const float*);
    template bool finiteValues<double>(std::size_t, const double*);
} // namespace emberfield::linalg::detail
