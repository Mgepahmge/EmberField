/** @file Operations.cu
 * @brief GPU校验规范模式、组合后向Euler系数并从原始矩阵同步消元全部规定值。
 */
#include <LinearAlgebra/IterationContext.cuh>
#include <LinearSystem/Operations.cuh>
#include <algorithm>
#include <cmath>
#include "../LinearAlgebra/DeviceChecks.cuh"
namespace thermofem::systems {
    namespace detail {
        /** @brief 接管已通过检查的系统候选。
         * @code auto s=SystemAccess::make(std::move(a),std::move(b)); @endcode */
        class SystemAccess {
        public:
            /** @brief 接管组合值。@param a 矩阵。@param b 右端。@pre 全部验证成功。
             * @return 拥有系统。@code auto s=make(a,b); @endcode */
            static LinearSystem make(Matrix a, Vector<double> b) { return LinearSystem(std::move(a), std::move(b)); }
        };
    } // namespace detail
    namespace {
        /** @brief 写入互不重复的规定温度。@param count 数量。@param ids 编号。@param g 值。
         * @param x 目标。@pre 数组合法且无重复。@note 写副本，无返回。
         * @code project<<<grid,128>>>(...); @endcode */
        __global__ void project(std::size_t count, const std::uint64_t* ids, const double* g, double* x) {
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
                 i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
                x[ids[i]] = g[i];
            }
        }
        /** @brief 检查温度场物理范围与规定值。@param n 维度。@param x 场。@param count 约束数。
         * @param dofs 约束。@param values 规定值。@param flag 状态。@pre 数组有效。
         * @note 写状态，无返回。@code temperature<<<grid,128>>>(...); @endcode */
        __global__ void temperature(std::size_t n, const double* x, std::size_t count, const std::uint64_t* dofs,
                                    const double* values, int* flag) {
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
                 i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
                if (!isfinite(x[i]) || x[i] < 0) {
                    atomicExch(flag, 1);
                }
                if (i < count && x[dofs[i]] != values[i]) {
                    atomicExch(flag, 1);
                }
            }
        }
        /** @brief 检查行内严格列序、对角与有限数值。@param n 行数。@param rp 偏移。
         * @param ci 列。@param v 值。@param flag 错误状态。@pre CSR基础结构有效。
         * @note 写状态，无返回。@code canonical<<<grid,128>>>(...); @endcode */
        __global__ void canonical(std::size_t n, const assembly::Index* rp, const assembly::Index* ci, const double* v,
                                  int* flag) {
            for (std::size_t r = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; r < n;
                 r += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
                bool diagonal = false;
                for (auto j = rp[r]; j < rp[r + 1]; ++j) {
                    if ((j > rp[r] && ci[j] <= ci[j - 1]) || !isfinite(v[j])) {
                        atomicExch(flag, 1);
                    }
                    diagonal |= ci[j] == static_cast<assembly::Index>(r);
                }
                if (!diagonal) {
                    atomicExch(flag, 1);
                }
            }
        }
        /** @brief 校验向量数值。@param n 数量。@param x 向量。@param flag 状态。
         * @pre 数组有效。@note 写状态，无返回。@code finiteVector<<<grid,128>>>(...); @endcode */
        __global__ void finiteVector(std::size_t n, const double* x, int* flag) {
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
                 i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
                if (!isfinite(x[i])) {
                    atomicExch(flag, 1);
                }
            }
        }
        /** @brief 比较同长结构数组。@param n 长度。@param a 左数组。@param b 右数组。
         * @param flag 状态。@pre 同设备。@note 写状态，无返回。@code equalPattern<<<grid,128>>>(...); @endcode */
        __global__ void equalPattern(std::size_t n, const assembly::Index* a, const assembly::Index* b, int* flag) {
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
                 i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
                if (a[i] != b[i]) {
                    atomicExch(flag, 1);
                }
            }
        }
        /** @brief 逐项组合空间与质量系数。@param n 非零数。@param spatial S。
         * @param mass 输入输出M/dt。@param dt 步长。@param out A。@param flag 状态。
         * @pre 输入有限。@note 写候选，无返回。@code combine<<<grid,128>>>(...); @endcode */
        __global__ void combine(std::size_t n, const double* spatial, double* mass, double dt, double* out, int* flag) {
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
                 i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
                const double old = mass[i], scaled = old / dt;
                out[i] = spatial[i] + scaled;
                mass[i] = scaled;
                if (!isfinite(scaled) || !isfinite(out[i]) || (old != 0 && scaled == 0)) {
                    atomicExch(flag, 1);
                }
            }
        }
        /** @brief 相加历史项和外载荷。@param n 维度。@param load 外载荷。@param rhs 输入历史项及输出。
         * @param flag 状态。@pre 同形。@note 写候选，无返回。@code addLoad<<<grid,128>>>(...); @endcode */
        __global__ void addLoad(std::size_t n, const double* load, double* rhs, int* flag) {
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
                 i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
                rhs[i] += load[i];
                if (!isfinite(rhs[i])) {
                    atomicExch(flag, 1);
                }
            }
        }
        /** @brief 用原矩阵同时执行规定值消元。@param n 维度。@param rp 偏移。@param ci 列。
         * @param old 原矩阵。@param oldRhs 原右端。@param mask 已知掩码。@param g 已知值。
         * @param out 新矩阵。@param rhs 新右端。@param flag 状态。@pre 规范图含全对角。
         * @note 写候选，无返回。@code eliminate<<<grid,128>>>(...); @endcode */
        __global__ void eliminate(std::size_t n, const assembly::Index* rp, const assembly::Index* ci,
                                  const double* old, const double* oldRhs, const unsigned char* mask, const double* g,
                                  double* out, double* rhs, int* flag) {
            for (std::size_t r = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; r < n;
                 r += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
                double b = oldRhs[r];
                for (auto j = rp[r]; j < rp[r + 1]; ++j) {
                    const auto c = ci[j];
                    if (mask[r]) {
                        out[j] = c == static_cast<assembly::Index>(r) ? 1 : 0;
                    }
                    else if (mask[c]) {
                        const double correction = old[j] * g[c];
                        b -= correction;
                        out[j] = 0;
                        if (!isfinite(correction) || !isfinite(b)) {
                            atomicExch(flag, 1);
                        }
                    }
                    else {
                        out[j] = old[j];
                    }
                }
                rhs[r] = mask[r] ? g[r] : b;
                if (!isfinite(rhs[r])) {
                    atomicExch(flag, 1);
                }
            }
        }
        /** @brief 校验规范矩阵。@param a 输入。@pre 无。@note 无返回；非法输入invalid_argument。
         * @code validate(a); @endcode */
        void validate(const Matrix& a) {
            if (!a.rowCount() || a.rowCount() != a.columnCount()) {
                throw std::invalid_argument("系统需要非空方阵");
            }
            const int dev = a.rowOffsets().deviceId();
            device::require(a.rowOffsets(), dev);
            device::require(a.columnIndices(), dev);
            device::require(a.values(), dev);
            device::ErrorFlag flag;
            canonical<<<device::grid(a.rowCount()), 128>>>(a.rowCount(), a.rowOffsets().deviceData(),
                                                           a.columnIndices().deviceData(), a.deviceValues(),
                                                           flag.data());
            try {
                flag.verify();
            }
            catch (const std::overflow_error&) {
                throw std::invalid_argument("系统CSR必须规范、含全对角且数值有限");
            }
        }
        /** @brief 校验向量形状、设备与有限性。@param x 向量。@param n 维度。@param dev 设备。
         * @pre 无。@note 无返回；非法输入invalid_argument。@code validate(x,n,dev); @endcode */
        void validate(const Vector<double>& x, std::size_t n, int dev) {
            if (x.size() != n) {
                throw std::invalid_argument("系统向量维度不匹配");
            }
            device::require(x.values(), dev);
            device::ErrorFlag flag;
            finiteVector<<<device::grid(n), 128>>>(n, x.deviceData(), flag.data());
            try {
                flag.verify();
            }
            catch (const std::overflow_error&) {
                throw std::invalid_argument("系统向量非有限");
            }
        }
    } // namespace
    void validateTemperatureState(const Vector<double>& x, std::span<const std::uint64_t> dofs,
                                  std::span<const double> values) {
        if (!x.size() || dofs.size() != values.size()) {
            throw std::invalid_argument("温度场或约束数组错误");
        }
        device::require(x.values(), x.deviceId());
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            if (dofs[i] >= x.size() || (i && dofs[i] <= dofs[i - 1]) || !std::isfinite(values[i]) || values[i] < 0) {
                throw std::invalid_argument("温度约束无效");
            }
        }
        CudaBuffer<std::uint64_t> ids(dofs);
        CudaBuffer<double> g(values);
        device::ErrorFlag flag;
        temperature<<<device::grid(x.size()), 128>>>(x.size(), x.deviceData(), dofs.size(), ids.deviceData(),
                                                     g.deviceData(), flag.data());
        try {
            flag.verify();
        }
        catch (const std::overflow_error&) {
            throw std::invalid_argument("历史温度必须有限非负并满足规定值");
        }
    }
    Vector<double> projectTemperatureState(const Vector<double>& x, std::span<const std::uint64_t> dofs,
                                           std::span<const double> values) {
        validateTemperatureState(x, {}, {});
        if (dofs.size() != values.size()) {
            throw std::invalid_argument("温度约束数组错误");
        }
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            if (dofs[i] >= x.size() || (i && dofs[i] <= dofs[i - 1]) || !std::isfinite(values[i]) || values[i] < 0) {
                throw std::invalid_argument("温度约束无效");
            }
        }
        Vector<double> result = x;
        if (!dofs.empty()) {
            CudaBuffer<std::uint64_t> ids(dofs);
            CudaBuffer<double> g(values);
            project<<<device::grid(dofs.size()), 128>>>(dofs.size(), ids.deviceData(), g.deviceData(),
                                                        result.deviceData());
            device::check(cudaGetLastError());
            device::check(cudaStreamSynchronize(nullptr));
        }
        return result;
    }
    LinearSystem buildSteadySystem(const Matrix& spatial, const Vector<double>& load) {
        validate(spatial);
        validate(load, spatial.rowCount(), spatial.rowOffsets().deviceId());
        return detail::SystemAccess::make(spatial, load);
    }
    LinearSystem buildBackwardEulerSystem(const Matrix& spatial, const Matrix& capacity, const Vector<double>& load,
                                          const Vector<double>& previous, double dt) {
        if (!std::isfinite(dt) || dt <= 0) {
            throw std::invalid_argument("实际时间步长必须正有限");
        }
        validate(spatial);
        validate(capacity);
        const auto n = spatial.rowCount(), nz = spatial.nonZeroCount();
        const int dev = spatial.rowOffsets().deviceId();
        device::require(capacity.values(), dev);
        if (capacity.rowCount() != n || capacity.nonZeroCount() != nz) {
            throw std::invalid_argument("空间和容量CSR模式不同");
        }
        validate(load, n, dev);
        validate(previous, n, dev);
        device::ErrorFlag pattern;
        equalPattern<<<device::grid(n + 1), 128>>>(n + 1, spatial.rowOffsets().deviceData(),
                                                   capacity.rowOffsets().deviceData(), pattern.data());
        equalPattern<<<device::grid(nz), 128>>>(nz, spatial.columnIndices().deviceData(),
                                                capacity.columnIndices().deviceData(), pattern.data());
        try {
            pattern.verify();
        }
        catch (const std::overflow_error&) {
            throw std::invalid_argument("空间和容量CSR模式不同");
        }
        Matrix a = spatial, scaledMass = capacity;
        device::ErrorFlag status;
        combine<<<device::grid(nz), 128>>>(nz, spatial.deviceValues(), scaledMass.deviceValues(), dt, a.deviceValues(),
                                           status.data());
        status.verify();
        linalg::IterationContext<double, assembly::Index> massOperation(scaledMass);
        Vector<double> rhs(n);
        massOperation.multiply(previous, rhs);
        addLoad<<<device::grid(n), 128>>>(n, load.deviceData(), rhs.deviceData(), status.data());
        status.verify();
        return detail::SystemAccess::make(std::move(a), std::move(rhs));
    }
    LinearSystem applyDirichlet(const LinearSystem& system, std::span<const std::uint64_t> dofs,
                                std::span<const double> values) {
        const auto& old = system.matrix();
        const auto n = old.rowCount();
        device::require(old.values(), old.rowOffsets().deviceId());
        if (dofs.size() != values.size()) {
            throw std::invalid_argument("规定值数组长度不匹配");
        }
        std::vector<unsigned char> mask(n, 0);
        std::vector<double> g(n, 0);
        for (std::size_t i = 0; i < dofs.size(); ++i) {
            if (dofs[i] >= n || (i && dofs[i] <= dofs[i - 1]) || !std::isfinite(values[i])) {
                throw std::invalid_argument("规定值编号必须有序唯一且数值有限");
            }
            mask[dofs[i]] = 1;
            g[dofs[i]] = values[i];
        }
        CudaBuffer<unsigned char> maskDevice{std::span<const unsigned char>(mask)};
        CudaBuffer<double> prescribed{std::span<const double>(g)};
        Matrix a = old;
        Vector<double> rhs(n);
        device::ErrorFlag flag;
        eliminate<<<device::grid(n), 128>>>(n, old.rowOffsets().deviceData(), old.columnIndices().deviceData(),
                                            old.deviceValues(), system.rhs().deviceData(), maskDevice.deviceData(),
                                            prescribed.deviceData(), a.deviceValues(), rhs.deviceData(), flag.data());
        flag.verify();
        return detail::SystemAccess::make(std::move(a), std::move(rhs));
    }
} // namespace thermofem::systems
