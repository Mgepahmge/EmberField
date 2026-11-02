/** @file PostprocessOperations.cu
 * @brief 有界批次的CUDA场重建和积分，不包含求解流程或文件操作。
 */
#include <Simulation/Thermal/PostprocessOperations.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
namespace emberfield::thermal {
    namespace {
        /** @brief 内核输入设备检查。@param t 温度。@pre 无。@note 无返回值，非法invalid_argument。
         * @code device(t); @endcode */
        void device(const Vector<double>& t) {
            int current;
            auto code = cudaGetDevice(&current);
            if (code != cudaSuccess) {
                throw CudaError(code, "postprocess device");
            }
            if (t.memoryType() != MemoryType::Global || t.deviceId() != current) {
                throw std::invalid_argument("后处理要求当前设备global温度");
            }
        }
        /** @brief 一线程一传感器的gather/重心插值。 */
        __global__ void interpolateKernel(const TemperatureSample* samples, std::size_t n, const double* temperature,
                                          double* out) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            double value = 0;
            for (unsigned a = 0; a < samples[i].count_; ++a) {
                value += samples[i].weights_[a] * temperature[samples[i].dofs_[a]];
            }
            out[i] = value;
        }
        /** @brief 同步检查内核；无参数无前提，无返回值。@code sync(); @endcode */
        void sync() {
            auto c = cudaGetLastError();
            if (c != cudaSuccess) {
                throw CudaError(c, "postprocess launch");
            }
            c = cudaDeviceSynchronize();
            if (c != cudaSuccess) {
                throw CudaError(c, "postprocess sync");
            }
        }
        /** @brief 重建一批单元；各线程拥有一个单元。 */
        __global__ void gradientKernel(const P1CellSample* cells, std::size_t n, const double* t, P1CellValue* out) {
            const auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            const auto& c = cells[i];
            P1CellValue v{};
            v.minimum_ = INFINITY;
            v.maximum_ = -INFINITY;
            for (unsigned a = 0; a <= c.dimension_; ++a) {
                const double x = t[c.dofs_[a]];
                v.minimum_ = fmin(v.minimum_, x);
                v.maximum_ = fmax(v.maximum_, x);
                v.integral_ += x * c.volume_ / (c.dimension_ + 1);
                for (unsigned k = 0; k < 3; ++k) {
                    v.gradient_[k] += x * c.gradients_[a * 3 + k];
                }
            }
            v.energy_ = v.integral_ * c.capacity_;
            out[i] = v;
        }
        /** @brief 逐单元Fourier运算。 */
        __global__ void fluxKernel(const double* g, const double* k, std::size_t n, double* q) {
            auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i < n) {
                for (int j = 0; j < 3; ++j) {
                    q[i * 3 + j] = -k[i] * g[i * 3 + j];
                }
            }
        }
        /** @brief 逐侧求值预积分功率。 */
        __global__ void boundaryKernel(const BoundaryPowerSample* s, std::size_t n, const double* t, double* p) {
            auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            if (i >= n) {
                return;
            }
            double v = s[i].constant_;
            for (unsigned a = 0; a < s[i].count_; ++a) {
                v += s[i].weights_[a] * t[s[i].dofs_[a]];
            }
            p[i] = v;
        }
        /** @brief 分块共享内存归约，每块写一个独立部分和。 */
        __global__ void sumKernel(const double* x, const std::uint64_t* ids, std::size_t n, double* partial) {
            __shared__ double cache[256];
            double value = 0;
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
                 i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
                value += x[ids ? ids[i] : i];
            }
            cache[threadIdx.x] = value;
            __syncthreads();
            for (unsigned s = 128; s; s /= 2) {
                if (threadIdx.x < s) {
                    cache[threadIdx.x] += cache[threadIdx.x + s];
                }
                __syncthreads();
            }
            if (!threadIdx.x) {
                partial[blockIdx.x] = cache[0];
            }
        }
        /** @brief 有限性检查。@param v 值。@pre 无。@return 原值；非法overflow_error。@code finite(x); @endcode */
        double finite(double v) {
            if (!std::isfinite(v)) {
                throw std::overflow_error("后处理结果非有限");
            }
            return v;
        }
        /** @brief 一个线程块以网格步进归约有界区域批。 */
        __global__ void regionKernel(const P1CellValue* values, const double* volumes, std::size_t n,
                                     RegionStatistics* result) {
            __shared__ RegionStatistics shared[256];
            RegionStatistics s{INFINITY, -INFINITY, 0, 0, 0};
            for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
                s.minimum_ = fmin(s.minimum_, values[i].minimum_);
                s.maximum_ = fmax(s.maximum_, values[i].maximum_);
                s.integral_ += values[i].integral_;
                s.energy_ += values[i].energy_;
                s.volume_ += volumes[i];
            }
            shared[threadIdx.x] = s;
            __syncthreads();
            for (unsigned stride = 128; stride; stride /= 2) {
                if (threadIdx.x < stride) {
                    auto& a = shared[threadIdx.x];
                    const auto& b = shared[threadIdx.x + stride];
                    a.minimum_ = fmin(a.minimum_, b.minimum_);
                    a.maximum_ = fmax(a.maximum_, b.maximum_);
                    a.integral_ += b.integral_;
                    a.energy_ += b.energy_;
                    a.volume_ += b.volume_;
                }
                __syncthreads();
            }
            if (!threadIdx.x) {
                *result = shared[0];
            }
        }
    } // namespace
    std::vector<P1CellValue> evaluateP1Gradient(std::span<const P1CellSample> cells, const Vector<double>& t) {
        device(t);
        if (cells.empty()) {
            return {};
        }
        for (const auto& c : cells) {
            if ((c.dimension_ != 2 && c.dimension_ != 3) || !(c.volume_ > 0) || !std::isfinite(c.volume_)) {
                throw std::invalid_argument("非法P1单元");
            }
            for (unsigned a = 0; a <= c.dimension_; ++a) {
                if (c.dofs_[a] >= t.size()) {
                    throw std::invalid_argument("P1 DOF越界");
                }
            }
            for (double x : c.gradients_) {
                finite(x);
            }
        }
        CudaBuffer<P1CellSample> input(cells);
        CudaBuffer<P1CellValue> output(cells.size());
        gradientKernel<<<static_cast<unsigned>((cells.size() + 255) / 256), 256>>>(input.deviceData(), cells.size(),
                                                                                   t.deviceData(), output.deviceData());
        sync();
        std::vector<P1CellValue> result(cells.size());
        output.copyToHost(result);
        for (const auto& v : result) {
            finite(v.minimum_);
            finite(v.maximum_);
            finite(v.integral_);
            finite(v.energy_);
            for (double x : v.gradient_) {
                finite(x);
            }
        }
        return result;
    }
    std::vector<std::array<double, 3>> applyFourierLaw(std::span<const std::array<double, 3>> g,
                                                       std::span<const double> k) {
        if (g.size() != k.size()) {
            throw std::invalid_argument("Fourier数组不等长");
        }
        if (g.empty()) {
            return {};
        }
        for (auto x : k) {
            if (!(finite(x) > 0)) {
                throw std::invalid_argument("导热率必须为正");
            }
        }
        for (const auto& v : g) {
            for (double x : v) {
                finite(x);
            }
        }
        CudaBuffer<std::array<double, 3>> dg(g), dq(g.size());
        CudaBuffer<double> dk(k);
        fluxKernel<<<static_cast<unsigned>((g.size() + 255) / 256), 256>>>(
            reinterpret_cast<const double*>(dg.deviceData()), dk.deviceData(), g.size(),
            reinterpret_cast<double*>(dq.deviceData()));
        sync();
        std::vector<std::array<double, 3>> result(g.size());
        dq.copyToHost(result);
        for (auto v : result) {
            for (double x : v) {
                finite(x);
            }
        }
        return result;
    }
    std::vector<double> integrateBoundaryPowerBatch(std::span<const BoundaryPowerSample> s, const Vector<double>& t) {
        device(t);
        if (s.empty()) {
            return {};
        }
        for (const auto& v : s) {
            if (v.count_ > 3) {
                throw std::invalid_argument("非法侧DOF数");
            }
            finite(v.constant_);
            for (unsigned a = 0; a < v.count_; ++a) {
                if (v.dofs_[a] >= t.size()) {
                    throw std::invalid_argument("侧DOF越界");
                }
                finite(v.weights_[a]);
            }
        }
        CudaBuffer<BoundaryPowerSample> ds(s);
        CudaBuffer<double> out(s.size());
        boundaryKernel<<<static_cast<unsigned>((s.size() + 255) / 256), 256>>>(ds.deviceData(), s.size(),
                                                                               t.deviceData(), out.deviceData());
        sync();
        std::vector<double> result(s.size());
        out.copyToHost(result);
        for (double x : result) {
            finite(x);
        }
        return result;
    }
    double sumValues(const Vector<double>& x, std::span<const std::uint64_t> selected) {
        device(x);
        const auto n = selected.empty() ? x.size() : selected.size();
        if (!n) {
            return 0;
        }
        for (auto i : selected) {
            if (i >= x.size()) {
                throw std::invalid_argument("归约索引越界");
            }
        }
        CudaBuffer<std::uint64_t> ids(selected);
        const auto blocks = std::min<std::size_t>(256, (n + 255) / 256);
        CudaBuffer<double> partial(blocks), total(1);
        sumKernel<<<static_cast<unsigned>(blocks), 256>>>(x.deviceData(), selected.empty() ? nullptr : ids.deviceData(),
                                                          n, partial.deviceData());
        sumKernel<<<1, 256>>>(partial.deviceData(), nullptr, blocks, total.deviceData());
        sync();
        double value;
        total.copyToHost(std::span<double>(&value, 1));
        return finite(value);
    }
    RegionStatistics reduceRegion(std::span<const P1CellValue> values, std::span<const double> volumes) {
        if (values.empty() || values.size() != volumes.size()) {
            throw std::invalid_argument("非法区域批");
        }
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (!(finite(volumes[i]) > 0)) {
                throw std::invalid_argument("区域体积非正");
            }
            finite(values[i].minimum_);
            finite(values[i].maximum_);
            finite(values[i].integral_);
            finite(values[i].energy_);
        }
        CudaBuffer<P1CellValue> input(values);
        CudaBuffer<double> v(volumes);
        CudaBuffer<RegionStatistics> out(1);
        regionKernel<<<1, 256>>>(input.deviceData(), v.deviceData(), values.size(), out.deviceData());
        sync();
        RegionStatistics result;
        out.copyToHost(std::span<RegionStatistics>(&result, 1));
        finite(result.integral_);
        finite(result.energy_);
        finite(result.volume_);
        return result;
    }
    std::vector<double> interpolateTemperature(std::span<const TemperatureSample> samples,
                                               const Vector<double>& temperature) {
        device(temperature);
        if (samples.empty()) {
            return {};
        }
        for (const auto& s : samples) {
            if (!s.count_ || s.count_ > 4) {
                throw std::invalid_argument("非法插值DOF数");
            }
            double sum = 0;
            for (unsigned a = 0; a < s.count_; ++a) {
                if (s.dofs_[a] >= temperature.size() || !std::isfinite(s.weights_[a]) || s.weights_[a] < 0) {
                    throw std::invalid_argument("非法插值索引或权重");
                }
                sum += s.weights_[a];
            }
            if (std::abs(sum - 1) > 128 * std::numeric_limits<double>::epsilon()) {
                throw std::invalid_argument("插值权重和不为1");
            }
        }
        CudaBuffer<TemperatureSample> ds(samples);
        CudaBuffer<double> out(samples.size());
        interpolateKernel<<<static_cast<unsigned>((samples.size() + 255) / 256), 256>>>(
            ds.deviceData(), samples.size(), temperature.deviceData(), out.deviceData());
        sync();
        std::vector<double> result(samples.size());
        out.copyToHost(result);
        for (auto x : result) {
            finite(x);
        }
        return result;
    }
} // namespace emberfield::thermal
