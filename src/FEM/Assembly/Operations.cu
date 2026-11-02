/** @file Operations.cu
 * @brief 实现无浮点原子的稳定设备归约、候选检查和统一提交。
 */
#include <FEM/Assembly/Operations.cuh>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "../../Numerics/LinearAlgebra/DeviceChecks.cuh"
namespace emberfield::assembly {
    namespace {
        /** @brief 一个目标线程顺序归约局部条目。@param n 目标数。@param targets 目标槽。
         * @param offsets 贡献分段。@param sources 原始局部下标。@param input 输入。
         * @param old 旧全局值。@param next 候选目标值。@param error 错误状态。@pre 数组有效。
         * @note 写候选，无返回。@code reduce<<<grid,128>>>(...); @endcode */
        __global__ void reduce(std::size_t n, const Index* targets, const std::size_t* offsets,
                               const std::size_t* sources, const double* input, const double* old, double* next,
                               int* error) {
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
                 i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
                double value = old[targets[i]];
                for (auto j = offsets[i]; j < offsets[i + 1]; ++j) {
                    const double x = input[sources[j]];
                    value += x;
                    if (!isfinite(x) || !isfinite(value)) {
                        atomicExch(error, 1);
                    }
                }
                next[i] = value;
            }
        }
        /** @brief 提交已检查目标候选。@param n 目标数。@param targets 目标槽。@param next 候选。
         * @param output 输出。@pre 候选全部有限。@note 原位写，无返回。
         * @code commit<<<grid,128>>>(...); @endcode */
        __global__ void commit(std::size_t n, const Index* targets, const double* next, double* output) {
            for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
                 i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
                output[targets[i]] = next[i];
            }
        }
    } // namespace
    AssemblyAccumulator::AssemblyAccumulator(std::shared_ptr<const AssemblyPlan> plan) : plan_(std::move(plan)) {
        if (!plan_) {
            throw std::invalid_argument("组装计划为空");
        }
        CudaBuffer<Index> offsets{std::span<const Index>(plan_->rowOffsets())};
        CudaBuffer<Index> columns{std::span<const Index>(plan_->columns())};
        CudaBuffer<double> zeros(plan_->columns().size());
        device::check(cudaMemset(zeros.deviceData(), 0, zeros.size() * sizeof(double)));
        matrix_ = Matrix::fromBuffers(plan_->dofCount(), plan_->dofCount(), std::move(offsets), std::move(columns),
                                      std::move(zeros));
        load_ = Vector<double>(plan_->dofCount());
        device::check(cudaMemset(load_.deviceData(), 0, load_.size() * sizeof(double)));
    }
    void AssemblyAccumulator::addMatrix(std::size_t first, std::size_t count, const CudaBuffer<double>& entries) {
        add(first, count, entries, true);
    }
    void AssemblyAccumulator::addVector(std::size_t first, std::size_t count, const CudaBuffer<double>& entries) {
        add(first, count, entries, false);
    }
    void AssemblyAccumulator::add(std::size_t first, std::size_t count, const CudaBuffer<double>& entries,
                                  bool matrix) {
        if (matrix_.rowCount() != plan_->dofCount()) {
            throw std::logic_error("累加器结果已接管");
        }
        if (!count || first > plan_->stencils().size() || count > plan_->stencils().size() - first) {
            throw std::out_of_range("组装切片为空或越界");
        }
        device::require(entries, matrix_.rowOffsets().deviceId());
        const auto& offsets = matrix ? plan_->matrixOffsets() : plan_->vectorOffsets();
        const auto start = offsets[first], length = offsets[first + count] - start;
        if (entries.size() != length) {
            throw std::invalid_argument("局部平铺输入数量不匹配");
        }
        std::vector<Index> slots;
        slots.reserve(length);
        if (matrix) {
            slots.insert(slots.end(), plan_->matrixSlots().begin() + start,
                         plan_->matrixSlots().begin() + start + length);
        }
        else {
            for (auto e = first; e < first + count; ++e) {
                for (auto dof : plan_->stencils()[e]) {
                    slots.push_back(static_cast<Index>(dof));
                }
            }
        }
        std::vector<std::size_t> order(length);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return slots[a] < slots[b]; });
        std::vector<Index> targets;
        std::vector<std::size_t> boundaries;
        for (std::size_t i = 0; i < length; ++i) {
            if (!i || slots[order[i]] != targets.back()) {
                targets.push_back(slots[order[i]]);
                boundaries.push_back(i);
            }
        }
        boundaries.push_back(length);
        CudaBuffer<Index> targetDevice{std::span<const Index>(targets)};
        CudaBuffer<std::size_t> boundaryDevice{std::span<const std::size_t>(boundaries)};
        CudaBuffer<std::size_t> orderDevice{std::span<const std::size_t>(order)};
        CudaBuffer<double> next(targets.size());
        device::ErrorFlag status;
        auto* output = matrix ? matrix_.deviceValues() : load_.deviceData();
        reduce<<<device::grid(targets.size()), 128>>>(targets.size(), targetDevice.deviceData(),
                                                      boundaryDevice.deviceData(), orderDevice.deviceData(),
                                                      entries.deviceData(), output, next.deviceData(), status.data());
        status.verify();
        commit<<<device::grid(targets.size()), 128>>>(targets.size(), targetDevice.deviceData(), next.deviceData(),
                                                      output);
        device::check(cudaGetLastError());
        device::check(cudaStreamSynchronize(nullptr));
    }
} // namespace emberfield::assembly
