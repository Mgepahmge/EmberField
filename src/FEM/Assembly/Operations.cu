/** @file Operations.cu
 * @brief 实现无浮点原子的稳定设备归约、候选检查和统一提交。
 */
#include <FEM/Assembly/Operations.cuh>
#include <NVTXProfiler.cuh>
#include <algorithm>
#include <cmath>
#include <limits>
#include <list>
#include <numeric>
#include <optional>
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
    AssemblyAccumulator::AssemblyAccumulator(std::shared_ptr<const AssemblyPlan> plan) :
        AssemblyAccumulator(plan, std::make_shared<detail::DeviceAssemblyPlan>(plan)) {}
    AssemblyAccumulator::AssemblyAccumulator(std::shared_ptr<const AssemblyPlan> plan,
                                             std::shared_ptr<detail::DeviceAssemblyPlan> execution) :
        plan_(std::move(plan)), execution_(std::move(execution)) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::assembly::AssemblyAccumulator::AssemblyAccumulator",
                                    emberfield::NvtxColor::Green);
        if (!plan_ || !execution_ || execution_->source() != plan_) {
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
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::assembly::AssemblyAccumulator::add", emberfield::NvtxColor::Green);
        EMBERFIELD_NVTX_SCOPE(matrix ? "emberfield::assembly::accumulateMatrix"
                                     : "emberfield::assembly::accumulateVector");
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
        execution_->accumulate(first, count, entries, matrix, matrix ? matrix_.deviceValues() : load_.deviceData());
    }

    namespace detail {
        /** @brief 已排序映射、设备缓存和候选工作区。@code Impl c(plan,budget); @endcode */
        struct DeviceAssemblyPlan::Impl {
            /** @brief 固定切片的稳定映射；不持有本次局部数值。@code Entry entry; @endcode */
            struct Entry {
                std::size_t first_{}, count_{}; ///< 模板切片键。
                bool matrix_{}; ///< 矩阵或向量类别。
                std::size_t bytes_{}; ///< 此条目的设备映射字节数。
                CudaBuffer<Index> targets_; ///< 设备目标槽。
                CudaBuffer<std::size_t> boundaries_, order_; ///< 设备段边界与稳定局部序。
            };
            std::shared_ptr<const AssemblyPlan> plan_; ///< 来源身份。
            std::size_t budget_, used_{}, builds_{}, hits_{}; ///< 映射预算、驻留与计数。
            int device_; ///< 固定设备。
            std::list<Entry> maps_; ///< 最近使用在前；每项只驻留设备，不重复保留主机数组。
            CudaBuffer<double> next_; ///< 有界高水位候选，超限仅本次兼容分配。
            CudaBuffer<int> status_; ///< 重用状态，每次重置。
            /** @brief 绑定来源。@param plan 来源。@param budget 缓存字节。
             * @pre 非空来源且设备可用。@return 无。@code Impl c(plan,budget); @endcode */
            Impl(std::shared_ptr<const AssemblyPlan> plan, std::size_t budget) :
                plan_(std::move(plan)), budget_(budget), status_(1) {
                if (!plan_) {
                    throw std::invalid_argument("组装计划为空");
                }
                device::check(cudaGetDevice(&device_));
            }
            /** @brief 创建一次固定映射。@param first 起点。@param count 数量。@param matrix 类别。
             * @pre 切片合法。@return 独立设备映射。@code auto m=build(first,count,true); @endcode */
            Entry build(std::size_t first, std::size_t count, bool matrix) {
                ++builds_;
                const auto& offsets = matrix ? plan_->matrixOffsets() : plan_->vectorOffsets();
                const auto start = offsets[first], length = offsets[first + count] - start;
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
                {
                    EMBERFIELD_NVTX_SCOPE("emberfield::assembly::AssemblyAccumulator::add/sortSlots");
                    std::stable_sort(order.begin(), order.end(),
                                     [&](std::size_t a, std::size_t b) { return slots[a] < slots[b]; });
                }
                std::vector<Index> targets;
                std::vector<std::size_t> boundaries;
                for (std::size_t i = 0; i < length; ++i) {
                    if (!i || slots[order[i]] != targets.back()) {
                        targets.push_back(slots[order[i]]);
                        boundaries.push_back(i);
                    }
                }
                boundaries.push_back(length);

                Entry result;
                result.first_ = first;
                result.count_ = count;
                result.matrix_ = matrix;
                const auto all =
                    emberfield::detail::checkedProduct(order.size() + boundaries.size(), sizeof(std::size_t));
                const auto targetBytes = emberfield::detail::checkedProduct(targets.size(), sizeof(Index));
                if (all > std::numeric_limits<std::size_t>::max() - targetBytes) {
                    throw std::overflow_error("映射字节计数溢出");
                }
                result.bytes_ = all + targetBytes;
                result.targets_ = CudaBuffer<Index>(targets);
                result.boundaries_ = CudaBuffer<std::size_t>(boundaries);
                result.order_ = CudaBuffer<std::size_t>(order);
                return result;
            }
        };
        DeviceAssemblyPlan::DeviceAssemblyPlan(std::shared_ptr<const AssemblyPlan> p, std::size_t budget) :
            impl_(std::make_unique<Impl>(std::move(p), budget)) {}
        DeviceAssemblyPlan::~DeviceAssemblyPlan() noexcept = default;
        const std::shared_ptr<const AssemblyPlan>& DeviceAssemblyPlan::source() const noexcept { return impl_->plan_; }
        std::size_t DeviceAssemblyPlan::mappingBuilds() const noexcept { return impl_->builds_; }
        std::size_t DeviceAssemblyPlan::cacheHits() const noexcept { return impl_->hits_; }
        void DeviceAssemblyPlan::accumulate(std::size_t first, std::size_t count, const CudaBuffer<double>& entries,
                                            bool matrix, double* output) {
            auto& s = *impl_;
            device::require(entries, s.device_);
            auto found = std::find_if(s.maps_.begin(), s.maps_.end(), [&](const Impl::Entry& e) {
                return e.first_ == first && e.count_ == count && e.matrix_ == matrix;
            });
            std::optional<Impl::Entry> transient;
            const Impl::Entry* map;
            if (found != s.maps_.end()) {
                ++s.hits_;
                s.maps_.splice(s.maps_.begin(), s.maps_, found);
                map = &s.maps_.front();
            }
            else {
                // 构建与执行均同步；只驱逐无人使用的已完成映射。
                auto candidate = s.build(first, count, matrix);
                if (candidate.bytes_ <= s.budget_) {
                    while (s.used_ > s.budget_ - candidate.bytes_) {
                        s.used_ -= s.maps_.back().bytes_;
                        s.maps_.pop_back();
                    }
                    s.maps_.push_front(std::move(candidate));
                    s.used_ += s.maps_.front().bytes_;
                    map = &s.maps_.front();
                }
                else {
                    transient.emplace(std::move(candidate));
                    map = &*transient;
                }
            }
            const auto n = map->targets_.size();
            if (!n) {
                return;
            }
            CudaBuffer<double> large;
            double* next;
            if (n <= 64ULL * 1024 * 1024 / sizeof(double)) {
                if (s.next_.size() < n) {
                    s.next_ = CudaBuffer<double>(n);
                }
                next = s.next_.deviceData();
            }
            else {
                large = CudaBuffer<double>(n);
                next = large.deviceData();
            } // 公开大切片沿用旧有界符号预算。
            device::check(cudaMemset(s.status_.deviceData(), 0, sizeof(int)));
            reduce<<<device::grid(n), 128>>>(n, map->targets_.deviceData(), map->boundaries_.deviceData(),
                                             map->order_.deviceData(), entries.deviceData(), output, next,
                                             s.status_.deviceData());
            device::check(cudaGetLastError());
            int bad{};
            device::check(cudaMemcpy(&bad, s.status_.deviceData(), sizeof(int), cudaMemcpyDeviceToHost));
            if (bad) {
                throw std::overflow_error("组装或线性系统出现不可表示的浮点结果");
            }
            commit<<<device::grid(n), 128>>>(n, map->targets_.deviceData(), next, output);
            device::check(cudaGetLastError());
            device::check(cudaStreamSynchronize(nullptr));
        }
    } // namespace detail
} // namespace emberfield::assembly
