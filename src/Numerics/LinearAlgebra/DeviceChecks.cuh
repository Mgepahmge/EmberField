/** @file DeviceChecks.cuh
 * @brief 共享组装与系统运算的设备前提和候选数值状态检查。
 */
#pragma once
#include <DataStructure/Storage/CudaBuffer.cuh>
#include <algorithm>
#include <array>
#include <stdexcept>

/** @brief 提供内部CUDA错误和设备契约辅助。 */
namespace emberfield::device {
    /** @brief 检查CUDA调用。@param status 返回码。@pre 无。@note 无返回，失败CudaError。
     * @code check(cudaGetLastError()); @endcode */
    inline void check(cudaError_t status) {
        if (status != cudaSuccess) {
            throw CudaError(status, "组装或线性系统CUDA操作");
        }
    }
    /** @brief 验证global缓冲区设备。@tparam T 缓冲元素。@param b 输入。@param expected 预期设备。
     * @pre 无。@note 无返回，策略或设备错误invalid_argument。@code require(b,device); @endcode */
    template <BufferElement T>
    void require(const CudaBuffer<T>& b, int expected) {
        int current = -1;
        check(cudaGetDevice(&current));
        if (b.memoryType() != MemoryType::Global || current != expected || (!b.empty() && b.deviceId() != expected)) {
            throw std::invalid_argument("操作要求当前设备上的global缓冲区");
        }
    }
    /** @brief 同步检查候选运算状态；状态非零时拒绝发布。
     * @code ErrorFlag status; kernel<<<1,1>>>(status.data()); status.verify(); @endcode */
    class ErrorFlag {
    public:
        /** @brief 创建零状态；无参数无前提。@note 无返回。@code ErrorFlag f; @endcode */
        ErrorFlag() : value_(1) { check(cudaMemset(value_.deviceData(), 0, sizeof(int))); }
        /** @brief 获取设备状态地址；无参数无前提。@return 借用指针。@code auto p=f.data(); @endcode */
        int* data() { return value_.deviceData(); }
        /** @brief 检查启动与数值状态；无参数无前提。@note 无返回，数值失败overflow_error。
         * @code f.verify(); @endcode */
        void verify() const {
            check(cudaGetLastError());
            std::array<int, 1> host{};
            value_.copyToHost(host);
            if (host[0]) {
                throw std::overflow_error("组装或线性系统出现不可表示的浮点结果");
            }
        }

    private:
        CudaBuffer<int> value_; ///< 拥有单个设备错误标志。
    };
    /** @brief 计算有界网格。@param n 非零工作量。@pre n>0。@return 128线程块数。
     * @code kernel<<<grid(n),128>>>(...); @endcode */
    inline unsigned grid(std::size_t n) {
        return static_cast<unsigned>(std::min<std::size_t>((n - 1) / 128 + 1, 65535));
    }
} // namespace emberfield::device
