// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Mingyu Wang, Ankang Xie, Song Wang, Chen Wang
/**
 * @file NVTXProfiler.cuh
 * @brief 可在编译期移除的RAII NVTX范围与快捷标签宏。
 * @details 改编自T-BLAEQ的NVTXProfiler，保留颜色与可选调用编号；修改为项目命名、
 * 线程局部编号、显式编译开关和禁止移动的栈范围。原许可证见third-party/NVTXProfiler/LICENSE。
 * 关闭开关时仅定义空宏，不包含NVTX头文件、不求值标签参数。
 */
#pragma once

#if defined(EMBERFIELD_ENABLE_NVTX)
#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <nvtx3/nvToolsExt.h>
#include <random>
#include <string>
#include <unordered_map>

/** @brief 跨模块的性能分析范围工具。 */
namespace emberfield {
    /** @brief 标准ARGB颜色，alpha通道固定为不透明。 */
    enum class NvtxColor : std::uint32_t {
        Red = 0xFFFF0000,
        Green = 0xFF00FF00,
        Blue = 0xFF0000FF,
        Cyan = 0xFF00FFFF,
        Magenta = 0xFFFF00FF,
        Yellow = 0xFFFFFF00,
        Orange = 0xFFFF8000,
        Purple = 0xFF8000FF,
        SpringGreen = 0xFF00FF80,
        LimeGreen = 0xFF80FF00,
        Rose = 0xFFFF0080,
        SkyBlue = 0xFF0080FF
    };

    /**
     * @brief 构造时压入、释放或析构时弹出一个当前线程NVTX范围。
     * @details 范围必须在同一线程按后进先出顺序结束，故禁止复制和移动。
     * 默认名称直接交给NVTX，不进行计数、字符串格式化或随机数初始化；countCalls=true
     * 时才按线程和名称计数，编号从第二次调用添加，消息最多保留255字节。
     * 不调用CUDA同步，范围表示主机提交区间，不等于GPU完成耗时。
     * @code
     * emberfield::NvtxProfiler range("assembly");
     * // 执行待分析操作。
     * range.release();
     * @endcode
     */
    class NvtxProfiler {
    public:
        /** @brief 无显式颜色、固定颜色或随机调色板。 */
        enum class ColorMode { None, Fixed, Random };

        /**
         * @brief 压入一个命名范围。
         * @param name 非空且以零结尾的标签；字符串只需在构造调用期间有效。
         * @param colorMode 颜色策略，默认不指定颜色。
         * @param fixedColor 固定ARGB颜色，仅Fixed时使用。
         * @param countCalls 是否启用按线程、按名称的调用编号，默认false。
         * @pre 当前线程按栈顺序管理所有活动范围，name有效，colorMode为合法枚举。
         * @note 无返回值。启用编号或随机颜色可能抛出分配或随机源异常；此时尚未压入范围。
         * @code NvtxProfiler range("solve", NvtxProfiler::ColorMode::Fixed, 0xFF00FF00); @endcode
         */
        explicit NvtxProfiler(const char* name, ColorMode colorMode = ColorMode::None,
                              std::uint32_t fixedColor = 0xFFFFFFFF, bool countCalls = false) {
            char numbered[256];
            if (countCalls) {
                auto& count = counters()[name];
                if (count != std::numeric_limits<std::uint64_t>::max()) {
                    ++count;
                }
                if (count > 1) {
                    std::snprintf(numbered, sizeof(numbered), "%s#%llu", name, static_cast<unsigned long long>(count));
                    name = numbered;
                }
            }
            nvtxEventAttributes_t attributes{};
            attributes.version = NVTX_VERSION;
            attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
            attributes.message.ascii = name;
            if (colorMode != ColorMode::None) {
                attributes.colorType = NVTX_COLOR_ARGB;
                attributes.color = colorMode == ColorMode::Random ? randomColor() : fixedColor;
            }
            nvtxRangePushEx(&attributes);
            active_ = true;
        }
        NvtxProfiler(const NvtxProfiler&) = delete; ///< 栈范围不可复制。
        NvtxProfiler& operator=(const NvtxProfiler&) = delete; ///< 栈范围不可复制赋值。
        NvtxProfiler(NvtxProfiler&&) = delete; ///< 禁止改变栈范围的生命周期顺序。
        NvtxProfiler& operator=(NvtxProfiler&&) = delete; ///< 禁止移动赋值。
        /** @brief 析构释放活动范围。无参数无返回值。
         * @pre 同构造线程，当前对象为最新活动范围。@code { NvtxProfiler range("step"); } @endcode */
        ~NvtxProfiler() noexcept { release(); }

        /** @brief 提前结束当前范围，重复调用无操作。
         * @pre 同构造线程，活动时必须为最新压入范围。无参数无返回值，不抛异常。
         * @code range.release(); @endcode */
        void release() noexcept {
            if (active_) {
                nvtxRangePop();
                active_ = false;
            }
        }
        /** @brief 查询本对象是否仍拥有一次待弹出的压栈。无参数无额外前提。
         * @return 活动状态；不表示外部分析工具是否已附加。@code auto active=range.isActive(); @endcode */
        bool isActive() const noexcept { return active_; }

    private:
        /** @brief 查询当前线程的名称计数表。无参数无额外前提。
         * @return 线程局部可变映射。@code auto& values=counters(); @endcode */
        static std::unordered_map<std::string, std::uint64_t>& counters() {
            thread_local std::unordered_map<std::string, std::uint64_t> values;
            return values;
        }
        /** @brief 随机选择预定义颜色。无参数无额外前提。
         * @return ARGB值；首次访问随机源可能抛异常。@code auto color=randomColor(); @endcode */
        static std::uint32_t randomColor() {
            thread_local std::mt19937 generator(std::random_device{}());
            thread_local std::uniform_int_distribution<std::size_t> distribution(0, palette_.size() - 1);
            return palette_[distribution(generator)];
        }
        bool active_{}; ///< 本对象是否尚未结束范围。
        static constexpr std::array<std::uint32_t, 12> palette_{
            static_cast<std::uint32_t>(NvtxColor::Red),
            static_cast<std::uint32_t>(NvtxColor::Green),
            static_cast<std::uint32_t>(NvtxColor::Blue),
            static_cast<std::uint32_t>(NvtxColor::Cyan),
            static_cast<std::uint32_t>(NvtxColor::Magenta),
            static_cast<std::uint32_t>(NvtxColor::Yellow),
            static_cast<std::uint32_t>(NvtxColor::Orange),
            static_cast<std::uint32_t>(NvtxColor::Purple),
            static_cast<std::uint32_t>(NvtxColor::SpringGreen),
            static_cast<std::uint32_t>(NvtxColor::LimeGreen),
            static_cast<std::uint32_t>(NvtxColor::Rose),
            static_cast<std::uint32_t>(NvtxColor::SkyBlue)}; ///< 标准ARGB调色板。
    };
} // namespace emberfield

/// @cond INTERNAL
#define EMBERFIELD_NVTX_JOIN_IMPL(left, right) left##right
#define EMBERFIELD_NVTX_JOIN(left, right) EMBERFIELD_NVTX_JOIN_IMPL(left, right)
/// @endcond
/** @brief 创建作用域标签；name为有效C字符串，关闭开关时参数不求值。
 * @param name 标签名称。@code EMBERFIELD_NVTX_SCOPE("assembly"); @endcode */
#define EMBERFIELD_NVTX_SCOPE(name)                                                                                    \
    [[maybe_unused]] ::emberfield::NvtxProfiler EMBERFIELD_NVTX_JOIN(emberfieldNvtxRange_, __COUNTER__)(name)
/** @brief 创建固定颜色作用域标签；关闭开关时全部参数不求值。
 * @param name 标签名称。@param color ARGB整数或NvtxColor。
 * @code EMBERFIELD_NVTX_SCOPE_COLOR("solve", emberfield::NvtxColor::Green); @endcode */
#define EMBERFIELD_NVTX_SCOPE_COLOR(name, color)                                                                       \
    [[maybe_unused]] ::emberfield::NvtxProfiler EMBERFIELD_NVTX_JOIN(emberfieldNvtxRange_, __COUNTER__)(               \
        name, ::emberfield::NvtxProfiler::ColorMode::Fixed, static_cast<std::uint32_t>(color))
/** @brief 创建随机调色板作用域标签；关闭开关时参数不求值。
 * @param name 标签名称。@code EMBERFIELD_NVTX_SCOPE_RANDOM("step"); @endcode */
#define EMBERFIELD_NVTX_SCOPE_RANDOM(name)                                                                             \
    [[maybe_unused]] ::emberfield::NvtxProfiler EMBERFIELD_NVTX_JOIN(emberfieldNvtxRange_, __COUNTER__)(               \
        name, ::emberfield::NvtxProfiler::ColorMode::Random)
#else
#define EMBERFIELD_NVTX_SCOPE(name)
#define EMBERFIELD_NVTX_SCOPE_COLOR(name, color)
#define EMBERFIELD_NVTX_SCOPE_RANDOM(name)
#endif
