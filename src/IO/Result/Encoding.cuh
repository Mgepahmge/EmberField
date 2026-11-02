/** @file Encoding.cuh
 * @brief 文件编码模块内部的XML转义与流格式守卫。
 */
#pragma once
#include <DataStructure/Field/Grid.cuh>
#include <iomanip>
#include <limits>
#include <locale>
#include <ostream>

namespace emberfield::io::detail {
    /** @brief 恢复借用流的格式状态。@code StreamFormat guard(stream); @endcode */
    class StreamFormat {
    public:
        /** @brief 保存并设置经典locale。@param stream 借用流。@pre 流存活到析构。@note 无返回值。
         * @code StreamFormat g(stream); @endcode */
        explicit StreamFormat(std::ostream& stream) :
            stream_(stream), flags_(stream.flags()), precision_(stream.precision()), width_(stream.width()),
            locale_(stream.getloc()) {
            stream_.imbue(std::locale::classic());
            stream_.flags(std::ios::dec);
            stream_.width(0);
            stream_ << std::setprecision(std::numeric_limits<double>::max_digits10);
        }
        /** @brief 尽力恢复，无参数无前提，不抛异常。@note 无返回值。@code { StreamFormat g(s); } @endcode */
        ~StreamFormat() noexcept {
            try {
                stream_.flags(flags_);
                stream_.precision(precision_);
                stream_.width(width_);
                stream_.imbue(locale_);
            }
            catch (...) {
            }
        }
        StreamFormat(const StreamFormat&) = delete; ///< 不复制借用守卫。
        StreamFormat& operator=(const StreamFormat&) = delete; ///< 不复制借用守卫。
    private:
        std::ostream& stream_; ///< 借用流。
        std::ios::fmtflags flags_; ///< 原格式。
        std::streamsize precision_; ///< 原精度。
        std::streamsize width_; ///< 原字段宽度。
        std::locale locale_; ///< 原区域设置。
    };
    /** @brief XML属性转义。@param value UTF-8。@pre 无。@return 转义文本，非法字符抛invalid_argument。
     * @code auto x=xml("a&b"); @endcode */
    inline std::string xml(std::string_view value) {
        field::validateText(value);
        std::string result;
        for (char c : value) {
            switch (c) {
            case '&':
                result += "&amp;";
                break;
            case '<':
                result += "&lt;";
                break;
            case '>':
                result += "&gt;";
                break;
            case '"':
                result += "&quot;";
                break;
            case '\'':
                result += "&apos;";
                break;
            case '\r':
                result += "&#13;";
                break;
            case '\n':
                result += "&#10;";
                break;
            case '\t':
                result += "&#9;";
                break;
            default:
                result += c;
            }
        }
        return result;
    }
    /** @brief 检查流。@param stream 借用流。@pre 无。@note 无返回值，失败ios_base::failure。
     * @code checked(stream); @endcode */
    inline void checked(const std::ostream& stream) {
        if (!stream) {
            throw std::ios_base::failure("结果编码写入失败");
        }
    }
} // namespace emberfield::io::detail
