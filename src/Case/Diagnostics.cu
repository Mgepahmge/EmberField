/** @file Diagnostics.cu
 * @brief 实现限长诊断与不丢失错误状态的合并。
 */
#include <Case/Diagnostics.cuh>
#include <stdexcept>
#include <utility>
namespace thermofem::problem {
    Diagnostics::Diagnostics(std::size_t limit) : limit_(limit) {
        if (limit == 0) {
            throw std::invalid_argument("诊断限额必须大于0");
        }
    }
    void Diagnostics::add(Diagnostic value) {
        hasErrors_ = hasErrors_ || value.severity_ == Severity::Error;
        if (entries_.size() < limit_) {
            entries_.push_back(std::move(value));
        }
        else {
            ++omitted_;
        }
    }
    void Diagnostics::append(const Diagnostics& other) {
        if (this == &other) {
            const auto copy = other;
            append(copy);
            return;
        }
        hasErrors_ = hasErrors_ || other.hasErrors_;
        for (const auto& value : other.entries_) {
            add(value);
        }
        omitted_ += other.omitted_;
    }
} // namespace thermofem::problem
