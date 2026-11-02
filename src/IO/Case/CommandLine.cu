/** @file CommandLine.cu @brief 真实进程路径与单级结果目录校验。 */
#include <DataStructure/Field/Grid.cuh>
#include <IO/Case/CommandLine.cuh>
#include <stdexcept>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif
namespace emberfield::io {
    std::filesystem::path executableDirectory() {
#ifdef _WIN32
        std::wstring value(32768, L'\0');
        const auto n = GetModuleFileNameW(nullptr, value.data(), static_cast<DWORD>(value.size()));
        if (!n || n >= value.size()) {
            throw std::runtime_error("无法获取可执行文件路径");
        }
        value.resize(n);
        return std::filesystem::path(value).parent_path();
#else
        std::string value(4096, '\0');
        for (;;) {
            const auto n = readlink("/proc/self/exe", value.data(), value.size());
            if (n < 0) {
                throw std::runtime_error("无法获取可执行文件路径");
            }
            if (static_cast<std::size_t>(n) < value.size()) {
                value.resize(n);
                return std::filesystem::path(value).parent_path();
            }
            if (value.size() > 1024 * 1024) {
                throw std::length_error("进程路径过长");
            }
            value.resize(value.size() * 2);
        }
#endif
    }
    std::filesystem::path outputDirectory(const std::filesystem::path& input, const std::filesystem::path& root,
                                          const std::optional<std::string>& name) {
        const auto folder = name.value_or(input.stem().string());
        field::validateText(folder);
        if (folder.empty() || folder == "." || folder == ".." ||
            folder.find_first_of("/\\:<>\"|?*") != std::string::npos || folder.back() == '.' || folder.back() == ' ') {
            throw std::invalid_argument("output必须是单个安全目录名");
        }
        for (unsigned char c : folder) {
            if (c < 32) {
                throw std::invalid_argument("output包含控制字符");
            }
        }
        std::string upper = folder.substr(0, folder.find('.'));
        for (auto& c : upper) {
            if (c >= 'a' && c <= 'z') {
                c -= 32;
            }
        }
        if (upper == "CON" || upper == "PRN" || upper == "AUX" || upper == "NUL" ||
            (upper.size() == 4 && (upper.starts_with("COM") || upper.starts_with("LPT")) && upper[3] >= '1' &&
             upper[3] <= '9')) {
            throw std::invalid_argument("output是系统保留名称");
        }
        if (root.empty()) {
            throw std::invalid_argument("output-root不能为空");
        }
        return std::filesystem::absolute(root / folder).lexically_normal();
    }
} // namespace emberfield::io
