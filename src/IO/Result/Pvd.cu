/** @file Pvd.cu
 * @brief 编码仅引用完整相对路径的ParaView时间索引。
 */
#include <IO/Result/Pvd.cuh>
#include <NVTXProfiler.cuh>
#include <cmath>
#include <set>
#include <stdexcept>
#include "Encoding.cuh"

namespace emberfield::io {
    void writePvd(std::ostream& stream, std::span<const PvdEntry> entries) {
        EMBERFIELD_NVTX_SCOPE_COLOR("emberfield::io::writePvd", emberfield::NvtxColor::Orange);
        if (entries.empty()) {
            throw std::invalid_argument("PVD集合不能为空");
        }
        std::set<std::filesystem::path> files;
        std::optional<double> previous;
        for (const auto& e : entries) {
            if (!std::isfinite(e.time_) || (previous && e.time_ <= *previous) || e.file_.empty() ||
                e.file_.is_absolute() || e.file_.has_root_name() || !files.insert(e.file_).second) {
                throw std::invalid_argument("PVD时间或路径非法");
            }
            const auto path = e.file_.generic_string();
            if (path.find('\\') != std::string::npos || path.find(':') != std::string::npos) {
                throw std::invalid_argument("PVD路径不跨平台安全");
            }
            for (const auto& part : e.file_) {
                if (part == ".." || part == ".") {
                    throw std::invalid_argument("PVD路径越界");
                }
            }
            field::validateText(path);
            previous = e.time_;
        }
        detail::StreamFormat guard(stream);
        stream << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
                  "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n  <Collection>\n";
        for (const auto& e : entries) {
            stream << "    <DataSet timestep=\"" << e.time_ << "\" group=\"\" part=\"0\" file=\""
                   << detail::xml(e.file_.generic_string()) << "\"/>\n";
        }
        stream << "  </Collection>\n</VTKFile>\n";
        detail::checked(stream);
    }
} // namespace emberfield::io
