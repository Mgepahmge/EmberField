/** @file Pvd.cuh
 * @brief 声明ParaView时间集合的独立文本编码。
 */
#pragma once
#include <filesystem>
#include <ostream>
#include <span>

namespace emberfield::io {
    /** @brief 一条完成的VTU时间引用，不拥有被引用文件。@code PvdEntry e{0.1,"fields/state_1.vtu"}; @endcode */
    struct PvdEntry {
        double time_{}; ///< 实际秒数。
        std::filesystem::path file_; ///< 相对PVD的安全路径。
    };
    /** @brief 编码完整PVD文档。@param stream 可写流。@param entries 已完成帧。
     * @pre 非空、时间严格递增且有限、路径相对安全；文件完成性由调用者保证。
     * @note 无返回值；不检查磁盘文件、不关闭流。
     * @throws std::invalid_argument 非法记录；ios_base::failure写失败。
     * @code io::writePvd(stream,entries); @endcode */
    void writePvd(std::ostream& stream, std::span<const PvdEntry> entries);
} // namespace emberfield::io
