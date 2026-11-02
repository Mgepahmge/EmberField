/** @file Msh.cuh
 * @brief 定义与网格本体分离的MSH格式适配、限额和文件文档容器。
 */
#pragma once
#include <DataStructure/Mesh/Mesh.cuh>
#include <filesystem>
#include <iosfwd>
#include <map>

/** @brief 提供网格格式适配，不把编码或版本状态放入Mesh。 */
namespace thermofem::io {
    /** @brief MSH输出编码。 */
    enum class MshEncoding { Ascii, Binary };
    /** @brief 输出格式版本；旧版保留逐单元标签，4.1保留丰富实体元数据。 */
    enum class MshVersion { V22, V41 };
    /** @brief 文件读取的资源限额，可按算例规模显式提高。
     * @code MshLimits limits; limits.maxRecords_=1000000; @endcode */
    struct MshLimits {
        std::size_t maxRecords_{100000000}; ///< 单数组最大记录数。
        std::size_t maxBytes_{2147483648ULL}; ///< 整个输入字节上限，阻止无界输入。
    };
    /** @brief 保存未知ASCII扩展段，属于文件而不是拓扑。
     * @code MshSection s{"Comments","generated externally\n"}; @endcode */
    struct MshSection {
        std::string name_; ///< 不含美元符号的段名。
        std::string text_; ///< 不含开始及结束标记的ASCII内容。
    };
    /** @brief 文件文档组合拥有中立网格及兼容性信息。
     * @details Mesh可脱离本类单独构建使用。旧版逐单元标签保留在IO层，
     * 结果场段明确拒绝而不是误认为拓扑。未知ASCII段可保留；未知二进制段拒绝，
     * 因为没有类型信息时无法可靠发现二进制段边界。
     * @code auto document=readMsh(stream); auto& mesh=document.mesh_; @endcode */
    struct MshDocument {
        mesh::Mesh mesh_; ///< 经过验证的中立网格。
        std::string sourceVersion_{"4.1"}; ///< 输入格式版本，仅IO使用。
        std::map<mesh::Tag, std::vector<int>> legacyElementTags_; ///< MSH2逐单元原标签。
        std::vector<MshSection> extensions_; ///< 未解释的ASCII文件扩展。
    };
    /** @brief 解析MSH4.1 ASCII/二进制或MSH2.2 ASCII/二进制。
     * @param input 借用输入流。@param limits 资源限额。
     * @pre 输入满足支持的格式；二进制流不得经过文本换行转换。
     * @return 独立拥有的网格文档；非法或不支持输入抛runtime_error。
     * @code auto document=readMsh(input); @endcode */
    MshDocument readMsh(std::istream& input, MshLimits limits = {});
    /** @brief 将文档写为所选MSH版本，写入前验证可表示性，不静默降级。
     * @param output 借用输出流。@param document 网格及文件元数据。@param encoding 输出编码。
     * @param version 目标版本；默认4.1，2.2写出要求可无损保留旧版模型。
     * @pre 输出有效，网格可由目标版本无损表示；遗留标签不得包含不能表示的信息。
     * @note 无返回值；格式约束或流错误抛runtime_error；流写入错误可能已部分输出。
     * @code writeMsh(output,document,MshEncoding::Binary); @endcode */
    void writeMsh(std::ostream& output, const MshDocument& document, MshEncoding encoding = MshEncoding::Ascii,
                  MshVersion version = MshVersion::V41);
    /** @brief 从二进制方式打开的文件加载网格。
     * @param path 文件路径。@param limits 限额。@pre 可读取。
     * @return 独立文档；打开失败抛runtime_error。@code auto d=loadMsh("mesh.msh"); @endcode */
    MshDocument loadMsh(const std::filesystem::path& path, MshLimits limits = {});
    /** @brief 保存文件，格式验证完成后才打开目标文件。
     * @param path 输出路径。@param document 文档。@param encoding 编码。
     * @param version 目标版本，默认4.1。
     * @pre 可写；覆盖现有文件。@note 无返回值，失败抛runtime_error；磁盘写失败可能留下部分文件。
     * @code saveMsh("copy.msh",document); @endcode */
    void saveMsh(const std::filesystem::path& path, const MshDocument& document,
                 MshEncoding encoding = MshEncoding::Ascii, MshVersion version = MshVersion::V41);
    /** @brief 将Gmsh类型编号映射为中立参考单元描述。
     * @param code 文件单元类型。@pre 为具有固定节点连接的已登记类型。
     * @return 描述符；未知或未定义二进制布局的类型抛runtime_error。
     * @code auto type=mshElementType(9); @endcode */
    mesh::ElementType mshElementType(int code);
} // namespace thermofem::io
