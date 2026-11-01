/** @file mesh-roundtrip.cu
 * @brief 展示独立网格文件适配器的加载、保存及无求解器网格访问。
 */
#include <IO/Msh.cuh>
#include <exception>
#include <iostream>
#include <string_view>
/** @brief 重编码MSH网格，并输出节点与单元数量。
 * @param argc 参数数量。@param argv 输入、输出、ascii或binary、可选版本2.2或4.1。
 * @pre 输入输出路径不同，输出可覆盖。@return 成功0、参数错误2、运行失败1。
 * @code mesh-roundtrip input.msh output.msh binary 4.1 @endcode
 */
int main(int argc, char** argv) {
    using namespace thermofem::io;
    if (argc < 4 || argc > 5 || (std::string_view(argv[3]) != "ascii" && std::string_view(argv[3]) != "binary") ||
        (argc == 5 && std::string_view(argv[4]) != "2.2" && std::string_view(argv[4]) != "4.1")) {
        std::cerr << "Usage: mesh-roundtrip input.msh output.msh ascii|binary [4.1|2.2]\n";
        return 2;
    }
    try {
        const auto document = loadMsh(argv[1]);
        saveMsh(argv[2], document, std::string_view(argv[3]) == "binary" ? MshEncoding::Binary : MshEncoding::Ascii,
                argc == 5 && std::string_view(argv[4]) == "2.2" ? MshVersion::V22 : MshVersion::V41);
        std::size_t cells = 0;
        for (const auto& block : document.mesh_.data().elements_) {
            cells += block.tags_.size();
        }
        std::cout << "nodes=" << document.mesh_.data().nodes_.size() << " elements=" << cells << '\n';
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
