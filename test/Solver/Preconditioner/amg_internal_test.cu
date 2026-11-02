/** @file amg_internal_test.cu
 * @brief 在 AMG 自身编译单元内验证内部阶段，保持生产接口封装。
 * @details 独立测试程序直接编译模块实现，内部阶段与主机基准逐项对照。
 * 静态库中的 AMG 编译单元不会被重复链接；其它库组件仍通过导出目标链接。
 */
#include "../../../src/Solver/Preconditioner/Amg.cu"
#include "amg_native_test.cpp"
#include "multigrid_operations_test.cpp"
