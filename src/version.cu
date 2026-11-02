/**
 * @file version.cu
 * @brief 实现 EmberField 库版本查询接口。
 */

#include <version.cuh>

namespace emberfield {

    Version version() noexcept { return {0, 1, 0}; }

} // namespace emberfield
