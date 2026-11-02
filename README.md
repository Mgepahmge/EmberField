# EmberField

EmberField 是本科毕业设计“基于有限元的热力学仿真求解器设计与实现”的代码项目，采用 CUDA 和 C++20 开发。

## 项目目标

- 建立稳态与瞬态热传导模型，描述网格、材料属性、热源及边界条件。
- 实现有限元离散、单元矩阵计算、GPU 全局稀疏组装及边界条件处理。
- 实现稀疏线性方程组求解与时间步进，计算稳态和瞬态温度场。
- 提供算例配置、求解及结果导出接口，通过典型算例验证精度与性能。

## 项目结构

```text
EmberField/
├── include/          # 公共头文件
├── src/              # CUDA 实现与程序入口
├── test/             # GoogleTest 测试
├── example/          # 示例程序与算例
├── third-party/      # 第三方依赖及许可证
├── scripts/          # 项目脚本
├── docs/             # 设计文档、用户手册及 Doxygen 配置
├── cmake/            # CMake 包配置模板
├── CMakeLists.txt
├── README.md
└── LICENSE
```

## 环境要求

| 依赖 | 版本与兼容性要求 | 用途与获取方式 | 说明 |
| --- | --- | --- | --- |
| CUDA Toolkit 与 NVCC | **12.0 或更新版本** | 构建必需；使用同一套 Toolkit 提供的 NVCC、CUDA Runtime、cuBLAS 和 cuSPARSE | 12.0 是 NVCC 最早支持 C++20 的版本；使用其他版本 NVCC 时，须自行查询对应版本的 HOST 编译器支持范围和驱动要求 |
| GCC/G++（Linux） | NVCC 12.0：**GCC/G++ 10 至 12.1** | 构建必需；GCC 与 G++ 使用匹配版本 | C++20 要求 GCC 10 起；12.2.1 为实验支持，建议选择正式支持的版本 |
| MSVC（Windows） | NVCC 12.0：**Visual Studio 2022 / MSVC 19.3x** | 构建必需；安装对应的 Visual Studio C++ 工具集与 Windows SDK | 建议 VS 2022 17.4，修复了早期版本的 C++20 `<complex>` / `std::_Bit_cast` 编译问题 |
| CMake | **3.24 或更新版本** | 配置、构建与安装必需 | 项目配置规定的最低版本 |
| NVIDIA GPU 与驱动 | CUDA 12.0 GA：最低 **Linux x86_64 525.60.13 / Windows x86_64 527.41** | 运行 GPU 计算及 GPU 测试时必需 | GPU 架构须受所选 Toolkit 支持；以上驱动基线仅针对 CUDA 12.0 GA，其他版本须另行核对 |
| GoogleTest | 推荐 **1.17.0**；默认下载 `v1.17.0` | 启用测试时需要；优先使用已安装的 CMake 包，否则通过 Git 获取，也可指定离线源码 | 不构建测试时无需安装 |
| Doxygen | 推荐 **1.9.1 或更新版本** | 仅自行生成文档时需要 | 项目未强制最低版本；不需要 Graphviz 或 LaTeX |
| Bash | 可执行项目中的 Bash 脚本 | 使用文档构建脚本时需要 | 也可通过 CMake 的 `docs` 目标生成文档 |
| picojson | 使用源码内附版本 | JSON 解析；位于 `third-party/picojson/` | 随项目发布，无需另行安装 |

## 构建与运行

以下命令在项目根目录执行，使用 Bash 语法：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/emberfield
ctest --test-dir build --output-on-failure
```

默认目标 GPU 架构为 `75`。请按实际 GPU 和 Toolkit 支持范围设置，例如 RTX A6000 使用 `86`：

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86
```

测试与示例默认启用，可分别通过 `-DBUILD_TESTING=OFF` 和 `-DEMBERFIELD_BUILD_EXAMPLES=OFF` 关闭。首次自动获取 GoogleTest 需要 Git 和 GitHub 网络访问；离线构建时指定已有源码：

```bash
cmake -S . -B build \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest
```

`emberfield` 当前输出库版本；完整算例的运行方法与示例见用户文档。

## 安装

```bash
cmake --install build --prefix "$PWD/install"
```

安装主程序、静态库、公共头文件、许可证及 CMake 包配置。其他 CUDA/C++20 项目可通过安装前缀查找并链接：

```cmake
find_package(EmberField 0.1 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE EmberField::emberfield)
```

配置使用方时传入 `-DCMAKE_PREFIX_PATH=/path/to/EmberField/install`。

## 文档

在线阅读：[EmberField 文档](https://mgepahmge.github.io/EmberField-Doc/)。源码中的文档入口见 [文档目录](docs/index.md)。

安装 Doxygen 后，在项目根目录执行：

```bash
bash scripts/build-docs.sh
```

生成的 HTML 入口为 `build/docs/html/index.html`。如果配置 CMake 时已找到 Doxygen，也可执行：

```bash
cmake --build build --target docs
```

## 贡献

本项目用于个人本科毕业设计，原则上不接受任何外部贡献，包括代码、文档或其他形式的提交；请勿提交 Pull Request。

## 算法参考

Jacobi、AMG 与 BiCGStab 部分的算法组织参考作者 Mgepahmge 的非公开项目 **YHAMG-GPU**。该参考项目不作为公开依赖分发。

## 许可证

本项目采用 [MIT License](LICENSE)。第三方依赖遵循各自的许可证。
