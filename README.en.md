# EmberField

[中文](README.md) | English

EmberField is the code project for an undergraduate thesis on the design and implementation of a finite-element thermal simulation solver. It is developed with CUDA and C++20.

## Goals

- Model steady-state and transient heat conduction with meshes, material properties, heat sources, and boundary conditions.
- Implement finite-element discretization, element matrices, GPU-accelerated global sparse assembly, and boundary-condition handling.
- Solve sparse linear systems and advance transient temperature fields in time.
- Provide case configuration, solver execution, and result export interfaces, with representative accuracy and performance tests.

## Project Layout

~~~text
EmberField/
├── include/          # Public headers
├── src/              # CUDA implementation and executable entry point
├── test/             # GoogleTest tests
├── example/          # Example programs and cases
├── third-party/      # Vendored dependencies and licenses
├── scripts/          # Project scripts
├── docs/             # Design documents, user guide, and Doxygen configuration
├── cmake/            # CMake package configuration template
├── CMakeLists.txt
├── README.md
└── LICENSE
~~~

## Requirements

| Dependency | Version and compatibility | Purpose and availability | Notes |
| --- | --- | --- | --- |
| CUDA Toolkit and NVCC | **12.0 or later** | Required to build; use the same Toolkit for NVCC, CUDA Runtime, cuBLAS, and cuSPARSE | NVCC 12.0 is the first release supporting C++20. For other NVCC versions, check their supported host compilers and driver requirements. |
| GCC/G++ (Linux) | With NVCC 12.0: **GCC/G++ 10 through 12.1** | Required to build; use matching GCC and G++ versions | GCC 10 or later is needed for C++20. GCC 12.2.1 support is experimental; use an officially supported version. |
| MSVC (Windows) | With NVCC 12.0: **Visual Studio 2022 / MSVC 19.3x** | Required to build; install the matching Visual Studio C++ tools and Windows SDK | Visual Studio 2022 17.4 is recommended because it fixes earlier C++20 complex and bit-cast compilation issues. |
| CMake | **3.24 or later** | Required to configure, build, and install | Minimum specified by this project. |
| NVIDIA GPU and driver | For CUDA 12.0 GA: at least **Linux x86_64 525.60.13 / Windows x86_64 527.41** | Required to execute GPU computations and GPU tests | The GPU architecture must be supported by the chosen Toolkit. These driver baselines apply only to CUDA 12.0 GA. |
| GoogleTest | **1.17.0** recommended; v1.17.0 fetched by default | Needed only when tests are enabled; use an installed CMake package, Git download, or local source | Not needed when tests are disabled. |
| Doxygen | **1.9.1 or later** recommended | Needed only to build documentation locally | No mandatory minimum; Graphviz and LaTeX are not required. |
| Bash | A Bash implementation capable of running project scripts | Needed for the documentation build script | The CMake docs target is an alternative. |
| picojson | Vendored source version | JSON parser in third-party/picojson/ | No separate installation. |
| CLI11 | **2.6.1**, vendored source | Command-line parser in third-party/CLI11/ | No separate installation. |

## Build and Run

Run these Bash commands from the project root:

~~~bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./build/emberfield example/cases/steady.json
ctest --test-dir build --output-on-failure
~~~

The default CUDA architecture is 75. Set it to match your GPU and Toolkit; for example, an RTX A6000 uses 86:

~~~bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=86
~~~

Tests and examples are enabled by default. Disable them with -DBUILD_TESTING=OFF and -DEMBERFIELD_BUILD_EXAMPLES=OFF, respectively. The first automatic GoogleTest download needs Git and access to GitHub. For an offline build, provide a local source tree:

~~~bash
cmake -S . -B build \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest
~~~

The command `emberfield <case.json> [--output-root path] [--output folder]` loads and solves a case, then creates a result directory. By default, `output-root` is the executable's directory and `output` is the case file's stem. Together, these options override `output.directory` in the case configuration. See the [Chinese User Guide](docs/user-guide.md) for the complete input and output contract.

## Install

~~~bash
cmake --install build --prefix "$PWD/install"
~~~

This installs the executable, static library, public headers, license, and CMake package configuration. Other CUDA/C++20 projects can link against it:

~~~cmake
find_package(EmberField 0.1 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE EmberField::emberfield)
~~~

Pass -DCMAKE_PREFIX_PATH=/path/to/EmberField/install when configuring a consumer.

## Documentation

Read the [published EmberField documentation](https://mgepahmge.github.io/EmberField-Doc/) or start from the [documentation index](docs/index.md) in this repository.

After installing Doxygen, run from the project root:

~~~bash
bash scripts/build-docs.sh
~~~

The generated HTML entry point is build/docs/html/index.html. If Doxygen was found when configuring CMake, you can also run:

~~~bash
cmake --build build --target docs
~~~

## Contributions

This is an individual undergraduate thesis project. External contributions, including code and documentation, are generally not accepted; please do not open pull requests.

## Algorithm Reference

The organization of the Jacobi, AMG, and BiCGStab algorithms draws on **YHAMG-GPU**, a non-public project by Mgepahmge. It is not distributed as a public dependency.

## License

EmberField uses the [MIT License](LICENSE). Vendored dependencies retain their respective licenses.
