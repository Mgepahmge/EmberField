# NVTX 性能分析标签

使用T-BLAEQ的NVTXProfiler适配实现，为Nsight Systems提供RAII主机范围。头文件为NVTXProfiler.cuh；来源、许可证与修改说明见third-party/NVTXProfiler/NOTICE。

## 构建开关

~~~bash
cmake -S . -B build-nvtx -DCMAKE_BUILD_TYPE=Release -DEMBERFIELD_ENABLE_NVTX=ON
cmake --build build-nvtx --parallel
~~~

EMBERFIELD_ENABLE_NVTX默认OFF。ON时CMake通过EmberField::emberfield的PUBLIC使用要求定义同名预处理宏；构建树及安装包的下游目标自动继承。关闭时不定义该宏，快捷宏展开为空，包括名称、颜色表达式均不求值，头文件不包含NVTX或其辅助标准库。修改开关后应重新构建所有使用相关头文件的目标，不在单个源文件中手动混用定义。

启用需要CUDA Toolkit的nvtx3/nvToolsExt.h。NVTX3为头文件接口；Unix平台由CMake传递动态加载支持库，不依赖已移除的旧libnvToolsExt。保持最低CMake 3.24要求，不依赖较新版本才提供的CUDA::nvtx3目标。

## 作用域接口

~~~cpp
#include <NVTXProfiler.cuh>

void assemble() {
    EMBERFIELD_NVTX_SCOPE("assembly");
    {
        EMBERFIELD_NVTX_SCOPE_COLOR("volume", emberfield::NvtxColor::Green);
        // 体积分与组装。
    }
    {
        EMBERFIELD_NVTX_SCOPE_RANDOM("boundary");
        // 边界积分。
    }
}
~~~

宏创建自动局部对象，退出当前C++作用域时弹出标签，异常路径同样释放。需要更短范围时使用花括号；不要把宏作为无花括号if/for的单条语句使用。局部变量名通过两层宏展开与__COUNTER__去重，同一行和同一作用域的多个标签均可使用。

启用时也可直接构造emberfield::NvtxProfiler并显式release()；重复release()无操作。isActive()表示对象是否尚未弹栈，与分析器是否附加无关。直接类型仅在开关开启时声明，跨配置代码使用快捷宏。

默认标签使用固定名称，无调用计数、字符串格式化、随机数初始化。固定色宏同样不启用编号；随机色宏会首次初始化线程局部随机源。确需编号可使用NvtxProfiler(name, mode, color, true)，计数按线程及完整名称保存，第二次开始追加#2等后缀；编号后的消息最多255字节，重复动态名称会增加线程局部映射存储，不建议用于细粒度热点。固定名称不截断。

NVTX push/pop属于当前线程的栈范围，必须同线程按后进先出顺序结束，因此包装类禁止复制和移动。宏内不做CUDA同步；范围覆盖的是主机提交区间，分析GPU执行应结合Nsight的CUDA流和内核时间线，不能直接将范围宽度解释为GPU完成耗时。

## 采集示例

~~~bash
nsys profile --trace=cuda,nvtx --sample=none -o case-profile \
  build-nvtx/emberfield example/cases/time-varying.json \
  --output-root /tmp --output emberfield-profile-result
~~~

结果目录须未被已有运行占用。当前runCase带有完整运行范围emberfield::problem::runCase；后续可在待调查的独立运算中插入嵌套标签。NVTX标签不会自动生成各模块内部层次，采样和采集开销也不包含在“关闭时空宏”的保证内。
