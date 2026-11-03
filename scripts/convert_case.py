#!/usr/bin/env python3
"""将受支持的Elmer SIF或CalculiX/Abaqus INP纯导热子集转为EmberField算例。"""
import argparse
from pathlib import Path
import sys

from caseconvert import elmer, inp
from caseconvert.common import ConversionError, number, writeBundle


def main(argv=None):
    """解析离线转换参数并事务发布算例目录；成功0、输入/能力错误2。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("format", choices=("elmer", "inp"), help="源格式；inp为受支持的CalculiX/Abaqus关键字子集")
    parser.add_argument("input", type=Path, help="源.sif或.inp文件")
    parser.add_argument("--output", type=Path, required=True, help="新建输出目录，必须不存在")
    parser.add_argument("--temperature-unit", dest="temperatureUnit", choices=("K", "C"), required=True, help="显式声明源温度单位；其他物理量与坐标必须是SI")
    parser.add_argument("--mesh-dir", dest="meshDir", type=Path, help="Elmer原生mesh.*目录；默认从SIF Mesh DB读取")
    parser.add_argument("--thickness", type=number, help="Elmer二维模型统一厚度[m]，必须显式提供")
    parser.add_argument("--initial-temperature", dest="initialTemperature", type=number, help="显式提供缺省初温，单位与--temperature-unit一致")
    parser.add_argument("--backward-euler-step", dest="backwardEulerStep", type=number,
                        help="显式同意瞬态重离散为后向Euler，并给定时间步[s]；记录于转换报告")
    args = parser.parse_args(argv)
    try:
        if args.format == "inp" and (args.meshDir is not None or args.thickness is not None):
            raise ConversionError("INP网格与厚度由文件提供，不接受--mesh-dir或--thickness")
        model = (elmer if args.format == "elmer" else inp).convert(args.input.resolve(), args)
        directory = writeBundle(model, args.output, args.format, args.input.stem)
        print(f"已转换: {directory / 'case.json'}")
        print(f"节点={len(model.nodes)} 单元={len(model.cells)}；转换说明见conversion-report.json")
        return 0
    except (ValueError, OSError, UnicodeError) as error:
        print(f"转换失败: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
