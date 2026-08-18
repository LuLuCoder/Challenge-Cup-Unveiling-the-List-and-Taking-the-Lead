"""第五章配图一键生成入口。

用法：
    cd figures
    python make_figures.py            # 输出到 figures/output/
    python make_figures.py --path     # 仅生成 5.2 路径规划图
    python make_figures.py --satc     # 仅生成 5.3 参数优化图
    set FIG_OUT=某个目录 && python make_figures.py   # 自定义输出目录

说明：
    - 图5-10（主界面/检查窗口截图）与图5-18（参数优化界面截图）
      需人工截取，本脚本不生成；
    - 运行需 pytorch 环境（已装 numpy/pandas/matplotlib/scipy/gmsh）。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action="store_true", help="仅生成 5.2 路径规划图")
    parser.add_argument("--satc", action="store_true", help="仅生成 5.3 参数优化图")
    args = parser.parse_args()

    if not args.path and not args.satc:
        args.path = args.satc = True

    if args.path:
        import figs_path

        figs_path.run_all()
    if args.satc:
        import figs_satc

        figs_satc.run_all()

    print("\n全部完成。输出目录：",
          __import__("common").output_dir())


if __name__ == "__main__":
    main()
