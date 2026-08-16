"""程序入口：python main.py"""

import argparse

from satc.threads import configure_blas_threads

# 必须在导入 numpy / sklearn 之前调用，避免 Windows BLAS 线程阻塞
configure_blas_threads()

from satc.pipeline import run_pipeline  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="SATC-NSGA-II 代理辅助热约束多目标优化"
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=3,
        metavar=("WT", "WB", "WS"),
        default=None,
        help="三个目标的权重（ΔT ΔB ΔS），如 0.3 0.3 0.4；默认等权",
    )
    parser.add_argument(
        "--auto-weights",
        action="store_true",
        help="根据 ANSYS 力学结果自动设置权重（需 --node-file）",
    )
    parser.add_argument(
        "--node-file", default=None,
        help="ANSYS 节点坐标文件（点云，自动权重用）",
    )
    parser.add_argument(
        "--stress-folder", default=None,
        help="六个应力分量文件夹（自动权重用，可选）",
    )
    parser.add_argument(
        "--deform-file", default=None,
        help="变形结果文件（自动权重用，可选）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weights = args.weights
    if args.auto_weights:
        if not args.node_file:
            raise SystemExit("--auto-weights 需要提供 --node-file。")
        from satc.mechanics import suggest_weights

        info = suggest_weights(
            args.node_file,
            stress_folder=args.stress_folder,
            deformation_path=args.deform_file,
        )
        weights = info["weights"]
        print(info["explanation"])
        print()

    result = run_pipeline(weights=weights)
    print()
    print("=" * 60)
    print("推荐参数：", result["best_x"])
    print("推荐目标：", result["best_f"])
    print(f"综合评分（百分制，100 分最好）：{result['score']:.2f}")
    print(f"目标权重：ΔT={result['weights'][0]:.3f}, "
          f"ΔB={result['weights'][1]:.3f}, ΔS={result['weights'][2]:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
