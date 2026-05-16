#!/usr/bin/env python3
"""
从四套主实验的 metrics.txt 生成识别率 / 损失曲线。
图题格式与 tools/run_postprocess.py 中混淆矩阵一致：
  Confusion Matrix (<dataset>)
  Recognition Rate Curve (<dataset>)
  Loss Curve (<dataset>)
"""

from __future__ import annotations

import argparse
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from run_postprocess import (  # noqa: E402
    load_metrics,
    plot_curves,
    prepare_paths,
    read_config,
    set_paper_style,
)

_PROJECT = os.path.dirname(_TOOLS_DIR)

# (config 相对 tools/.., 结果子目录名, 图题括号内数据集名 — 与 --confusion-title 一致)
DEFAULT_JOBS: list[tuple[str, str, str]] = [
    ("configs/abfnet_rafdb.yaml", "1.1", "RAF-DB"),
    ("configs/abfnet_ferplus.yaml", "1.2", "FER+"),
    ("configs/abfnet_affectnet7.yaml", "1.4", "AffectNet-7"),
    # train_affectnet_8cls.yaml 里 EXPERIMENT_NAME 可能为 affectnet_8cls，但历史跑数常在 results/1.5
    ("configs/abfnet_affectnet8.yaml", "1.5", "AffectNet-8"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="若某套 metrics.txt 不存在则跳过，不中断其它套",
    )
    args = ap.parse_args()

    set_paper_style()

    for rel_cfg, exp_dir, dataset_title in DEFAULT_JOBS:
        cfg_path = os.path.join(_PROJECT, rel_cfg)
        config = read_config(cfg_path)
        paths = prepare_paths(config, exp_dir)
        mpath = paths["metrics"]
        if not os.path.isfile(mpath):
            msg = f"[skip] no metrics: {mpath}"
            if args.skip_missing:
                print(msg, file=sys.stderr)
                continue
            raise FileNotFoundError(msg)
        df = load_metrics(mpath)
        plot_curves(df, paths["out"], dataset_title, args.dpi)
        print(f"OK {dataset_title}: {paths['out']}/curve_accuracy.png")


if __name__ == "__main__":
    main()
