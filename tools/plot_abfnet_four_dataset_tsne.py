#!/usr/bin/env python3
"""
四套主实验测试集上做 t-SNE（bridge 全局平均池化特征）。
图题与混淆矩阵 / 识别率曲线一致：t-SNE (<数据集名>)

默认任务表与 plot_baiyr_four_dataset_curves.py 相同。
"""

from __future__ import annotations

import argparse
import os
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_TOOLS_DIR)
for _p in (_PROJECT, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_postprocess import read_config  # noqa: E402
from visualize_tsne import generate_tsne  # noqa: E402

DEFAULT_JOBS: list[tuple[str, str, str]] = [
    ("configs/abfnet_rafdb.yaml", "1.1", "RAF-DB"),
    ("configs/abfnet_ferplus.yaml", "1.2", "FER+"),
    ("configs/abfnet_affectnet7.yaml", "1.4", "AffectNet-7"),
    ("configs/abfnet_affectnet8.yaml", "1.5", "AffectNet-8"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--max-samples", type=int, default=3000)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="缺少 checkpoint 时跳过该套",
    )
    args = ap.parse_args()

    for rel_cfg, exp_dir, dataset_title in DEFAULT_JOBS:
        cfg_path = os.path.join(_PROJECT, rel_cfg)
        config = read_config(cfg_path)
        ckpt_dir = os.path.join(
            config["SYSTEM"]["SAVE_DIR"], exp_dir, "checkpoint", f"{args.checkpoint}.pth"
        )
        if not os.path.isfile(ckpt_dir):
            msg = f"[skip] no checkpoint: {ckpt_dir}"
            if args.skip_missing:
                print(msg, file=sys.stderr)
                continue
            raise FileNotFoundError(msg)

        png, csv = generate_tsne(
            config,
            exp_name_override=exp_dir,
            dataset_title=dataset_title,
            checkpoint=args.checkpoint,
            max_samples=args.max_samples,
            perplexity=args.perplexity,
            seed=args.seed,
            dpi=args.dpi,
        )
        print(f"OK {dataset_title}: {png}")
        print(f"    CSV: {csv}")


if __name__ == "__main__":
    main()
