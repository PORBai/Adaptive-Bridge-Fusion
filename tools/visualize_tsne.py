"""从 Baiyr bridge 特征做 t-SNE 可视化（测试集 ImageFolder）。"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import sys
from typing import Optional, Tuple

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TOOLS_DIR)
for _p in (_PROJECT_ROOT, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib.pyplot as plt
import torch
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets

from datasets.predata import build_test_transform
from models.build import build_model
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Generate t-SNE from Baiyr bridge features.")
    parser.add_argument(
        "--config",
        default=os.path.join(_PROJECT_ROOT, "configs/abfnet_rafdb.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint path or keyword: best/latest.",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        help="覆盖 SYSTEM.EXPERIMENT_NAME（与 results 子目录一致，如 1.1）。",
    )
    parser.add_argument(
        "--dataset-title",
        default=None,
        help="图题括号内名称，如 RAF-DB；默认与实验目录名一致。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3000,
        help="Max number of samples for t-SNE.",
    )
    parser.add_argument(
        "--perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity（会自动降到小于样本数）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="导出分辨率（与论文曲线脚本一致时可设 300）。",
    )
    return parser.parse_args()


def resolve_checkpoint_path(config, checkpoint_arg):
    experiment_name = str(config["SYSTEM"]["EXPERIMENT_NAME"])
    checkpoint_dir = os.path.join(config["SYSTEM"]["SAVE_DIR"], experiment_name, "checkpoint")
    if checkpoint_arg in {"best", "latest"}:
        return os.path.join(checkpoint_dir, f"{checkpoint_arg}.pth")
    return checkpoint_arg


def build_test_loader(config):
    test_transform = build_test_transform(
        img_size=int(config["DATA"]["IMG_SIZE"]),
        tighter_crop_ratio=float(config["DATA"].get("TIGHTER_CROP_RATIO", 1.0)),
    )
    test_dir = os.path.join(config["DATA"]["DATA_PATH"], "test")
    dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    loader = DataLoader(
        dataset,
        batch_size=int(config["DATA"]["BATCH_SIZE"]),
        shuffle=False,
        num_workers=int(config["DATA"]["NUM_WORKERS"]),
        pin_memory=True,
    )
    return dataset, loader


def extract_bridge_features(model, loader, device, max_samples):
    features = []
    labels = []
    paths = []
    collected = 0
    model.eval()

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            fused_map = model.bridge(model.frontend(images))
            pooled = torch.flatten(torch.mean(fused_map, dim=(2, 3)), start_dim=1)

            batch_size = pooled.size(0)
            take_n = min(batch_size, max_samples - collected)
            if take_n <= 0:
                break

            features.append(pooled[:take_n].cpu())
            labels.extend(targets[:take_n].tolist())

            start = collected
            for i in range(take_n):
                paths.append(loader.dataset.samples[start + i][0])

            collected += take_n

            if collected >= max_samples:
                break

    if not features:
        raise RuntimeError("No features extracted. Check data path and loader settings.")

    return torch.cat(features, dim=0).numpy(), labels, paths


def _safe_perplexity(n_samples: float, perplexity: float) -> float:
    # perplexity < n_samples；常用 heuristic：不超过 (n-1)/3
    if n_samples < 2:
        return 1.0
    upper = max(5.0, (n_samples - 1.0) / 3.0)
    return float(min(perplexity, upper, n_samples - 1.0001))


def generate_tsne(
    config: dict,
    *,
    exp_name_override: Optional[str] = None,
    dataset_title: Optional[str] = None,
    checkpoint: str = "best",
    max_samples: int = 3000,
    perplexity: float = 30.0,
    seed: int = 42,
    dpi: int = 300,
) -> Tuple[str, str]:
    """
    返回 (png_path, csv_path)。
    图题格式与混淆矩阵 / 识别率曲线一致：t-SNE (<dataset_title>)
    """
    cfg = copy.deepcopy(config)
    if exp_name_override is not None:
        cfg.setdefault("SYSTEM", {})["EXPERIMENT_NAME"] = str(exp_name_override)

    experiment_name = str(cfg["SYSTEM"]["EXPERIMENT_NAME"])
    title_suffix = dataset_title if dataset_title is not None else experiment_name

    device = cfg["TRAIN"]["DEVICE"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    checkpoint_path = resolve_checkpoint_path(cfg, checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(cfg).to(device)
    checkpoint_obj = load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(checkpoint_obj["model_state_dict"])

    dataset, test_loader = build_test_loader(cfg)
    feats, labels, paths = extract_bridge_features(model, test_loader, device, max_samples)

    n = float(len(labels))
    perp = _safe_perplexity(n, perplexity)

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(feats)

    out_dir = os.path.join(cfg["SYSTEM"]["SAVE_DIR"], experiment_name, "visualization", "tsne")
    os.makedirs(out_dir, exist_ok=True)

    # 与 tools/run_postprocess.set_paper_style 对齐
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )

    classes = dataset.classes
    plt.figure(figsize=(10, 8))
    for class_id, class_name in enumerate(classes):
        idx = [i for i, y in enumerate(labels) if y == class_id]
        if not idx:
            continue
        plt.scatter(emb[idx, 0], emb[idx, 1], s=10, alpha=0.72, label=str(class_name))
    plt.title(f"t-SNE ({title_suffix})")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(markerscale=2, fontsize=9)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "tsne_bridge.png")
    pdf_path = os.path.join(out_dir, "tsne_bridge.pdf")
    plt.savefig(fig_path, dpi=dpi)
    plt.savefig(pdf_path, dpi=dpi)
    plt.close()

    csv_path = os.path.join(out_dir, "tsne_bridge_points.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label_id", "label_name", "sample_path"])
        for i in range(len(labels)):
            writer.writerow([emb[i, 0], emb[i, 1], labels[i], classes[labels[i]], paths[i]])

    return fig_path, csv_path


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    fig_path, csv_path = generate_tsne(
        config,
        exp_name_override=args.exp_name,
        dataset_title=args.dataset_title,
        checkpoint=args.checkpoint,
        max_samples=args.max_samples,
        perplexity=args.perplexity,
        seed=args.seed,
        dpi=args.dpi,
    )
    print(f"Saved figure: {fig_path}")
    print(f"Saved points: {csv_path}")


if __name__ == "__main__":
    main()
