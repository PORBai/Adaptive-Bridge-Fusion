#!/usr/bin/env python3
"""
RAF-DB (MANUAL5, folders 1..7): one row per expression — English name on top;
each example is three panels like visualize_gradcam.py: Input | Grad-CAM | Overlay.
Two examples per row are placed side by side (2 × 3 = 6 image panels per row).

Run from project root (same as visualize_gradcam.py):
  python tools/visualize_gradcam_rafdb_panel.py
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import yaml
from PIL import Image

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TOOLS_DIR)
for _p in (_PROJECT_ROOT, _TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.predata import build_test_transform  # noqa: E402
from models.build import build_model  # noqa: E402
from utils.checkpoint import load_checkpoint  # noqa: E402

import visualize_gradcam as _vg  # noqa: E402

blend_gradcam_overlay = _vg.blend_gradcam_overlay
compute_gradcam = _vg.compute_gradcam
resolve_checkpoint_path = _vg.resolve_checkpoint_path
to_uint8_rgb = _vg.to_uint8_rgb

# ImageFolder with folders "1".."7" (sorted) -> idx 0..6.
# MANUAL5: 1=Surprise … 7=Neutral（与同仓库 Ada-DF / NLFER 说明一致）
RAF_CLASS_ENGLISH = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral",
]


def _list_test_images(class_folder: str, k: int) -> list[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    names = sorted(
        f
        for f in os.listdir(class_folder)
        if f.lower().endswith(exts) and os.path.isfile(os.path.join(class_folder, f))
    )
    if len(names) < k:
        raise FileNotFoundError(f"need {k} images in {class_folder}, found {len(names)}")
    return [os.path.join(class_folder, names[i]) for i in range(k)]


def _list_all_test_images(class_folder: str) -> list[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    names = sorted(
        f
        for f in os.listdir(class_folder)
        if f.lower().endswith(exts) and os.path.isfile(os.path.join(class_folder, f))
    )
    return [os.path.join(class_folder, n) for n in names]


def _pick_best_fear_paths(
    class_dir: str,
    k: int,
    model: torch.nn.Module,
    transform,
    device: str,
    class_idx: int,
    pool_limit: int = 220,
    shortlist: int = 36,
) -> list[str]:
    """从 Fear（或其它难例）中选 CAM 更清晰且预测为该类的样本，避免字典序前两张云图难看。"""
    all_paths = _list_all_test_images(class_dir)
    if len(all_paths) < k:
        raise FileNotFoundError(f"need {k} images in {class_dir}, found {len(all_paths)}")
    pool = all_paths[:pool_limit]

    quick: list[tuple[str, float, bool]] = []
    model.eval()
    with torch.no_grad():
        for ipath in pool:
            pil = Image.open(ipath).convert("RGB")
            tensor = transform(pil).unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred = int(probs.argmax().item())
            p_true = float(probs[class_idx].item())
            quick.append((ipath, p_true, pred == class_idx))

    correct = [(p, pt) for p, pt, ok in quick if ok]
    ranked = sorted(correct, key=lambda x: x[1], reverse=True)
    if len(ranked) < k:
        ranked = sorted(((p, pt) for p, pt, _ in quick), key=lambda x: x[1], reverse=True)

    candidates = [p for p, _ in ranked[: max(shortlist, k)]]

    scored: list[tuple[str, float]] = []
    for ipath in candidates:
        pil = Image.open(ipath).convert("RGB")
        tensor = transform(pil).unsqueeze(0).to(device)
        with torch.enable_grad():
            cam, logits = compute_gradcam(model, tensor, class_idx)
        focus = float(cam.std())
        probs = torch.softmax(logits, dim=1)[0]
        p_true = float(probs[class_idx].item())
        scored.append((ipath, p_true * (1.0 + 2.5 * focus)))

    scored.sort(key=lambda x: x[1], reverse=True)
    out = [p for p, _ in scored[:k]]
    if len(out) < k:
        rest = [p for p in all_paths if p not in out]
        out.extend(rest[: k - len(out)])
    print(f"[Fear smart-pick] using {len(out)} samples:", *[os.path.basename(x) for x in out])
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default=os.path.join(_PROJECT_ROOT, "configs/abfnet_rafdb.yaml"),
    )
    p.add_argument("--checkpoint", default="best", help="best / latest / path to .pth")
    p.add_argument(
        "--exp-name",
        default=None,
        help="Override SYSTEM.EXPERIMENT_NAME (checkpoint + default --out directory).",
    )
    p.add_argument(
        "--test-root",
        default=None,
        help="Override RAF test dir; default DATA_PATH/test from config.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Default SAVE_DIR/<exp>/visualization/gradcam/rafdb_gradcam_panel.png",
    )
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument(
        "--per-row",
        type=int,
        default=2,
        help="number of example images per expression (side by side)",
    )
    p.add_argument(
        "--no-fear-smart-pick",
        action="store_true",
        help="Fear 行仍用字典序前两张；默认启用「预测正确 + 置信度 × CAM 反差」自动挑更清晰示例",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.exp_name is not None:
        config.setdefault("SYSTEM", {})["EXPERIMENT_NAME"] = str(args.exp_name)

    device = config["TRAIN"]["DEVICE"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    cp = resolve_checkpoint_path(config, args.checkpoint)
    if not os.path.isfile(cp):
        raise FileNotFoundError(f"Checkpoint not found: {cp}")

    test_root = args.test_root or os.path.join(str(config["DATA"]["DATA_PATH"]), "test")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"test root not found: {test_root}")

    img_size = int(config["DATA"]["IMG_SIZE"])
    tighter = float(config["DATA"].get("TIGHTER_CROP_RATIO", 1.0))
    transform = build_test_transform(img_size=img_size, tighter_crop_ratio=tighter)

    model = build_model(config).to(device)
    ck = load_checkpoint(cp, device=device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    k = max(1, args.per_row)
    exp_name = str(config["SYSTEM"]["EXPERIMENT_NAME"])
    out_path = args.out
    if not out_path:
        out_dir = os.path.join(config["SYSTEM"]["SAVE_DIR"], exp_name, "visualization", "gradcam")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "rafdb_gradcam_panel.png")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    n_classes = len(RAF_CLASS_ENGLISH)
    # 顶隙与「上一类图下方 → 下一类情绪名」之间的白边用同一 height_ratio（与 Surprise 顶上留高一致）
    gap_ratio = 0.14
    label_ratio = 0.038
    img_ratio = 1.0
    ratios: list[float] = [gap_ratio]
    for i in range(n_classes):
        ratios.extend([label_ratio, img_ratio])
        if i < n_classes - 1:
            ratios.append(gap_ratio)

    col_w = 1.28
    fig_h = 1.22 * n_classes + 0.2 * (n_classes + 1)
    fig = plt.figure(
        figsize=(col_w * k * 3 + 0.35, fig_h),
        constrained_layout=False,
    )
    outer = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.0, wspace=0.0)

    sum_ratios = sum(ratios)
    edge_pad = gap_ratio / sum_ratios
    pair_wspace = gap_ratio

    triplet_titles = ("Input", "Grad-CAM", "Overlay")
    title_fs = 6
    emotion_fs = 11

    row = 0
    ax_top = fig.add_subplot(outer[row])
    ax_top.axis("off")
    row += 1

    for class_idx, en_name in enumerate(RAF_CLASS_ENGLISH):
        folder = str(class_idx + 1)
        class_dir = os.path.join(test_root, folder)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"missing class folder: {class_dir}")

        if class_idx == 1 and not args.no_fear_smart_pick:
            paths = _pick_best_fear_paths(
                class_dir,
                k,
                model,
                transform,
                device,
                class_idx,
            )
        else:
            paths = _list_test_images(class_dir, k)

        ax_title = fig.add_subplot(outer[row])
        ax_title.axis("off")
        ax_title.text(0.5, 0.55, en_name, ha="center", va="center", fontsize=emotion_fs, fontweight="600")
        row += 1

        row_gs = outer[row].subgridspec(1, k, wspace=pair_wspace)
        for j, ipath in enumerate(paths):
            pil = Image.open(ipath).convert("RGB")
            tensor = transform(pil).unsqueeze(0).to(device)
            rgb = to_uint8_rgb(pil, img_size)

            with torch.enable_grad():
                cam, _ = compute_gradcam(model, tensor, class_idx)
            overlay, cam_resized = blend_gradcam_overlay(rgb, cam)

            triplet_gs = row_gs[0, j].subgridspec(1, 3, wspace=0.028)
            ax_in = fig.add_subplot(triplet_gs[0, 0])
            ax_cam = fig.add_subplot(triplet_gs[0, 1])
            ax_ov = fig.add_subplot(triplet_gs[0, 2])

            ax_in.imshow(rgb)
            ax_cam.imshow(cam_resized, cmap="jet", vmin=0.0, vmax=1.0)
            ax_ov.imshow(overlay)
            for ax, t in zip((ax_in, ax_cam, ax_ov), triplet_titles):
                ax.set_title(t, fontsize=title_fs, pad=0)
                ax.axis("off")

        row += 1
        if class_idx < n_classes - 1:
            ax_gap = fig.add_subplot(outer[row])
            ax_gap.axis("off")
            row += 1

    fig.subplots_adjust(
        left=edge_pad,
        right=1.0 - edge_pad,
        top=1.0 - edge_pad,
        bottom=edge_pad,
    )
    pad_in = float(edge_pad * min(fig.get_size_inches()))
    fig.savefig(out_path, dpi=args.dpi, bbox_inches=None, pad_inches=pad_in)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
