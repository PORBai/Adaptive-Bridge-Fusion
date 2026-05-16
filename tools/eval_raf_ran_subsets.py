"""
在 RAN / Challenge-condition-FER 公布的 RAF-DB 子集列表上评测已训好的 7 类模型。

三个子集（与常见论文表 Occlusion / Pose≥30 / Pose≥45 对应）：
- rafdb_occlusion_list.txt
- val_raf_db_list.txt + val_raf_db_label.txt（Pose≥30）
- val_raf_db_list_45.txt + val_raf_db_label_45.txt（Pose≥45）

图像须使用官方 **DATASET_ALIGNED_7CLS**（test/1..7/test_xxxx_aligned.jpg）。
列表里类号为 0～6，磁盘文件夹为 1～7，本脚本会自动 +1。

用法（示例）：
  cd ABFNet_TVC_OpenSource
  python tools/eval_raf_ran_subsets.py \\
    --config configs/abfnet_rafdb.yaml \\
    --checkpoint results/1/checkpoint/best.pth
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml
from PIL import Image
from torch import nn

# 项目根：tools/ 的上级
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.predata import build_test_transform
from models.build import build_model

DEFAULT_RAF_ALIGNED = "./data/RAFDB_ALIGNED"
DEFAULT_RAN_DIR = "./datasets/lists/RAF_DB_dir"


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_readable_file(path: str) -> bool:
    """与 FER+ 子集评测一致：只统计可读的真实文件/有效软链，跳过断链。"""
    if not path:
        return False
    if os.path.isfile(path) and not os.path.islink(path):
        return True
    if os.path.islink(path) and os.path.exists(path) and os.path.isfile(path):
        return True
    return False


def _iter_raf_test_class_dirs(test_root: str) -> list[str]:
    tdir = os.path.join(test_root, "test")
    if not os.path.isdir(tdir):
        return []
    return [os.path.join(tdir, str(n)) for n in range(1, 8) if os.path.isdir(os.path.join(tdir, str(n)))]


def _build_raf_basename_to_path(test_root: str) -> dict[str, str]:
    """
    RAN 名单按「某类文件夹 + 全文件名」拼路径；若断链/错误文件夹，可凭文件名在 test/1..7/ 中回退。
    与 eval_ferplus_ran_subsets 中 stem 索引思想一致。键为「含扩展名」如 test_0004_aligned.jpg。
    """
    m: dict[str, str] = {}
    for d in _iter_raf_test_class_dirs(test_root):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if not name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            p = os.path.join(d, name)
            if not _is_readable_file(p):
                continue
            if name not in m:
                m[name] = p
    return m


def _parse_pose_pairs(
    list_path: str, label_path: str, test_root: str, name_index: dict[str, str] | None
) -> tuple[list[str], list[int]]:
    """读取 val_raf_db_list / val_raf_db_list_45 与对应 label 文件。"""
    with open(list_path, "r", encoding="utf-8") as f:
        rels = [ln.strip() for ln in f if ln.strip()]
    with open(label_path, "r", encoding="utf-8") as f:
        raw = f.readlines()
    first = raw[0].strip().split()
    if len(first) == 2 and first[0].isdigit() and first[1].isdigit():
        label_vals = [int(ln.strip()) for ln in raw[1:] if ln.strip()]
    else:
        label_vals = [int(ln.strip()) for ln in raw if ln.strip()]
    if len(rels) != len(label_vals):
        raise ValueError(
            f"列表与标签条数不一致: {list_path} ({len(rels)}) vs {label_path} ({len(label_vals)})"
        )
    paths: list[str] = []
    ys: list[int] = []
    for rel, yi in zip(rels, label_vals):
        c_str, name = rel.split("/", 1)
        folder = str(int(c_str) + 1)
        p = os.path.join(test_root, "test", folder, name)
        if not _is_readable_file(p) and name_index is not None and name in name_index:
            p = name_index[name]
        paths.append(p)
        ys.append(yi)
    return paths, ys


def _parse_occlusion(
    path: str, test_root: str, name_index: dict[str, str] | None
) -> tuple[list[str], list[int]]:
    """
    rafdb_occlusion_list.txt 每行：test_xxxx_aligned <gt_0_6> <其他数字>
    使用第二列为 0～6 类标，拼路径 test/{gt+1}/test_xxxx_aligned.jpg
    """
    paths: list[str] = []
    ys: list[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            stem = parts[0]
            gt = int(parts[1])
            fname = stem if stem.endswith(".jpg") else f"{stem}.jpg"
            folder = str(gt + 1)
            p = os.path.join(test_root, "test", folder, fname)
            if not _is_readable_file(p) and name_index is not None and fname in name_index:
                p = name_index[fname]
            paths.append(p)
            ys.append(gt)
    return paths, ys


def _filter_existing(paths: list[str], ys: list[int], tag: str) -> tuple[list[str], list[int]]:
    good_p, good_y = [], []
    for p, y in zip(paths, ys):
        if _is_readable_file(p):
            good_p.append(p)
            good_y.append(y)
        else:
            print(f"[{tag}] 缺少文件: {p}")
    if not good_p:
        print(f"[{tag}] 无有效图像，请检查 DATASET_ALIGNED_7CLS 与列表是否来自同一版 RAF。")
    else:
        print(f"[{tag}] 使用 {len(good_p)} / {len(paths)} 张图。")
    return good_p, good_y


@torch.no_grad()
def _run(
    model: nn.Module,
    device: torch.device,
    transform,
    paths: list[str],
    ys: list[int],
    batch_size: int,
) -> float:
    if not paths:
        return float("nan")
    n_correct = 0
    n_total = 0
    for i in range(0, len(paths), batch_size):
        batch_p = paths[i : i + batch_size]
        batch_y = ys[i : i + batch_size]
        imgs = []
        for p in batch_p:
            im = Image.open(p).convert("RGB")
            imgs.append(transform(im))
        x = torch.stack(imgs, dim=0).to(device)
        t = torch.tensor(batch_y, dtype=torch.long, device=device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        n_correct += (pred == t).sum().item()
        n_total += len(batch_y)
    return n_correct / max(n_total, 1)


def main():
    ap = argparse.ArgumentParser(description="RAF-DB RAN 子集 Occ / Pose30 / Pose45 评测")
    ap.add_argument(
        "--config",
        default=os.path.join(_PROJECT_ROOT, "config", "train_rafdb.yaml"),
        help="与训练时结构一致的 yaml（7 类）",
    )
    ap.add_argument("--checkpoint", required=True, type=str, help="best.pth 或 latest.pth")
    ap.add_argument(
        "--raf-aligned-root",
        default=None,
        help="RAN 评测用 **DATASET_ALIGNED_7CLS**（*aligned.jpg）；未指定时默认用下方常数。勿与训练用 MANUAL5 根目录混用。",
    )
    ap.add_argument("--ran-dir", default=DEFAULT_RAN_DIR, help="RAF_DB_dir（含 list）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU。")
        args.device = "cpu"
    device = torch.device(args.device)

    test_root = os.path.normpath(
        args.raf_aligned_root if args.raf_aligned_root else DEFAULT_RAF_ALIGNED
    )
    ran = os.path.normpath(args.ran_dir)
    if not os.path.isdir(os.path.join(test_root, "test")):
        raise FileNotFoundError(f"未找到 {test_root}/test，请确认 DATASET_ALIGNED_7CLS 路径。")

    config = _load_yaml(args.config)
    config_path_abs = os.path.abspath(args.config)
    img_size = int(config["DATA"]["IMG_SIZE"])
    tighter = float(config["DATA"].get("TIGHTER_CROP_RATIO", 1.0))
    transform = build_test_transform(img_size=img_size, tighter_crop_ratio=tighter)

    model = build_model(config)
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    # 与 FER+ 子集评测类似：为断链/路径不一致时按文件名在 test/1..7/ 中回退
    print("正在扫描 test/1..7 中可读 RAF-Aligned 图（建文件名索引）…")
    raf_name_index = _build_raf_basename_to_path(test_root)
    print(f"  已索引 {len(raf_name_index)} 个不重复文件名。")

    # ---- 三份子集 ----
    occ_p, occ_y = _parse_occlusion(
        os.path.join(ran, "rafdb_occlusion_list.txt"), test_root, raf_name_index
    )
    occ_p, occ_y = _filter_existing(occ_p, occ_y, "Occlusion")

    p30_p, p30_y = _parse_pose_pairs(
        os.path.join(ran, "val_raf_db_list.txt"),
        os.path.join(ran, "val_raf_db_label.txt"),
        test_root,
        raf_name_index,
    )
    p30_p, p30_y = _filter_existing(p30_p, p30_y, "Pose>=30")

    p45_p, p45_y = _parse_pose_pairs(
        os.path.join(ran, "val_raf_db_list_45.txt"),
        os.path.join(ran, "val_raf_db_label_45.txt"),
        test_root,
        raf_name_index,
    )
    p45_p, p45_y = _filter_existing(p45_p, p45_y, "Pose>=45")

    print()
    print("config:", config_path_abs)
    print("checkpoint:", os.path.abspath(args.checkpoint))
    print("测试图根目录:", test_root)
    print("列表目录:", ran)
    print()

    a_occ = _run(model, device, transform, occ_p, occ_y, args.batch_size)
    a30 = _run(model, device, transform, p30_p, p30_y, args.batch_size)
    a45 = _run(model, device, transform, p45_p, p45_y, args.batch_size)

    print("--- 结果（Test Accuracy）---")
    print(f"  Occlusion-RAF-DB:   {a_occ * 100:.2f}%" if a_occ == a_occ else "  Occlusion-RAF-DB:   nan")
    print(f"  Pose-RAF-DB (≥30):  {a30 * 100:.2f}%" if a30 == a30 else "  Pose-RAF-DB (≥30):  nan")
    print(f"  Pose-RAF-DB (≥45):  {a45 * 100:.2f}%" if a45 == a45 else "  Pose-RAF-DB (≥45):  nan")
    print()
    print("说明：与论文/他人表格对比时，请标明子集来自 kaiwang960112/Challenge-condition-FER-dataset 的 RAF_DB_dir。")


if __name__ == "__main__":
    main()
