"""
从 AffectNet 子集（zip 内常见结构 archive (3)/Train, Test, labels.csv）构建 ImageFolder。

- 以 labels.csv 的 **label 列** 为类别（pth 仅用于定位图片；与常见 Kaggle 包一致）
- 8 类：与 FER+ 8 类名顺序一致，输出子目录名 1..8
- 7 类：在 8 类基础上去掉 **contempt**（不复制 contempt 行）

用法:
  python datasets/build_affectnet_imagefolder.py --src-root "..." --out-root "..." --num-classes 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter


# 与仓库内 FER+ 的 EMOTION_ORDER 一致（1..8）
EMOTION_EIGHT = [
    "neutral",  # 1
    "happy",  # 2  (论文里常写 happiness)
    "surprise",  # 3
    "sad",  # 4  (sadness)
    "anger",  # 5
    "disgust",  # 6
    "fear",  # 7
    "contempt",  # 8
]

# 7 类 = 8 类去掉最后一项 contempt
EMOTION_SEVEN = EMOTION_EIGHT[:7]
LABEL_STR_TO_FER8 = {e: i + 1 for i, e in enumerate(EMOTION_EIGHT)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src-root",
        type=str,
        default=None,
        help="解压后含 Train/、Test/ 与 labels.csv 的目录；默认可为 zip 内 archive (3) 解压位置",
    )
    p.add_argument(
        "--affectnet-zip",
        type=str,
        default=None,
        help="若给出，将先把 archive (3) 解压到 --zip-extract-to",
    )
    p.add_argument(
        "--zip-extract-to",
        type=str,
        default="./data/affectnet_unpacked",
    )
    p.add_argument("--out-root", type=str, required=True)
    p.add_argument("--num-classes", type=int, choices=[7, 8], required=True)
    p.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    p.add_argument("--clean-output", action="store_true")
    return p.parse_args()


def ensure_extract_from_zip(zip_path, out_root):
    import zipfile

    key = "archive (3)/"
    os.makedirs(out_root, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        names = [n for n in z.namelist() if n.startswith(key) and not n.endswith("/")]
        for n in names:
            z.extract(n, out_root)
    return os.path.join(out_root, "archive (3)")


def _find_image(src_root, pth: str):
    """pth 形如 anger/image0000006.jpg，Train/小写、Test/首字母大写 均尝试。"""
    pth = pth.replace("\\", "/").strip()
    if "/" not in pth:
        return None
    parts = pth.split("/")
    fname = parts[-1]
    class0 = parts[0]
    mid = parts[1:-1]  # 多余层（若有）
    for split in ("Train", "Test"):
        for c in (class0, class0.capitalize(), class0.upper()):
            cand = os.path.join(src_root, split, c, *mid, fname)
            if os.path.isfile(cand):
                return cand
    return None


def _fer8_from_label_str(label: str) -> int | None:
    s = (label or "").strip().lower()
    if s == "happiness":  # 若 csv 出现别名
        s = "happy"
    if s == "sadness":
        s = "sad"
    if s not in LABEL_STR_TO_FER8:
        return None
    return LABEL_STR_TO_FER8[s]


def _split_fer8(fer8: int, out_k: int) -> int | None:
    if out_k == 8:
        return fer8
    # 7: 1..7 为 EMOTION_SEVEN，无 contempt(8)
    if fer8 == 8:
        return None
    return fer8  # 仍是 1..7


def place_file(src, dst, copy_mode: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def main():
    args = parse_args()
    if args.affectnet_zip and not args.src_root:
        src_root = ensure_extract_from_zip(args.affectnet_zip, args.zip_extract_to)
    elif args.src_root:
        src_root = os.path.abspath(args.src_root)
    else:
        raise SystemExit("请指定 --src-root 或 --affectnet-zip")

    labels_path = os.path.join(src_root, "labels.csv")
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"未找到 {labels_path}")

    if args.clean_output and os.path.isdir(args.out_root):
        shutil.rmtree(args.out_root)
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)
    ncls = args.num_classes
    for sp in ("train", "test"):
        for c in range(1, ncls + 1):
            os.makedirs(os.path.join(out_root, sp, str(c)), exist_ok=True)

    stats = Counter()
    by_split_class = {"train": Counter(), "test": Counter()}
    # 避免同名：跨 split 的 basename 可能重复，用 hash 不美观；用 pth 做文件名成分
    used_names = set()

    with open(labels_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for line_no, row in enumerate(r, start=2):
            pth = (row.get("pth") or row.get("path") or "").strip()
            label = row.get("label", "").strip()
            if not pth or pth == "pth":
                continue
            fer8 = _fer8_from_label_str(label)
            if fer8 is None:
                stats["skip_bad_label"] += 1
                continue
            out_cls = _split_fer8(fer8, args.num_classes)
            if out_cls is None:
                stats["skip_excluded_7contempt"] += 1
                continue
            full = _find_image(src_root, pth)
            if full is None:
                stats["skip_image_not_found"] += 1
                if stats["skip_image_not_found"] <= 5:
                    print(f"[miss] pth={pth}")
                continue
            norm = full.replace("\\", "/")
            if "/Train/" in norm:
                part = "train"
            elif "/Test/" in norm:
                part = "test"
            else:
                stats["skip_unknown_split"] += 1
                continue
            out_dir = os.path.join(out_root, part, str(out_cls))
            safe = pth.replace("/", "__").replace("\\", "__")
            dst = os.path.join(out_dir, safe)
            if dst in used_names:
                base, ext = os.path.splitext(safe)
                safe = f"{base}__L{line_no}{ext}"
                dst = os.path.join(out_dir, safe)
            used_names.add(dst)
            try:
                place_file(full, dst, args.copy_mode)
            except OSError as e:
                stats["place_error"] += 1
                if stats["place_error"] <= 2:
                    print(f"[err] {e}")
                continue
            by_split_class[part][str(out_cls)] += 1
            stats["placed"] += 1

    summary = {
        "src_root": src_root,
        "out_root": out_root,
        "num_classes": args.num_classes,
        "emotions_order_1_to_K": (EMOTION_SEVEN if args.num_classes == 7 else EMOTION_EIGHT).copy(),
        "copy_mode": args.copy_mode,
        "stats": dict(stats),
        "per_split_class": {k: dict(v) for k, v in by_split_class.items()},
    }
    with open(os.path.join(out_root, "build_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2))
    print("Summary ->", os.path.join(out_root, "build_summary.json"))


if __name__ == "__main__":
    main()
