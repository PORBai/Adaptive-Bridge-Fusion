"""
在 RAN / Challenge-condition-FER 公布的 FER+ 子集列表上评测已训好的 7 类模型。

三个子集（与常见论文表 Occlusion / Pose≥30 / Pose≥45 对应）：
- jianfei_occlusion_list.txt
- pose_30_ferplus_list.txt
- pose_45_ferplus_list.txt

图像使用 **FERPLUS_OFFICIAL_IMAGEFOLDER_7CLS**（test/1..7/fer*.png）。
遮挡列表行首为 `{0-6}_ferxxxx.png` 表示 7 类表情标与文件名；含 `7_` 的样本为 8 类中的第 8 类，
7 类训练不覆盖，本脚本 **跳过**。
姿态列表每行 `c/ferxxxx.jpg`（c 为 0～6），磁盘上优先匹配 `.png` 再 `.jpg`。

用法：
  cd ABFNet_TVC_OpenSource
  python tools/eval_ferplus_ran_subsets.py \\
    --config configs/abfnet_ferplus.yaml \\
    --checkpoint results/1.2/checkpoint/best.pth
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch
import yaml
from PIL import Image
from torch import nn

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.predata import build_test_transform
from models.build import build_model

DEFAULT_FER_ROOT = "./data/FERPlus"
DEFAULT_FERPLUS_RAN_DIR = (
    "./datasets/lists/FERplus_dir"
)

OCC_FIRST = re.compile(r"^([0-7])_(fer[0-9]+)\.png$", re.IGNORECASE)


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_readable_file(path: str) -> bool:
    if not path:
        return False
    if os.path.isfile(path) and not os.path.islink(path):
        return True
    if os.path.islink(path) and os.path.exists(path) and os.path.isfile(path):
        return True
    return False


def _iter_test_class_dirs(test_root: str) -> list[str]:
    tdir = os.path.join(test_root, "test")
    if not os.path.isdir(tdir):
        return []
    out: list[str] = []
    for n in range(1, 8):
        d = os.path.join(tdir, str(n))
        if os.path.isdir(d):
            out.append(d)
    return out


def _build_fer_stem_to_path(test_root: str) -> dict[str, str]:
    """
    各 fer 图片 id 在数据集中应唯一，遍历 test/1..7/ 用 stem 建索引；
    若同 stem 多路径，保留首次遇到的。
    仅记录当前可读的真实文件，跳过断链。
    """
    m: dict[str, str] = {}
    for d in _iter_test_class_dirs(test_root):
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            stem, ext = os.path.splitext(name)
            if ext.lower() not in (".png", ".jpg", ".jpeg") or not stem.lower().startswith("fer"):
                continue
            p = os.path.join(d, name)
            if not _is_readable_file(p):
                continue
            if stem not in m:
                m[stem] = p
    return m


def _resolve_fer_file(
    test_root: str, y: int, stem: str, stem_index: dict[str, str] | None = None
) -> str | None:
    """优先 test/{y+1}/{stem}.(png|jpg|jpeg)；若断链/缺失，再用 stem 全局索引回退。"""
    folder = str(y + 1)
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(test_root, "test", folder, f"{stem}{ext}")
        if _is_readable_file(p):
            return p
    if stem_index is not None and stem in stem_index:
        return stem_index[stem]
    return None


def _parse_ferplus_occlusion(
    list_path: str,
    test_root: str,
    stem_index: dict[str, str] | None = None,
) -> tuple[list[str], list[int], int]:
    """行首 `c_ferxxxx.png`：c∈0～6 为标签；c=7 跳过。"""
    paths: list[str] = []
    ys: list[int] = []
    skipped7 = 0
    bad = 0
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok = line.split()[0]
            m = OCC_FIRST.match(tok)
            if not m:
                bad += 1
                continue
            gt = int(m.group(1))
            stem = m.group(2)
            if gt > 6:
                skipped7 += 1
                continue
            p = _resolve_fer_file(test_root, gt, stem, stem_index)
            if p is None:
                bad += 1
                continue
            paths.append(p)
            ys.append(gt)
    if skipped7:
        print(
            f"[Occlusion] 跳过 8 类标签（前缀 7_）共 {skipped7} 条（7 类模型不包含）"
        )
    if bad:
        print(f"[Occlusion] 无法解析或缺图行数: {bad}")
    return paths, ys, skipped7


def _parse_ferplus_pose(
    list_path: str, test_root: str, stem_index: dict[str, str] | None = None
) -> tuple[list[str], list[int], int]:
    """每行 `c/ferxxxx.jpg`，标签 c∈0～6。"""
    paths: list[str] = []
    ys: list[int] = []
    bad = 0
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "/" not in line:
                bad += 1
                continue
            c_str, name = line.split("/", 1)
            y = int(c_str)
            stem, _ = os.path.splitext(name)
            p = _resolve_fer_file(test_root, y, stem, stem_index)
            if p is None:
                bad += 1
                continue
            paths.append(p)
            ys.append(y)
    if bad:
        print(f"[Pose] {os.path.basename(list_path)} 无法解析或缺图: {bad} 行")
    return paths, ys, bad


def _print_subset(tag: str, paths: list[str], ys: list[int]) -> None:
    if not paths:
        print(f"[{tag}] 无有效图像，请检查 FER+ 根目录与列表。")
    else:
        print(f"[{tag}] 使用 {len(paths)} 张图。")


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
    ap = argparse.ArgumentParser(description="FER+ RAN 子集 Occ / Pose30 / Pose45 评测")
    ap.add_argument(
        "--config",
        default=os.path.join(_PROJECT_ROOT, "config", "train_ferplus_official.yaml"),
        help="与训练时结构一致的 yaml（7 类）",
    )
    ap.add_argument("--checkpoint", required=True, type=str, help="best.pth 等")
    ap.add_argument(
        "--fer-root",
        default=None,
        help="含 test/1..7/ 的 FER+ 根目录；未指定时读取 yaml 的 DATA.DATA_PATH",
    )
    ap.add_argument(
        "--ferplus-ran-dir",
        default=DEFAULT_FERPLUS_RAN_DIR,
        help="含 jianfei_occlusion / pose_30 / pose_45 列表的 FERplus_dir",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，使用 CPU。")
        args.device = "cpu"
    device = torch.device(args.device)

    config = _load_yaml(args.config)
    fer_root = args.fer_root or config.get("DATA", {}).get("DATA_PATH") or DEFAULT_FER_ROOT
    test_root = os.path.normpath(fer_root)
    ran = os.path.normpath(args.ferplus_ran_dir)
    if not os.path.isdir(os.path.join(test_root, "test")):
        raise FileNotFoundError(f"未找到 {test_root}/test。")
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

    print("正在扫描 test/1..7 中可读的 FER+ 图（建 stem 索引，供断链时回退）…")
    stem_index = _build_fer_stem_to_path(test_root)
    print(f"  已索引 {len(stem_index)} 个不重复 stem。")

    occ_p, occ_y, _ = _parse_ferplus_occlusion(
        os.path.join(ran, "jianfei_occlusion_list.txt"), test_root, stem_index
    )
    _print_subset("Occlusion", occ_p, occ_y)

    p30_p, p30_y, _ = _parse_ferplus_pose(
        os.path.join(ran, "pose_30_ferplus_list.txt"), test_root, stem_index
    )
    _print_subset("Pose>=30", p30_p, p30_y)

    p45_p, p45_y, _ = _parse_ferplus_pose(
        os.path.join(ran, "pose_45_ferplus_list.txt"), test_root, stem_index
    )
    _print_subset("Pose>=45", p45_p, p45_y)

    print()
    print("config:", config_path_abs)
    print("checkpoint:", os.path.abspath(args.checkpoint))
    print("测试图根目录:", test_root)
    print("列表目录:", ran)
    print()

    a_occ = _run(model, device, transform, occ_p, occ_y, args.batch_size)
    a30 = _run(model, device, transform, p30_p, p30_y, args.batch_size)
    a45 = _run(model, device, transform, p45_p, p45_y, args.batch_size)

    print("--- 结果（Test Accuracy, %）---")
    if a_occ == a_occ:
        print(f"  Occlusion-FER+:   {a_occ * 100:.2f}")
    else:
        print("  Occlusion-FER+:   nan")
    if a30 == a30:
        print(f"  Pose-FER+ (>=30): {a30 * 100:.2f}")
    else:
        print("  Pose-FER+ (>=30): nan")
    if a45 == a45:
        print(f"  Pose-FER+ (>=45): {a45 * 100:.2f}")
    else:
        print("  Pose-FER+ (>=45): nan")
    print()
    print("说明：子集来自 kaiwang960112/Challenge-condition-FER-dataset 的 FERplus_dir。")


if __name__ == "__main__":
    main()
