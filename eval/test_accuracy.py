#!/usr/bin/env python3
"""Evaluate ABF-Net / LABF-Net accuracy on an ImageFolder test split."""
import argparse
from pathlib import Path
import sys

import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.build import build_dataloader
from models.build import build_model
from train.evaluate import evaluate
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/abfnet_rafdb.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth checkpoint.")
    parser.add_argument("--save-details", default=None, help="Optional CSV path for per-sample predictions.")
    parser.add_argument("--tta", action="store_true", help="Use horizontal-flip test-time augmentation.")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = config["TRAIN"].get("DEVICE", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    _, test_loader = build_dataloader(
        data_path=config["DATA"]["DATA_PATH"],
        batch_size=config["DATA"]["BATCH_SIZE"],
        img_size=config["DATA"]["IMG_SIZE"],
        num_workers=config["DATA"]["NUM_WORKERS"],
        use_weighted_sampler=False,
        use_random_erasing=False,
        tighter_crop_ratio=float(config["DATA"].get("TIGHTER_CROP_RATIO", 1.0)),
        dataloader_seed=int(config["TRAIN"].get("SEED", 42)),
    )
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config["TRAIN"].get("LABEL_SMOOTHING", 0.0)))
    loss, acc = evaluate(model, test_loader, criterion, device, save_details_path=args.save_details, use_tta=args.tta)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")


if __name__ == "__main__":
    main()
