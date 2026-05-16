import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

from datasets.predata import build_test_transform
from models.build import build_model
from utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for Baiyr.")
    parser.add_argument(
        "--config",
        default="configs/abfnet_rafdb.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="Checkpoint path or keyword: best/latest.",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Input image path.",
    )
    parser.add_argument(
        "--class-idx",
        type=int,
        default=None,
        help="Target class idx. Default uses model prediction.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help=(
            "Subfolder name under SAVE_DIR for Grad-CAM output "
            "(.../<name>/visualization/gradcam). "
            "Checkpoint still loaded from config EXPERIMENT_NAME unless --checkpoint is a file path."
        ),
    )
    return parser.parse_args()


def resolve_checkpoint_path(config, checkpoint_arg):
    experiment_name = str(config["SYSTEM"]["EXPERIMENT_NAME"])
    checkpoint_dir = os.path.join(config["SYSTEM"]["SAVE_DIR"], experiment_name, "checkpoint")
    if checkpoint_arg in {"best", "latest"}:
        return os.path.join(checkpoint_dir, f"{checkpoint_arg}.pth")
    return checkpoint_arg


def to_uint8_rgb(pil_image, img_size):
    resized = pil_image.convert("RGB").resize((img_size, img_size))
    return np.array(resized, dtype=np.uint8)


def compute_gradcam(model, input_tensor, target_class_idx):
    model.eval()

    # Use bridge output feature map for CAM so it works across backend variants.
    fused_map = model.bridge(model.frontend(input_tensor))
    fused_map.retain_grad()
    logits = model.backend(fused_map)

    score = logits[:, target_class_idx].sum()
    model.zero_grad()
    score.backward()

    grads = fused_map.grad  # [B, C, H, W]
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam = (weights * fused_map).sum(dim=1, keepdim=True)  # [B, 1, H, W]
    cam = torch.relu(cam)
    cam = cam[0, 0].detach().cpu().numpy()

    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam, logits.detach()


def blend_gradcam_overlay(rgb_uint8, cam):
    """Return (overlay_rgb_uint8[H,W,3], cam_resized[H,W] float)."""
    h, w, _ = rgb_uint8.shape
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * rgb_uint8 + 0.45 * heatmap).astype(np.uint8)
    return overlay, cam_resized


def save_overlay(rgb_uint8, cam, out_path):
    overlay, cam_resized = blend_gradcam_overlay(rgb_uint8, cam)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Input")
    plt.imshow(rgb_uint8)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM")
    plt.imshow(cam_resized, cmap="jet")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = config["TRAIN"]["DEVICE"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    checkpoint_path = resolve_checkpoint_path(config, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    img_size = int(config["DATA"]["IMG_SIZE"])
    tighter_crop_ratio = float(config["DATA"].get("TIGHTER_CROP_RATIO", 1.0))
    transform = build_test_transform(img_size=img_size, tighter_crop_ratio=tighter_crop_ratio)

    image_pil = Image.open(args.image).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    image_rgb = to_uint8_rgb(image_pil, img_size)

    with torch.enable_grad():
        if args.class_idx is None:
            with torch.no_grad():
                pred_logits = model(image_tensor)
                class_idx = int(torch.argmax(pred_logits, dim=1).item())
        else:
            class_idx = int(args.class_idx)
        cam, logits = compute_gradcam(model, image_tensor, class_idx)

    pred_idx = int(torch.argmax(logits, dim=1).item())
    pred_prob = float(torch.softmax(logits, dim=1)[0, pred_idx].item())

    experiment_name = str(config["SYSTEM"]["EXPERIMENT_NAME"])
    out_root = args.output_subdir if args.output_subdir is not None else experiment_name
    out_dir = os.path.join(config["SYSTEM"]["SAVE_DIR"], out_root, "visualization", "gradcam")
    os.makedirs(out_dir, exist_ok=True)

    img_stem = os.path.splitext(os.path.basename(args.image))[0]
    out_path = os.path.join(out_dir, f"{img_stem}_gradcam.png")
    save_overlay(image_rgb, cam, out_path)

    print(f"Saved Grad-CAM: {out_path}")
    print(f"Target class idx: {class_idx}")
    print(f"Predicted class idx: {pred_idx}, prob={pred_prob:.4f}")


if __name__ == "__main__":
    main()
