import os
from pathlib import Path
import importlib.util

import torch
import torch.nn as nn


def _ir50_py_path() -> Path:
    override = os.environ.get("BAIYR_IR50_PY", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent / "third_party" / "poster_v2_backbone" / "ir50.py"


def _load_ir50_module():
    ir50_path = _ir50_py_path()
    if not ir50_path.is_file():
        raise FileNotFoundError(
            f"未找到 IR50 Backbone 源码: {ir50_path}（可设置环境变量 BAIYR_IR50_PY）"
        )
    spec = importlib.util.spec_from_file_location("baiyr_vendor_ir50", ir50_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_IR50_MODULE = _load_ir50_module()
Backbone = _IR50_MODULE.Backbone
load_pretrained_weights = _IR50_MODULE.load_pretrained_weights


class IR50Branch(nn.Module):
    def __init__(
        self,
        pretrained_path="",
        freeze=False,
        return_multi_scale=False,
        train_body2=False,
        train_last_stage=False,
    ):
        super(IR50Branch, self).__init__()

        self.return_multi_scale = return_multi_scale
        self.backbone = Backbone(50, 0.0, "ir")

        if pretrained_path:
            checkpoint = torch.load(
                pretrained_path,
                map_location=lambda storage, loc: storage
            )
            self.backbone = load_pretrained_weights(self.backbone, checkpoint)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 实验15：在整体冻结的前提下，只放开高层 stage。
        # 让预训练 backbone 主要部分保持稳定，同时给 bridge 更充足的高层适配空间。
        if freeze and train_body2:
            for param in self.backbone.body2.parameters():
                param.requires_grad = True

        if freeze and train_last_stage:
            for param in self.backbone.body3.parameters():
                param.requires_grad = True

    def forward(self, x):
        x1, x2, x3 = self.backbone(x)

        if self.return_multi_scale:
            return {
                "x1": x1,
                "x2": x2,
                "x3": x3,
            }

        return x3
