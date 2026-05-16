

import os
from pathlib import Path
import importlib.util

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mobilefacenet_py_path() -> Path:
    override = os.environ.get("BAIYR_MOBILEFACENET_PY", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (
        Path(__file__).resolve().parent
        / "third_party"
        / "poster_v2_backbone"
        / "mobilefacenet.py"
    )


def _load_mobile_module():
    mobile_path = _mobilefacenet_py_path()
    if not mobile_path.is_file():
        raise FileNotFoundError(
            f"未找到 MobileFaceNet 源码: {mobile_path}（可设置环境变量 BAIYR_MOBILEFACENET_PY）"
        )
    spec = importlib.util.spec_from_file_location("baiyr_vendor_mobilefacenet", mobile_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MOBILE_MODULE = _load_mobile_module()
MobileFaceNet = _MOBILE_MODULE.MobileFaceNet


class MobileBranch(nn.Module):
   

    def __init__(self, pretrained_path: str = "", freeze: bool = False, return_multi_scale: bool = False):
        super(MobileBranch, self).__init__()

        self.return_multi_scale = return_multi_scale

        # 建立师兄的 MobileFaceNet 主干
        self.backbone = MobileFaceNet([112, 112], 136)

        # 如果提供了预训练路径，则加载权重
        if pretrained_path:
            checkpoint = torch.load(
                pretrained_path,
                map_location=lambda storage, loc: storage
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.backbone.load_state_dict(state_dict, strict=False)

        # 如果需要冻结，则不更新 MobileFaceNet 的参数
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        # MobileFaceNet 期望输入分辨率为 112，因此先插值到 112x112
        x_face = F.interpolate(x, size=112)

        # 师兄的 MobileFaceNet forward 返回三个特征：
        # out3, out4, conv_features = self.backbone(x_face)
        x1, x2, x3 = self.backbone(x_face)

        if self.return_multi_scale:
            return {
                "x1": x1,  # 较浅层特征
                "x2": x2,  # 中层特征
                "x3": x3,  # 深层特征（512 通道, 7x7）
            }

        # 默认只返回最后一层 conv_features 作为单尺度特征
        return x3

