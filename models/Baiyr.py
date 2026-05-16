
import torch
import torch.nn as nn

from .dual_branch_frontend import DualBranchFrontend
from .bridge import BridgeLayer
from .lightweight_backend import LightConvSEBackend
from .mamba_backend import MambaBackend
from .transformer_backend import TransformerBackend


class Baiyr(nn.Module):
 
    def __init__(
        self,
        num_classes=7,
        frontend_mode="simple",
        branch_a_mode="a_simple",
        branch_b_mode="b_ir50_multi",
        bridge_out_channels=32,
        bridge_use_multi_scale_residual=False,
        bridge_use_scale_weighting=False,
        backend_type="transformer",
        light_backend_hidden_channels=64,
        mamba_embed_dim=128,
        mamba_token_len=49,
        mamba_num_layers=2,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_use_cls=False,
        mamba_version="v1_plain",
        mamba_local_kernel_size=3,
        use_aux_head=False,
        ir50_pretrained_path="",
        ir50_freeze=False,
        ir50_train_body2=False,
        ir50_train_last_stage=False,
        mobile_pretrained_path="",
        mobile_freeze=False,
    ):
        super(Baiyr, self).__init__()

        # 前端：负责双分支特征提取
        self.frontend = DualBranchFrontend(
            frontend_mode=frontend_mode,
            branch_a_mode=branch_a_mode,
            branch_b_mode=branch_b_mode,
            ir50_pretrained_path=ir50_pretrained_path,
            ir50_freeze=ir50_freeze,
            ir50_train_body2=ir50_train_body2,
            ir50_train_last_stage=ir50_train_last_stage,
            mobile_pretrained_path=mobile_pretrained_path,
            mobile_freeze=mobile_freeze,
        )

        # 桥梁层：负责把前端输出整理成后端能接的统一特征
        self.bridge = BridgeLayer(
            out_channels=bridge_out_channels,
            use_multi_scale_residual=bridge_use_multi_scale_residual,
            use_scale_weighting=bridge_use_scale_weighting,
        )
        self.backend_type = backend_type
        self.use_aux_head = use_aux_head

        # 后端：支持保留原 Transformer 版，或切换为更轻的 Conv+SE+GAP 版
        if backend_type == "transformer":
            self.backend = TransformerBackend(
                num_classes=num_classes,
                in_channels=bridge_out_channels,
            )
        elif backend_type == "light_conv_se_gap":
            self.backend = LightConvSEBackend(
                num_classes=num_classes,
                in_channels=bridge_out_channels,
                hidden_channels=light_backend_hidden_channels,
            )
        elif backend_type == "mamba":
            self.backend = MambaBackend(
                num_classes=num_classes,
                in_channels=bridge_out_channels,
                embed_dim=mamba_embed_dim,
                token_len=mamba_token_len,
                num_layers=mamba_num_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                use_cls=mamba_use_cls,
                version=mamba_version,
                local_kernel_size=mamba_local_kernel_size,
            )
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
        if self.use_aux_head:
            self.aux_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(bridge_out_channels, num_classes),
            )
        else:
            self.aux_head = None

    def forward(self, x):
        
        features = self.frontend(x)

        # 第二步：桥梁层做接口适配
        fused_feature_map = self.bridge(features)

        # 第三步：后端进行 token 化和 Transformer 分类
        out = self.backend(fused_feature_map)
        if self.use_aux_head and self.training:
            aux_out = self.aux_head(fused_feature_map)
            return out, aux_out

        return out

    def get_gate_balance_loss(self):
        return self.bridge.get_gate_balance_loss()