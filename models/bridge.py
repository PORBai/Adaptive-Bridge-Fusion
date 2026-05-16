import torch
import torch.nn as nn


class BridgeLayer(nn.Module):
    """
    桥梁层：把 A 分支和 B 分支的输出整理成统一空间尺寸，再融合给后端用。

    
    - 轻量分支：32 通道特征
    - IR50 单头：256 通道
    - Mobile 单头：512 通道
    - 多头：dict，里面有 x1/x2/x3，通道数区分 IR50 (x3=256) 和 Mobile (x3=512)
    """

    def __init__(self, out_channels=32, use_multi_scale_residual=False, use_scale_weighting=False):
        super(BridgeLayer, self).__init__()
        self.last_gate_weight = None
        self.out_channels = out_channels
        self.use_multi_scale_residual = use_multi_scale_residual
        self.use_scale_weighting = use_scale_weighting

        # 32 通道分支下采样到 14x14（轻量 / SimpleBackbone）
        self.branch_32_adapter = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((14, 14))#将特征图压缩成14x14
        )

        # IR50 单尺度适配（256 → 32 通道）
        self.ir50_single_adapter = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Mobile 单尺度适配（512 → 32 通道）
        self.mobile_single_adapter = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Mobile 多尺度适配：
        # x1: out3,  通道数 64
        # x2: out4,  通道数 128
        # x3: conv_features, 通道数 512
        self.mobile_x1_adapter = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.mobile_x2_adapter = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.mobile_x3_adapter = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )


        # IR50 多尺度适配
        self.ir50_x1_adapter = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.ir50_x2_adapter = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.ir50_x3_adapter = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 多尺度分支内部仍先压回 32 通道，保证 A/B 两支在门控前维度一致。
        self.multi_scale_fuse = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 把 A(32通道) + B(32通道) 拼成 64，再压成更宽的桥接通道。
        self.fuse = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 门控融合：根据 A/B 两支的全局描述，动态生成分支权重。
        # 这里保持实验9验证过的“分支级门控”。
        self.gate_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.branch_gate = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1, stride=1, padding=0)
        )

        # 残差式桥接：在保留原始拼接融合的同时，
        # 额外用门控后的加权和生成一条“增量特征”支路。
        self.gated_residual = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 实验24：让多尺度分支的每个尺度先分别参与补偿，再汇总成最终残差。
        self.scale_residual_x1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.scale_residual_x2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.scale_residual_x3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.scale_residual_merge = nn.Sequential(
            nn.Conv2d(96, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.scale_logits = nn.Parameter(torch.zeros(3))




    def _adapt_multi_scale_branch(self, feat):
        x1 = feat["x1"]
        x2 = feat["x2"]
        x3 = feat["x3"]

        c3 = x3.shape[1]
        if c3 == 256:
            # IR50 多头
            x1 = self.ir50_x1_adapter(x1)
            x2 = self.ir50_x2_adapter(x2)
            x3 = self.ir50_x3_adapter(x3)
        elif c3 == 512:
            # Mobile 多头
            x1 = self.mobile_x1_adapter(x1)
            x2 = self.mobile_x2_adapter(x2)
            x3 = self.mobile_x3_adapter(x3)
        else:
            raise ValueError(f"Unsupported multi-scale branch with x3 channels={c3}")

        x1 = nn.functional.adaptive_avg_pool2d(x1, (14, 14))
        x2 = nn.functional.adaptive_avg_pool2d(x2, (14, 14))
        x3 = nn.functional.adaptive_avg_pool2d(x3, (14, 14))
        fused = torch.cat([x1, x2, x3], dim=1)  # (B, 96, 14, 14)

        return {
            "fused": self.multi_scale_fuse(fused),  # (B, 32, 14, 14)
            "scales": [x1, x2, x3],
        }

    def _adapt__branch(self, feat):
        """
        把一条分支的输出适配成 (B, 32, 14, 14)。
        根据输出的类型和通道数，自动识别是 轻量 / IR50 / Mobile，单头还是多头。
        """
        # 多头：dict，包含 x1/x2/x3
        if isinstance(feat, dict):
            return self._adapt_multi_scale_branch(feat)

        # 单头：tensor
        c = feat.shape[1]
        if c == 32:
            # 轻量级或 SimpleBackbone：直接当 32 通道特征下采样到 14x14
            return {"fused": self.branch_32_adapter(feat), "scales": None}
        elif c == 256:
            # IR50 单头：256 → 32
            x = self.ir50_single_adapter(feat)
            return {"fused": nn.functional.adaptive_avg_pool2d(x, (14, 14)), "scales": None}
        elif c == 512:
            # Mobile 单头：512 → 32
            x = self.mobile_single_adapter(feat)
            return {"fused": nn.functional.adaptive_avg_pool2d(x, (14, 14)), "scales": None}
        else:
            raise ValueError(f"Unsupported single-scale branch with channels={c}")

    def forward(self, features):
    
        branch_a = self._adapt__branch(features["branch_a"])
        branch_b = self._adapt__branch(features["branch_b"])
        branch_a_fused = branch_a["fused"]
        branch_b_fused = branch_b["fused"]

        # 先看两支的整体响应，再自适应决定每一支在当前样本里的权重。
        gate_input = torch.cat([branch_a_fused, branch_b_fused], dim=1)   # (B, 64, 14, 14)
        gate_input = self.gate_pool(gate_input)               # (B, 64, 1, 1)
        gate_logits = self.branch_gate(gate_input)            # (B, 2, 1, 1)
        gate_weight = torch.softmax(gate_logits, dim=1)
        self.last_gate_weight = gate_weight

        weight_a = gate_weight[:, 0:1, :, :]
        weight_b = gate_weight[:, 1:2, :, :]

        # 主路：保留原来的固定拼接融合，避免门控改动过大导致原始信息被破坏。
        fused = torch.cat([branch_a_fused, branch_b_fused], dim=1)#把两条分支拼起来后的特征
        fused_feature_map = self.fuse(fused)#压缩整理成统一的桥接通道特征

        # 实验24：如果只有一条分支是多尺度，就让该分支各尺度分别参与补偿，再统一汇总。
        if self.use_multi_scale_residual and branch_a["scales"] is None and branch_b["scales"] is not None:
            scale_residuals = []
            scale_weights = None
            if self.use_scale_weighting:
                scale_weights = torch.softmax(self.scale_logits, dim=0)
            for idx, (scale_feat, scale_block) in enumerate(zip(
                branch_b["scales"],
                [self.scale_residual_x1, self.scale_residual_x2, self.scale_residual_x3],
            )):
                scale_pair = torch.cat([branch_a_fused * weight_a, scale_feat * weight_b], dim=1)
                scale_residual = scale_block(scale_pair)
                if scale_weights is not None:
                    scale_residual = scale_residual * scale_weights[idx]
                scale_residuals.append(scale_residual)
            merged_residual = self.scale_residual_merge(torch.cat(scale_residuals, dim=1))
            fused_feature_map = fused_feature_map + merged_residual
        elif self.use_multi_scale_residual and branch_b["scales"] is None and branch_a["scales"] is not None:
            scale_residuals = []
            scale_weights = None
            if self.use_scale_weighting:
                scale_weights = torch.softmax(self.scale_logits, dim=0)
            for idx, (scale_feat, scale_block) in enumerate(zip(
                branch_a["scales"],
                [self.scale_residual_x1, self.scale_residual_x2, self.scale_residual_x3],
            )):
                scale_pair = torch.cat([scale_feat * weight_a, branch_b_fused * weight_b], dim=1)
                scale_residual = scale_block(scale_pair)
                if scale_weights is not None:
                    scale_residual = scale_residual * scale_weights[idx]
                scale_residuals.append(scale_residual)
            merged_residual = self.scale_residual_merge(torch.cat(scale_residuals, dim=1))
            fused_feature_map = fused_feature_map + merged_residual
        else:
            # 默认支路：门控后做加权和，作为桥接模块自己的“动态补偿”信息。
            gated_sum = branch_a_fused * weight_a + branch_b_fused * weight_b
            gated_residual = self.gated_residual(gated_sum)
            fused_feature_map = fused_feature_map + gated_residual

        return fused_feature_map

    def get_gate_balance_loss(self):
        """
        轻量门控均衡正则：
        只约束一个 batch 的平均门控不要长期塌到单分支，
        不强迫每个样本都平均分配。
        """
        if self.last_gate_weight is None:
            return None

        mean_gate = self.last_gate_weight.mean(dim=(0, 2, 3))
        target_gate = torch.full_like(mean_gate, 0.5)
        return torch.mean((mean_gate - target_gate) ** 2)
  