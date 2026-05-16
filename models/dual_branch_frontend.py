import torch
import torch.nn as nn
from .ir50_branch import IR50Branch
from .mobile_branch import MobileBranch

class SimpleIRBlock(nn.Module):
    def __init__(self, channels):
        super(SimpleIRBlock, self).__init__()
        self.res_layer = nn.Sequential(#主路经过一次卷积和一次批归一化
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)#激活函数
    def forward(self, x):
        shortcut = x
        res = self.res_layer(x)
        out = res + shortcut#把做过两层卷积的特征图与原图相加
        out = self.relu(out)
        return out


class SimpleBackboneBranch(nn.Module):
    def __init__(self):
        super(SimpleBackboneBranch, self).__init__()

        # stem 先把输入变成 backbone 能处理的基础特征
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # blocks 用多个残差块模拟更像主干网络的组织方式
        self.blocks = nn.Sequential(
            SimpleIRBlock(32),
            SimpleIRBlock(32)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x


class DualBranchFrontend(nn.Module):
    def __init__(
        self,
        frontend_mode="simple",
        branch_a_mode="a_simple",
        branch_b_mode="b_simple",
        ir50_pretrained_path="",
        ir50_freeze=False,
        ir50_train_body2=False,
        ir50_train_last_stage=False,
        mobile_pretrained_path="",
        mobile_freeze=False,
    ):
        super(DualBranchFrontend, self).__init__()
        self.frontend_mode = frontend_mode
        self.branch_a_mode = branch_a_mode
        self.branch_b_mode = branch_b_mode

        # -------- A 分支：根据 BRANCH_A_MODE 决定用谁 --------
        if branch_a_mode == "a_simple":
            # A = 轻量级分支
            self.branch_a = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        elif branch_a_mode == "a_mobile_single":
            # A = Mobile 单头（只用最后一层）
            self.branch_a = MobileBranch(
                pretrained_path=mobile_pretrained_path,
                freeze=mobile_freeze,
                return_multi_scale=False,
            )
        elif branch_a_mode == "a_mobile_multi":
            # A = Mobile 多头（多尺度 x1+x2+x3）
            self.branch_a = MobileBranch(
                pretrained_path=mobile_pretrained_path,
                freeze=mobile_freeze,
                return_multi_scale=True,
            )
        else:
            raise ValueError(f"Unsupported branch_a_mode: {branch_a_mode}")

        # -------- B 分支：根据 BRANCH_B_MODE 决定用谁 --------
        if branch_b_mode == "b_simple":
            # B = 轻量级（用你原来的 SimpleBackboneBranch）
            self.branch_b = SimpleBackboneBranch()
        elif branch_b_mode == "b_ir50_single":
            # B = IR50 单头
            self.branch_b = IR50Branch(
                pretrained_path=ir50_pretrained_path,
                freeze=ir50_freeze,
                return_multi_scale=False,
                train_body2=ir50_train_body2,
                train_last_stage=ir50_train_last_stage,
            )
        elif branch_b_mode == "b_ir50_multi":
            # B = IR50 多头
            self.branch_b = IR50Branch(
                pretrained_path=ir50_pretrained_path,
                freeze=ir50_freeze,
                return_multi_scale=True,
                train_body2=ir50_train_body2,
                train_last_stage=ir50_train_last_stage,
            )
        else:
            raise ValueError(f"Unsupported branch_b_mode: {branch_b_mode}")

    def forward(self, x):
        feat_a = self.branch_a(x)
        feat_b = self.branch_b(x)

        features = {
            "branch_a": feat_a,
            "branch_b": feat_b,
        }

        return features