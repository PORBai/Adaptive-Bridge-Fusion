import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class ResidualMambaBlock(nn.Module):
    def __init__(self, embed_dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4.0):
        super(ResidualMambaBlock, self).__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is not installed. Please run: pip install mamba-ssm"
            )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mamba = Mamba(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BiMambaBlock(nn.Module):
    def __init__(self, embed_dim, d_state=16, d_conv=4, expand=2, mlp_ratio=4.0):
        super(BiMambaBlock, self).__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is not installed. Please run: pip install mamba-ssm"
            )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mamba_fwd = Mamba(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.fuse = nn.Linear(embed_dim * 2, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x, height=None, width=None):
        x_norm = self.norm1(x)
        x_fwd = self.mamba_fwd(x_norm)
        x_bwd = torch.flip(self.mamba_bwd(torch.flip(x_norm, dims=[1])), dims=[1])
        x = x + self.fuse(torch.cat([x_fwd, x_bwd], dim=-1))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBiMambaBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        d_state=16,
        d_conv=4,
        expand=2,
        mlp_ratio=4.0,
        local_kernel_size=3,
    ):
        super(ConvBiMambaBlock, self).__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is not installed. Please run: pip install mamba-ssm"
            )

        self.norm_local = nn.LayerNorm(embed_dim)
        self.local_dwconv = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=local_kernel_size,
            stride=1,
            padding=local_kernel_size // 2,
            groups=embed_dim,
        )
        self.local_pwconv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.local_act = nn.GELU()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.mamba_fwd = Mamba(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba_bwd = Mamba(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.fuse = nn.Linear(embed_dim * 2, embed_dim)

        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x, height=None, width=None):
        if height is None or width is None:
            raise ValueError("ConvBiMambaBlock requires height and width")

        batch_size, _, channels = x.shape
        x_local = self.norm_local(x)
        x_local = x_local.transpose(1, 2).reshape(batch_size, channels, height, width)
        x_local = self.local_pwconv(self.local_act(self.local_dwconv(x_local)))
        x_local = x_local.flatten(2).transpose(1, 2)
        x = x + x_local

        x_norm = self.norm1(x)
        x_fwd = self.mamba_fwd(x_norm)
        x_bwd = torch.flip(self.mamba_bwd(torch.flip(x_norm, dims=[1])), dims=[1])
        x = x + self.fuse(torch.cat([x_fwd, x_bwd], dim=-1))
        x = x + self.mlp(self.norm2(x))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionPooling, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(self.norm(x)), dim=1)
        return (x * weights).sum(dim=1)


class MambaBackend(nn.Module):
    """
    Mamba 后端模块。

    结构：
    bridge feature map -> pool -> proj -> conv refine -> token sequence
    -> Mamba blocks -> pooling -> classifier
    """

    def __init__(
        self,
        num_classes=7,
        in_channels=32,
        embed_dim=128,
        token_len=49,
        num_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        use_cls=False,
        version="v1_plain",
        local_kernel_size=3,
    ):
        super(MambaBackend, self).__init__()

        self.use_cls = use_cls
        self.embed_dim = embed_dim
        self.token_len = token_len
        self.version = version

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0)
        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        seq_len = token_len + (1 if use_cls else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        if version == "v1_plain":
            block_cls = ResidualMambaBlock
            block_kwargs = {}
        elif version in {"v2_bi", "v3_bi_attn"}:
            block_cls = BiMambaBlock
            block_kwargs = {}
        elif version == "v4_conv_bi_attn":
            block_cls = ConvBiMambaBlock
            block_kwargs = {"local_kernel_size": local_kernel_size}
        else:
            raise ValueError(f"Unsupported Mamba version: {version}")

        self.blocks = nn.ModuleList(
            [
                block_cls(
                    embed_dim=embed_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **block_kwargs,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_pool = AttentionPooling(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.pool(x)
        x = self.proj(x)
        x = self.refine(x)
        _, _, height, width = x.shape

        x = x.flatten(2).transpose(1, 2)

        if self.use_cls:
            batch_size = x.shape[0]
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + self.pos_embed[:, : x.size(1), :]

        for block in self.blocks:
            if self.version == "v4_conv_bi_attn":
                x = block(x, height=height, width=width)
            else:
                x = block(x)

        x = self.norm(x)
        if self.use_cls:
            x = x[:, 0, :]
        elif self.version in {"v3_bi_attn", "v4_conv_bi_attn"}:
            x = self.attn_pool(x)
        else:
            x = x.mean(dim=1)

        return self.head(x)
