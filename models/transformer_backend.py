"""
transformer_backend.py

这个文件定义 Transformer 后端模块。

作用：
1. 接收桥梁层输出的统一特征图
2. 将特征图整理成 token 序列
3. 加入 class token
4. 加入 position embedding
5. 送入 Transformer 编码器
6. 通过分类头输出最终类别结果

来源说明：
该模块整体思想主要参考王佳森师兄的后端结构设计。
"""

import torch
import torch.nn as nn


class TransformerBackend(nn.Module):
    """
    Transformer 后端模块。
    当前先写一个清晰的结构骨架，后续再逐步补充具体实现。
    """

    def __init__(self, num_classes=7, embed_dim=128, token_len=49, in_channels=32):
        super(TransformerBackend, self).__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.token_len = token_len
        self.in_channels = in_channels

        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, stride=1, padding=0)

        self.refine = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, token_len + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):

        #第一部分
        x = self.pool(x)
        x = self.proj(x)
        
        x = self.refine(x)

        #第二部分

        batch_size = x.shape[0]

        x = x.flatten(2).transpose(1, 2)

        x_cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((x_cls, x), dim=1)

        x = x + self.pos_embed


        #第三部分
        x = self.blocks(x)
        x = self.norm(x)

        x_cls = x[:, 0, :]
        out = self.head(x_cls)

        return out