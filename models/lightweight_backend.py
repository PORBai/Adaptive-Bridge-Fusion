import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.shape
        scale = self.pool(x).view(batch_size, channels)
        scale = self.fc(scale).view(batch_size, channels, 1, 1)
        return x * scale


class LightConvSEBackend(nn.Module):
    """
    轻量后端：
    Conv -> BN/ReLU -> Conv -> BN/ReLU -> SE -> GAP -> Linear
    """

    def __init__(self, num_classes=7, in_channels=32, hidden_channels=64):
        super(LightConvSEBackend, self).__init__()
        self.hidden_channels = hidden_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(hidden_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.se(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
