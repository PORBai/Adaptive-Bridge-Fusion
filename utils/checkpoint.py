"""
checkpoint.py

这个文件用于保存和加载模型 checkpoint。

作用：
1. 保存训练中间结果
2. 保存 best 模型
3. 后续支持中断后继续训练
"""

import os
import torch


def save_checkpoint(state, save_path):
    """
    保存 checkpoint。

    参数说明：
        state: 要保存的内容，通常是一个字典
        save_path: 保存路径
    """

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)


def load_checkpoint(load_path, device="cpu"):
    """
    加载 checkpoint。

    参数说明：
        load_path: checkpoint 文件路径
        device: 加载到哪个设备

    返回：
        checkpoint
    """

    checkpoint = torch.load(load_path, map_location=device)
    return checkpoint