"""
data/build.py

这个文件用于构建数据集和 DataLoader。

作用：
1. 根据配置读取 RAF-DB 数据
2. 构建训练集和测试集
3. 调用 predata.py 里的预处理流程
4. 返回 train_loader 和 test_loader

来源说明：
该文件的整体职责参考王佳森师兄项目中的 data/build.py 组织方式，
同时也结合徐睿师兄项目里已经验证可运行的数据读取逻辑。
"""

import os
import random

import numpy as np
import torch
from torchvision import datasets

from .predata import build_train_transform, build_test_transform


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(
    data_path,
    batch_size=32,
    img_size=224,
    num_workers=4,
    use_weighted_sampler=False,
    use_random_erasing=True,
    tighter_crop_ratio=1.0,
    dataloader_seed=None,
):
    """
    构建训练集和测试集的 DataLoader。

    参数说明：
        data_path: 数据集总路径，例如 ./data/RAFDB
        batch_size: 每个 batch 的图片数量
        img_size: 图像输入尺寸
        num_workers: DataLoader 使用的并行读取线程数
        use_weighted_sampler: 是否对训练集启用加权采样
        use_random_erasing: 是否对训练集启用随机擦除
        tighter_crop_ratio: 是否在 aligned 图上做更紧中心裁剪

    返回：
        train_loader, test_loader
    """

    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")

    train_transform = build_train_transform(
        img_size=img_size,
        use_random_erasing=use_random_erasing,
        tighter_crop_ratio=tighter_crop_ratio,
    )
    test_transform = build_test_transform(
        img_size=img_size,
        tighter_crop_ratio=tighter_crop_ratio,
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

    train_sampler = None
    train_shuffle = True

    if use_weighted_sampler:
        targets = train_dataset.targets
        class_counts = torch.bincount(torch.tensor(targets), minlength=len(train_dataset.classes)).float()
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[torch.tensor(targets)]
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_shuffle = False

    generator = None
    worker_init_fn = None
    if dataloader_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(dataloader_seed))
        if num_workers > 0:
            worker_init_fn = _seed_worker

    train_loader = torch.utils.data.DataLoader(#数据集打包
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )

    return train_loader, test_loader