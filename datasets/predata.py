"""
predata.py

这个文件用于定义数据预处理流程。

作用：
1. 定义训练集的图像预处理
2. 定义测试集的图像预处理
3. 后续供 data/build.py 调用

来源说明：
当前预处理思路参考两个已有项目中已经验证可运行的图像处理方式，
包括 Resize、Flip、ToTensor、Normalize、RandomErasing 等。
"""

from torchvision import transforms


def build_train_transform(img_size=224, use_random_erasing=True, tighter_crop_ratio=1.0):
    """
    构建训练集预处理流程。
    """

    train_transforms = [transforms.Resize((img_size, img_size))]
    if tighter_crop_ratio < 1.0:
        crop_size = max(1, int(img_size * tighter_crop_ratio))
        train_transforms.extend([
            transforms.CenterCrop(crop_size),
            transforms.Resize((img_size, img_size)),
        ])

    train_transforms.extend([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    if use_random_erasing:
        train_transforms.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.1)))

    train_transform = transforms.Compose(train_transforms)

    return train_transform


def build_test_transform(img_size=224, tighter_crop_ratio=1.0):
    """
    构建测试集预处理流程。
    """

    test_transforms = [transforms.Resize((img_size, img_size))]
    if tighter_crop_ratio < 1.0:
        crop_size = max(1, int(img_size * tighter_crop_ratio))
        test_transforms.extend([
            transforms.CenterCrop(crop_size),
            transforms.Resize((img_size, img_size)),
        ])
    test_transforms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose(test_transforms)

    return test_transform