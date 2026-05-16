"""
evaluate.py

这个文件用于定义验证/测试流程。

作用：
1. 遍历测试集
2. 前向传播
3. 计算损失和准确率
4. 不更新模型参数
5. 可选导出每个测试样本的 y_true / y_pred 明细
"""

import csv
import os

import torch


def evaluate(model, test_loader, criterion, device, save_details_path=None, use_tta=False):
    """
    执行一次验证/测试。

    参数说明：
        model: 当前模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 运行设备

    返回：
        avg_loss: 平均损失
        avg_acc: 平均准确率
    """

    model.eval() #切换到测试模式

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    sample_offset = 0
    details = []

    dataset = test_loader.dataset
    samples = getattr(dataset, "samples", None)
    classes = getattr(dataset, "classes", None)

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            if use_tta:
                flipped_images = torch.flip(images, dims=[3])
                outputs_flip = model(flipped_images)
                outputs = (outputs + outputs_flip) / 2.0
            loss = criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += images.size(0)

            if save_details_path is not None:
                batch_size = images.size(0)
                preds_cpu = preds.cpu().tolist()
                targets_cpu = targets.cpu().tolist()

                for i in range(batch_size):
                    sample_path = ""
                    if samples is not None:
                        sample_path = samples[sample_offset + i][0]

                    y_true = targets_cpu[i]
                    y_pred = preds_cpu[i]
                    details.append(
                        {
                            "sample_path": sample_path,
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "y_true_name": classes[y_true] if classes is not None else "",
                            "y_pred_name": classes[y_pred] if classes is not None else "",
                            "correct": int(y_true == y_pred),
                        }
                    )

                sample_offset += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    if save_details_path is not None:
        os.makedirs(os.path.dirname(save_details_path), exist_ok=True)
        with open(save_details_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sample_path",
                    "y_true",
                    "y_pred",
                    "y_true_name",
                    "y_pred_name",
                    "correct",
                ],
            )
            writer.writeheader()
            writer.writerows(details)

    return avg_loss, avg_acc