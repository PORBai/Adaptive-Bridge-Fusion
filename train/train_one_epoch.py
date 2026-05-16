"""
train_one_epoch.py

这个文件用于定义一轮训练的流程。

作用：
1. 遍历训练集
2. 前向传播
3. 计算损失
4. 反向传播
5. 更新参数
6. 统计训练过程中的损失和准确率

说明：与主项目 `my project/train/train_one_epoch.py` 一致，仅 `model.train()`，
不包含基线里的 BN 特殊处理。
"""

import torch


def _compute_total_loss(
    model,
    images,
    targets,
    criterion,
    gate_balance_lambda=0.0,
    aux_loss_weight=0.0,
):
    outputs = model(images)
    aux_loss = None
    if isinstance(outputs, tuple):
        outputs, aux_outputs = outputs
        aux_loss = criterion(aux_outputs, targets)

    cls_loss = criterion(outputs, targets)
    loss = cls_loss
    if aux_loss is not None:
        loss = loss + aux_loss_weight * aux_loss

    if gate_balance_lambda > 0.0 and hasattr(model, "get_gate_balance_loss"):
        gate_balance_loss = model.get_gate_balance_loss()
        if gate_balance_loss is not None:
            loss = loss + gate_balance_lambda * gate_balance_loss

    return loss, outputs


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    gate_balance_lambda=0.0,
    aux_loss_weight=0.0,
    ema=None,
):
    """
    执行一轮训练。

    参数说明：
        model: 当前模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 运行设备，例如 cuda

    返回：
        avg_loss: 当前 epoch 平均损失
        avg_acc: 当前 epoch 平均准确率
    """

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        use_sam = hasattr(optimizer, "first_step") and hasattr(optimizer, "second_step")
        if use_sam:
            optimizer.zero_grad()
            first_loss, _ = _compute_total_loss(
                model=model,
                images=images,
                targets=targets,
                criterion=criterion,
                gate_balance_lambda=gate_balance_lambda,
                aux_loss_weight=aux_loss_weight,
            )
            first_loss.backward()
            optimizer.first_step(zero_grad=True)

            second_loss, outputs = _compute_total_loss(
                model=model,
                images=images,
                targets=targets,
                criterion=criterion,
                gate_balance_lambda=gate_balance_lambda,
                aux_loss_weight=aux_loss_weight,
            )
            second_loss.backward()
            optimizer.second_step(zero_grad=True)
            loss = second_loss
        else:
            loss, outputs = _compute_total_loss(
                model=model,
                images=images,
                targets=targets,
                criterion=criterion,
                gate_balance_lambda=gate_balance_lambda,
                aux_loss_weight=aux_loss_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc
