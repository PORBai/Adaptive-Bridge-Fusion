import argparse
from collections import Counter
import os

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from datasets.build import build_dataloader
from models.build import build_model
from train.train_one_epoch import train_one_epoch
from train.evaluate import evaluate
from utils.logger import make_log_file, print_and_log
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.ema import ModelEMA
from utils.sam import SAM
from utils.seed import set_seed


def load_config(config_path):
    """
    读取 YAML 配置文件。
    """

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def compute_class_weights(train_loader, num_classes):
    """
    按训练集各类样本数计算 CrossEntropyLoss 的 class weight。
    w_i = N / (K * n_i)，与 sklearn class_weight='balanced' 一致。
    """
    ds = train_loader.dataset
    if hasattr(ds, "targets"):
        targets = ds.targets
    else:
        targets = [s[1] for s in ds.samples]
    counts = Counter(targets)
    class_sample_count = torch.tensor(
        [counts[i] for i in range(num_classes)], dtype=torch.float32
    )
    if (class_sample_count == 0).any():
        raise ValueError("某类别在训练集中样本数为 0，无法计算 class weight")
    n_samples = class_sample_count.sum()
    weight = n_samples / (num_classes * class_sample_count)
    return weight


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/abfnet_rafdb.yaml",
        help="训练配置文件路径",
    )
    return parser.parse_args()


def build_param_groups(model, config):
    base_lr = float(config["TRAIN"]["LR"])
    ir50_body2_lr = float(config["TRAIN"].get("IR50_BODY2_LR", base_lr))
    ir50_last_stage_lr = float(config["TRAIN"].get("IR50_LAST_STAGE_LR", base_lr))
    train_body2 = config["MODEL"].get("IR50_TRAIN_BODY2", False)
    train_last_stage = config["MODEL"].get("IR50_TRAIN_LAST_STAGE", False)
    body2_prefix = "frontend.branch_b.backbone.body2"
    last_stage_prefix = "frontend.branch_b.backbone.body3"

    other_params = []
    ir50_body2_params = []
    ir50_last_stage_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if train_body2 and name.startswith(body2_prefix):
            ir50_body2_params.append(param)
        elif train_last_stage and name.startswith(last_stage_prefix):
            ir50_last_stage_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    group_logs = []

    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})
        group_logs.append(f"other_trainable={len(other_params)} params @ {base_lr}")
    if train_body2 and ir50_body2_params:
        param_groups.append({"params": ir50_body2_params, "lr": ir50_body2_lr})
        group_logs.append(f"ir50_body2={len(ir50_body2_params)} params @ {ir50_body2_lr}")
    if train_last_stage and ir50_last_stage_params:
        param_groups.append({"params": ir50_last_stage_params, "lr": ir50_last_stage_lr})
        group_logs.append(f"ir50_body3={len(ir50_last_stage_params)} params @ {ir50_last_stage_lr}")

    if not param_groups:
        raise ValueError("No trainable parameters found for optimizer.")

    return param_groups, group_logs


def build_optimizer(model, config, log_path):
    """
    对 IR50 高层 stage 使用更小学习率，其余可训练模块保持基础学习率。
    支持普通优化器和 SAM。
    """
    base_lr = float(config["TRAIN"]["LR"])
    optimizer_name = str(config["TRAIN"].get("OPTIMIZER", "adam")).lower()
    use_sam = config["TRAIN"].get("USE_SAM", False)
    sam_rho = float(config["TRAIN"].get("SAM_RHO", 0.05))
    sam_adaptive = config["TRAIN"].get("SAM_ADAPTIVE", False)
    momentum = float(config["TRAIN"].get("MOMENTUM", 0.9))
    weight_decay = float(config["TRAIN"].get("WEIGHT_DECAY", 0.0))

    param_groups, group_logs = build_param_groups(model, config)

    if optimizer_name == "adam":
        base_optimizer_cls = optim.Adam
    elif optimizer_name == "adamw":
        base_optimizer_cls = optim.AdamW
    elif optimizer_name == "sgd":
        base_optimizer_cls = optim.SGD
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    optimizer_kwargs = {
        "lr": base_lr,
        "weight_decay": weight_decay,
    }
    if optimizer_name == "sgd":
        optimizer_kwargs["momentum"] = momentum

    if use_sam:
        optimizer = SAM(
            param_groups,
            base_optimizer_cls,
            rho=sam_rho,
            adaptive=sam_adaptive,
            **optimizer_kwargs,
        )
        print_and_log(
            log_path,
            "Optimizer: SAM("
            f"{optimizer_name}, rho={sam_rho}, adaptive={sam_adaptive}, weight_decay={weight_decay})",
        )
    else:
        optimizer = base_optimizer_cls(param_groups, **optimizer_kwargs)
        print_and_log(
            log_path,
            f"Optimizer: {optimizer_name}(weight_decay={weight_decay})",
        )

    print_and_log(log_path, "Optimizer groups: " + ", ".join(group_logs))
    return optimizer


def build_scheduler(optimizer, config, log_path):
    scheduler_name = str(config["TRAIN"].get("LR_SCHEDULER", "none")).lower()
    if scheduler_name in {"", "none"}:
        print_and_log(log_path, "LR scheduler: none")
        return None
    if scheduler_name == "exponential":
        gamma = float(config["TRAIN"].get("LR_GAMMA", 0.98))
        print_and_log(log_path, f"LR scheduler: exponential(gamma={gamma})")
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    raise ValueError(f"Unsupported LR scheduler: {scheduler_name}")


def format_learning_rates(optimizer):
    return ", ".join(
        f"group{idx}={group['lr']:.8f}" for idx, group in enumerate(optimizer.param_groups)
    )


def count_parameters(module):
    total_params = sum(param.numel() for param in module.parameters())
    trainable_params = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total_params, trainable_params


def main():
    """
    项目主函数。
    """

    args = parse_args()
    config_path = args.config
    config = load_config(config_path)
    project_name = config["SYSTEM"]["PROJECT_NAME"]
    experiment_name = config["SYSTEM"]["EXPERIMENT_NAME"]
    resume = config["TRAIN"]["RESUME"]
    eval_only = config["TEST"]["EVAL_ONLY"]
    use_tta = config["TEST"].get("USE_TTA", False)
    checkpoint_experiment = config["TEST"].get("CHECKPOINT_EXPERIMENT", experiment_name)
    save_dir = config["SYSTEM"]["SAVE_DIR"]
    batch_size = config["DATA"]["BATCH_SIZE"]
    img_size = config["DATA"]["IMG_SIZE"]
    tighter_crop_ratio = float(config["DATA"].get("TIGHTER_CROP_RATIO", 1.0))
    lr = config["TRAIN"]["LR"]
    epochs = config["TRAIN"]["EPOCHS"]
    use_weighted_sampler = config["TRAIN"].get("USE_WEIGHTED_SAMPLER", False)
    use_random_erasing = config["TRAIN"].get("USE_RANDOM_ERASING", True)
    gate_balance_lambda = float(config["TRAIN"].get("GATE_BALANCE_LAMBDA", 0.0))
    label_smoothing = float(config["TRAIN"].get("LABEL_SMOOTHING", 0.0))
    aux_loss_weight = float(config["TRAIN"].get("AUX_LOSS_WEIGHT", 0.0))
    use_ema = config["TRAIN"].get("USE_EMA", False)
    ema_decay = float(config["TRAIN"].get("EMA_DECAY", 0.999))
    early_stopping_patience = int(config["TRAIN"].get("EARLY_STOPPING_PATIENCE", 0))
    early_stopping_min_delta = float(config["TRAIN"].get("EARLY_STOPPING_MIN_DELTA", 0.0))
    experiment_dir = os.path.join(save_dir, experiment_name)
    log_dir = os.path.join(experiment_dir, "log")
    log_path = make_log_file(log_dir)
    metrics_path = os.path.join(experiment_dir, "metrics.txt")#指标记录文件路径
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    prediction_dir = os.path.join(experiment_dir, "predictions")
    
    with open(config_save_path, "w", encoding="utf-8") as f:#以写入模式打开目标 YAML 文件，并把配置内容写入文件
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
    train_seed = int(config["TRAIN"].get("SEED", 42))
    deterministic = bool(config["TRAIN"].get("DETERMINISTIC", False))
    set_seed(train_seed, deterministic=deterministic)
    print_and_log(
        log_path,
        f"Random seed: {train_seed}, deterministic_algorithms: {deterministic}",
    )
    if not (resume and os.path.exists(metrics_path) and not eval_only):
        with open(metrics_path, "w", encoding="utf-8") as f:#以写入模式打开目标文本文件，并把指标记录写入文件
            f.write("Epoch\tTrain Loss\tTrain Acc\tTest Loss\tTest Acc\n")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoint")
    best_acc = 0.0
    best_epoch = 0
    no_improve_count = 0

    device = config["TRAIN"]["DEVICE"]
    
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print_and_log(log_path, f"Experiment directory: {experiment_dir}")
    print_and_log(log_path, f"Project name: {project_name}")
    print_and_log(log_path, f"Experiment name: {experiment_name}")
    print_and_log(log_path, f"Current device: {device}")
    print_and_log(log_path, f"Batch size: {batch_size}")
    print_and_log(log_path, f"Image size: {img_size}")
    print_and_log(log_path, f"Tighter crop ratio: {tighter_crop_ratio}")
    print_and_log(log_path, f"Learning rate: {lr}")
    print_and_log(log_path, f"Epochs: {epochs}")
    print_and_log(log_path, f"Use weighted sampler: {use_weighted_sampler}")
    print_and_log(log_path, f"Use random erasing: {use_random_erasing}")
    print_and_log(log_path, f"Gate balance lambda: {gate_balance_lambda}")
    print_and_log(log_path, f"Label smoothing: {label_smoothing}")
    print_and_log(log_path, f"Aux loss weight: {aux_loss_weight}")
    print_and_log(log_path, f"Use EMA: {use_ema}")
    print_and_log(log_path, f"EMA decay: {ema_decay}")
    print_and_log(log_path, f"Early stopping patience: {early_stopping_patience}")
    print_and_log(log_path, f"Early stopping min delta: {early_stopping_min_delta}")
    print_and_log(log_path, "Loading data...")
    print_and_log(log_path, f"Eval only mode: {eval_only}")
    print_and_log(log_path, f"TTA mode: {use_tta}")
    print_and_log(log_path, f"Resume mode: {resume}")

    train_loader, test_loader = build_dataloader(#进入data/build
        data_path=config["DATA"]["DATA_PATH"],
        batch_size=config["DATA"]["BATCH_SIZE"],
        img_size=config["DATA"]["IMG_SIZE"],
        num_workers=config["DATA"]["NUM_WORKERS"],
        use_weighted_sampler=use_weighted_sampler,
        use_random_erasing=use_random_erasing,
        tighter_crop_ratio=tighter_crop_ratio,
        dataloader_seed=train_seed,
    )

    print_and_log(log_path, "Building model...")
    model = build_model(config)#建模型进入modle/build
    model = model.to(device)
    total_params, trainable_params = count_parameters(model)
    print_and_log(log_path, f"Model total params: {total_params}")
    print_and_log(log_path, f"Model trainable params: {trainable_params}")
    if hasattr(model, "backend"):
        backend_total_params, backend_trainable_params = count_parameters(model.backend)
        print_and_log(log_path, f"Backend total params: {backend_total_params}")
        print_and_log(log_path, f"Backend trainable params: {backend_trainable_params}")
# 交叉熵与优化器：与主项目 formal-main 一致（Adam，无 weight_decay / 无 Cosine）
    use_class_weight = config["TRAIN"].get("USE_CLASS_WEIGHT", False)
    if use_class_weight:
        num_classes = int(config["DATA"]["NUM_CLASSES"])
        class_weights = compute_class_weights(train_loader, num_classes).to(device)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        print_and_log(
            log_path,
            "USE_CLASS_WEIGHT=true（实验6），各类权重 0..K-1: "
            + ", ".join(f"{w:.4f}" for w in class_weights.tolist()),
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = build_optimizer(model, config, log_path)
    scheduler = build_scheduler(optimizer, config, log_path)
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    
    start_epoch = 0#开始训练的轮次，默认从0开始
    if resume and not eval_only:
        latest_path = os.path.join(checkpoint_dir, "latest.pth")
        if os.path.exists(latest_path):
            checkpoint = load_checkpoint(latest_path, device=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if ema is not None and "ema_state_dict" in checkpoint:
                ema.load_state_dict(checkpoint["ema_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_acc = checkpoint.get("best_acc", 0.0)
            best_epoch = checkpoint.get("best_epoch", 0)
            no_improve_count = checkpoint.get("no_improve_count", 0)

            print_and_log(log_path, f"Resumed training from: {latest_path}")
            print_and_log(log_path, f"Start epoch: {start_epoch + 1}")
            print_and_log(log_path, f"Best accuracy so far: {best_acc:.4f} @ epoch {best_epoch}")
        else:
            print_and_log(log_path, f"Resume checkpoint not found: {latest_path}")
            print_and_log(log_path, "Training will start from scratch.")
            print_and_log(log_path, f"Best accuracy so far: {best_acc:.4f}")
    
    
   
    if eval_only:
        
        print_and_log(log_path, "Evaluating model...")#继承最好的权重直接训练
        best_path = os.path.join(save_dir, checkpoint_experiment, "checkpoint", "best.pth")
        if not os.path.exists(best_path):
            print_and_log(log_path, f"Checkpoint not found: {best_path}")
            print_and_log(log_path, "Please train the model first or check the experiment path.")
            return

        checkpoint = load_checkpoint(best_path, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if ema is not None and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
            ema.apply_shadow(model)
        print_and_log(log_path, f"Loaded checkpoint from: {best_path}")
        print_and_log(log_path, f"Checkpoint epoch: {checkpoint['epoch']}")
        print_and_log(log_path, f"Checkpoint best accuracy: {checkpoint.get('best_acc', 0.0):.4f}")

        test_loss, test_acc = evaluate(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            save_details_path=os.path.join(prediction_dir, "eval_only_details.csv"),
            use_tta=use_tta,
        )
        print_and_log(log_path, f"Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.4f}")
        print_and_log(log_path, f"Prediction details saved to: {os.path.join(prediction_dir, 'eval_only_details.csv')}")
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"eval_only\t-\t-\t{test_loss:.4f}\t{test_acc:.4f}\n")
        if ema is not None and "ema_state_dict" in checkpoint:
            ema.restore(model)
       


    else:
        for epoch in range(start_epoch, epochs):
            print_and_log(log_path, f"Epoch [{epoch + 1}/{epochs}]")
            print_and_log(log_path, f"Current LR(s): {format_learning_rates(optimizer)}")
            train_loss, train_acc = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                gate_balance_lambda=gate_balance_lambda,
                aux_loss_weight=aux_loss_weight,
                ema=ema,
            )
            ema_applied = False
            if ema is not None:
                ema.apply_shadow(model)
                ema_applied = True
            test_loss, test_acc = evaluate(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device
            )
            if ema_applied:
                ema.restore(model)
            if scheduler is not None:
                scheduler.step()
            print_and_log(log_path, f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print_and_log(log_path, f"Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.4f}")
            with open(metrics_path, "a", encoding="utf-8") as f:#以追加模式打开目标文本文件，并把指标记录写入文件
                f.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_acc:.4f}\t{test_loss:.4f}\t{test_acc:.4f}\n")

            improved = test_acc > (best_acc + early_stopping_min_delta)
            if improved:
                best_acc = test_acc
                best_epoch = epoch + 1
                no_improve_count = 0
            else:
                no_improve_count += 1

            latest_path = os.path.join(checkpoint_dir, "latest.pth")
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "best_acc": best_acc,
                    "best_epoch": best_epoch,
                    "no_improve_count": no_improve_count,
                    "ema_state_dict": ema.state_dict() if ema is not None else None,
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                },
                latest_path
            )
            print_and_log(log_path, f"Latest checkpoint saved to: {latest_path}")#打印最新检查点保存路径
            print_and_log(log_path, f"Best accuracy so far: {best_acc:.4f} @ epoch {best_epoch}")
           
            if improved:
                best_path = os.path.join(checkpoint_dir, "best.pth")
                
                
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "test_loss": test_loss,
                        "test_acc": test_acc,
                        "best_acc": best_acc,
                        "best_epoch": best_epoch,
                        "no_improve_count": no_improve_count,
                        "ema_state_dict": ema.state_dict() if ema is not None else None,
                        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                        
                    },
                    best_path
                )
                print_and_log(log_path, f"New best accuracy: {best_acc:.4f}")
                print_and_log(log_path, f"Best checkpoint saved to: {best_path}")
                best_detail_path = os.path.join(prediction_dir, "best_details.csv")
                if ema is not None:
                    ema.apply_shadow(model)
                evaluate(
                    model=model,
                    test_loader=test_loader,
                    criterion=criterion,
                    device=device,
                    save_details_path=best_detail_path,
                )
                if ema is not None:
                    ema.restore(model)
                print_and_log(log_path, f"Best prediction details saved to: {best_detail_path}")

            if early_stopping_patience > 0 and no_improve_count >= early_stopping_patience:
                print_and_log(
                    log_path,
                    "Early stopping triggered: "
                    f"no improvement for {no_improve_count} epochs (best={best_acc:.4f} @ epoch {best_epoch}).",
                )
                break


if __name__ == "__main__":
    main()