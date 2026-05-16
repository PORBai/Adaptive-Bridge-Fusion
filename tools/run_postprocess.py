import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="One-click paper-style postprocess for Baiyr experiments.")
    parser.add_argument(
        "--config",
        default="configs/abfnet_rafdb.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        help="Experiment name override. Default reads from config.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI (paper-ready suggested: 300+).",
    )
    parser.add_argument(
        "--confusion-title",
        default=None,
        help=(
            "Substring inside the confusion matrix title after 'Confusion Matrix ('. "
            "Default: experiment name. Example: --confusion-title RAF-DB "
            "yields title 'Confusion Matrix (RAF-DB)' (ASCII recommended for portable fonts)."
        ),
    )
    parser.add_argument(
        "--curve-title",
        default=None,
        help=(
            "Substring inside the recognition-rate / loss curve titles, same as confusion-title "
            "parenthetical. Default: same as --confusion-title (or experiment name if both unset)."
        ),
    )
    parser.add_argument(
        "--only-confusion",
        action="store_true",
        help="Only read best_details.csv and write confusion matrix PNG/PDF + CM/report CSVs (no curves, no summary.json).",
    )
    return parser.parse_args()


def set_paper_style():
    plt.rcParams.update(
        {
            # Latin-only titles render reliably on common paper fonts (avoid CJK tofu on some hosts)
            "font.family": "DejaVu Serif",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def read_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_paths(config, exp_name_override=None):
    exp_name = exp_name_override or str(config["SYSTEM"]["EXPERIMENT_NAME"])
    root = os.path.join(config["SYSTEM"]["SAVE_DIR"], exp_name)
    paths = {
        "exp_name": exp_name,
        "root": root,
        "metrics": os.path.join(root, "metrics.txt"),
        "details": os.path.join(root, "predictions", "best_details.csv"),
        "out": os.path.join(root, "reports", "paper"),
    }
    os.makedirs(paths["out"], exist_ok=True)
    return paths


def find_first_existing(candidates):
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def discover_latest_file(root_dir, filename):
    latest_path = None
    latest_mtime = -1.0
    if not os.path.exists(root_dir):
        return None
    for dirpath, _, files in os.walk(root_dir):
        if filename in files:
            candidate = os.path.join(dirpath, filename)
            mtime = os.path.getmtime(candidate)
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = candidate
    return latest_path


def load_metrics(metrics_path):
    df = pd.read_csv(metrics_path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Epoch"].astype(str).str.isdigit()].copy()
    df["Epoch"] = df["Epoch"].astype(int)
    num_cols = ["Train Loss", "Train Acc", "Test Loss", "Test Acc"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=num_cols)
    return df


def plot_curves(df_metrics, out_dir, title_suffix, dpi):
    epochs = df_metrics["Epoch"].values
    tr_acc = df_metrics["Train Acc"].values
    te_acc = df_metrics["Test Acc"].values
    tr_loss = df_metrics["Train Loss"].values
    te_loss = df_metrics["Test Loss"].values

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(epochs, tr_acc, linewidth=2.0, label="Train Accuracy")
    ax.plot(epochs, te_acc, linewidth=2.0, label="Test Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Recognition Rate Curve ({title_suffix})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(os.path.join(out_dir, "curve_accuracy.png"), dpi=dpi)
    fig.savefig(os.path.join(out_dir, "curve_accuracy.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(epochs, tr_loss, linewidth=2.0, label="Train Loss")
    ax.plot(epochs, te_loss, linewidth=2.0, label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curve ({title_suffix})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(os.path.join(out_dir, "curve_loss.png"), dpi=dpi)
    fig.savefig(os.path.join(out_dir, "curve_loss.pdf"))
    plt.close(fig)


def _ordered_class_names(df_details) -> List[str]:
    if "y_true_name" in df_details.columns:
        tmp = (
            df_details[["y_true", "y_true_name"]]
            .drop_duplicates(subset=["y_true"])
            .sort_values("y_true")
        )
        return tmp["y_true_name"].astype(str).tolist()
    label_ids = sorted(df_details["y_true"].unique().tolist())
    return [str(i) for i in label_ids]


def build_tables(df_details, out_dir):
    y_true = df_details["y_true"].astype(int).values
    y_pred = df_details["y_pred"].astype(int).values
    label_ids = sorted(np.unique(y_true).tolist())
    class_names = _ordered_class_names(df_details)

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    cm_norm = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cmn_df = pd.DataFrame(cm_norm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix_counts.csv"), encoding="utf-8-sig")
    cmn_df.to_csv(os.path.join(out_dir, "confusion_matrix_normalized.csv"), encoding="utf-8-sig")

    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(out_dir, "classification_report.csv"), encoding="utf-8-sig")

    # Per-class accuracy table
    class_total = cm.sum(axis=1)
    class_correct = np.diag(cm)
    class_acc = class_correct / np.clip(class_total, 1, None)
    class_acc_df = pd.DataFrame(
        {
            "class_id": label_ids,
            "class_name": class_names,
            "correct": class_correct,
            "total": class_total,
            "accuracy": class_acc,
        }
    )
    avg_acc = float((y_true == y_pred).mean())
    class_acc_df.loc[len(class_acc_df)] = [-1, "average", class_correct.sum(), class_total.sum(), avg_acc]
    class_acc_df.to_csv(os.path.join(out_dir, "class_accuracy_table.csv"), index=False, encoding="utf-8-sig")

    latex_df = class_acc_df.copy()
    latex_df["accuracy"] = latex_df["accuracy"].map(lambda x: f"{x * 100:.2f}")
    latex_df.to_latex(
        os.path.join(out_dir, "class_accuracy_table.tex"),
        index=False,
        escape=False,
        caption="Per-class accuracy and average accuracy.",
        label="tab:class_acc",
    )

    return cm, cm_norm, class_names, avg_acc


def plot_confusion(cm, cm_norm, class_names, out_dir, title_suffix, dpi):
    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Normalized", rotation=90)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix ({title_suffix})")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"
            color = "white" if cm_norm[i, j] > 0.45 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix_paper.png"), dpi=dpi)
    fig.savefig(os.path.join(out_dir, "confusion_matrix_paper.pdf"))
    plt.close(fig)


def write_summary_json(df_metrics, avg_acc, out_dir):
    best_idx = df_metrics["Test Acc"].idxmax()
    summary = {
        "best_epoch": int(df_metrics.loc[best_idx, "Epoch"]),
        "best_test_acc": float(df_metrics.loc[best_idx, "Test Acc"]),
        "best_test_loss": float(df_metrics.loc[best_idx, "Test Loss"]),
        "last_epoch": int(df_metrics["Epoch"].iloc[-1]),
        "last_test_acc": float(df_metrics["Test Acc"].iloc[-1]),
        "last_test_loss": float(df_metrics["Test Loss"].iloc[-1]),
        "avg_accuracy_from_details": float(avg_acc),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    set_paper_style()
    config = read_config(args.config)
    paths = prepare_paths(config, args.exp_name)
    confusion_title = args.confusion_title if args.confusion_title is not None else paths["exp_name"]
    curve_title = args.curve_title if args.curve_title is not None else confusion_title

    details_path = find_first_existing(
        [
            paths["details"],
            discover_latest_file(paths["root"], "best_details.csv"),
            discover_latest_file(config["SYSTEM"]["SAVE_DIR"], "best_details.csv"),
        ]
    )
    if details_path is None:
        raise FileNotFoundError(
            "best_details.csv not found. Please ensure best checkpoint evaluation finished "
            "and predictions were exported."
        )

    df_details = pd.read_csv(details_path)
    cm, cm_norm, class_names, avg_acc = build_tables(df_details, paths["out"])
    plot_confusion(cm, cm_norm, class_names, paths["out"], confusion_title, args.dpi)

    if args.only_confusion:
        print(f"Confusion matrix + tables saved to: {paths['out']}")
        print(f"Using details: {details_path}")
        print("Generated files:")
        print("- confusion_matrix_paper.png/.pdf")
        print("- confusion_matrix_counts.csv")
        print("- confusion_matrix_normalized.csv")
        print("- classification_report.csv")
        print("- class_accuracy_table.csv/.tex")
        return

    metrics_path = find_first_existing(
        [
            paths["metrics"],
            discover_latest_file(paths["root"], "metrics.txt"),
            discover_latest_file(config["SYSTEM"]["SAVE_DIR"], "metrics.txt"),
        ]
    )
    if metrics_path is None:
        raise FileNotFoundError(
            "metrics.txt not found. Please ensure training produced it, "
            "or pass correct --config/--exp-name."
        )

    df_metrics = load_metrics(metrics_path)
    plot_curves(df_metrics, paths["out"], curve_title, args.dpi)
    write_summary_json(df_metrics, avg_acc, paths["out"])

    print(f"Paper-style outputs saved to: {paths['out']}")
    print(f"Using metrics: {metrics_path}")
    print(f"Using details: {details_path}")
    print("Generated files:")
    print("- curve_accuracy.png/.pdf")
    print("- curve_loss.png/.pdf")
    print("- confusion_matrix_paper.png/.pdf")
    print("- confusion_matrix_counts.csv")
    print("- confusion_matrix_normalized.csv")
    print("- classification_report.csv")
    print("- class_accuracy_table.csv/.tex")
    print("- summary.json")


if __name__ == "__main__":
    main()
