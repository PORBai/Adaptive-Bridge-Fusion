import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict


EMOTION_ORDER = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FERPlus ImageFolder dataset from official fer2013new.csv protocol."
    )
    parser.add_argument(
        "--ferplus-csv",
        required=True,
        help="Path to official fer2013new.csv.",
    )
    parser.add_argument(
        "--images-root",
        required=True,
        help="Root directory containing ferXXXXXXX.png files (searched recursively).",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Output ImageFolder root: output/{train,val,test}/{class_id}/xxx.png",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        choices=[7, 8],
        default=7,
        help="7: exclude contempt; 8: include contempt.",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=1,
        help="Minimum top vote count to keep sample.",
    )
    parser.add_argument(
        "--keep-ties",
        action="store_true",
        help="Keep tied top-vote samples (choose first by emotion order).",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="symlink",
        help="How to place images into output ImageFolder.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output-root before building.",
    )
    return parser.parse_args()


def find_all_images(images_root):
    index = {}
    duplicates = defaultdict(list)
    for dirpath, _, filenames in os.walk(images_root):
        for name in filenames:
            lower = name.lower()
            if not lower.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            full = os.path.join(dirpath, name)
            if name in index:
                duplicates[name].append(full)
            else:
                index[name] = full
    return index, duplicates


def usage_to_split(usage):
    usage = usage.strip().lower()
    if usage == "training":
        return "train"
    if usage == "publictest":
        return "val"
    if usage == "privatetest":
        return "test"
    return None


def choose_label(row, num_classes, min_votes, keep_ties):
    selected = EMOTION_ORDER[:7] if num_classes == 7 else EMOTION_ORDER[:8]
    votes = [int(row[e]) for e in selected]

    top_vote = max(votes)
    if top_vote < min_votes:
        return None, "low_votes"

    top_indices = [i for i, v in enumerate(votes) if v == top_vote]
    if len(top_indices) > 1 and not keep_ties:
        return None, "tie"

    label_index = top_indices[0]
    class_id = str(label_index + 1)  # keep compatibility with existing configs (1..K)
    return class_id, None


def ensure_dirs(output_root, num_classes):
    for split in ["train", "val", "test"]:
        for class_id in range(1, num_classes + 1):
            os.makedirs(os.path.join(output_root, split, str(class_id)), exist_ok=True)


def place_file(src, dst, copy_mode):
    if os.path.exists(dst):
        return
    if copy_mode == "copy":
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def main():
    args = parse_args()

    if args.clean_output and os.path.isdir(args.output_root):
        shutil.rmtree(args.output_root)
    os.makedirs(args.output_root, exist_ok=True)

    image_index, duplicates = find_all_images(args.images_root)
    if duplicates:
        print(f"[warn] duplicated basenames found: {len(duplicates)}")
        print("[warn] first path is used by default")

    ensure_dirs(args.output_root, args.num_classes)

    stats = Counter()
    split_class_counter = defaultdict(Counter)
    skipped_examples = []

    with open(args.ferplus_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"Usage", "Image name", "unknown", "NF", *EMOTION_ORDER}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

        for row in reader:
            stats["rows_total"] += 1
            split = usage_to_split(row["Usage"])
            if split is None:
                stats["skip_unknown_usage"] += 1
                continue

            image_name = row["Image name"].strip()
            src = image_index.get(image_name)
            if src is None:
                stats["skip_image_missing"] += 1
                if len(skipped_examples) < 20:
                    skipped_examples.append({"image": image_name, "reason": "image_missing"})
                continue

            # Follow FERPlus protocol: remove samples mainly labeled unknown/NF.
            unknown_votes = int(row["unknown"])
            nf_votes = int(row["NF"])
            emotion_votes = [int(row[e]) for e in (EMOTION_ORDER[:7] if args.num_classes == 7 else EMOTION_ORDER)]
            if max([unknown_votes, nf_votes]) > max(emotion_votes):
                stats["skip_unknown_or_nf_dominant"] += 1
                continue

            class_id, reason = choose_label(
                row=row,
                num_classes=args.num_classes,
                min_votes=args.min_votes,
                keep_ties=args.keep_ties,
            )
            if class_id is None:
                stats[f"skip_{reason}"] += 1
                continue

            dst = os.path.join(args.output_root, split, class_id, image_name)
            place_file(src, dst, args.copy_mode)
            stats["kept_total"] += 1
            split_class_counter[split][class_id] += 1

    summary = {
        "ferplus_csv": args.ferplus_csv,
        "images_root": args.images_root,
        "output_root": args.output_root,
        "num_classes": args.num_classes,
        "copy_mode": args.copy_mode,
        "min_votes": args.min_votes,
        "keep_ties": args.keep_ties,
        "stats": dict(stats),
        "split_class_count": {k: dict(v) for k, v in split_class_counter.items()},
        "skipped_examples": skipped_examples,
    }

    summary_path = os.path.join(args.output_root, "build_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Build finished.")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
