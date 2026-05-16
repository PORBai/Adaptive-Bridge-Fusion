# Dataset Preparation

Due to license restrictions, RAF-DB, FERPlus, and AffectNet are not redistributed in this repository. Users must obtain the datasets from their official providers and prepare them locally.

## Expected ImageFolder Layout

The training and evaluation code uses `torchvision.datasets.ImageFolder`. Each dataset root must contain `train/` and `test/`, and each split must contain one subdirectory per class.

```text
data/
├── RAFDB/
│   ├── train/
│   └── test/
├── FERPlus/
│   ├── train/
│   └── test/
├── AffectNet7/
│   ├── train/
│   └── test/
└── AffectNet8/
    ├── train/
    └── test/
```

`ImageFolder` sorts class folders lexicographically. For numeric folders, use consistent names such as `1`, `2`, ..., `7` or `0`, `1`, ..., `6`, and keep the same convention across train and test splits.

## Seven-Class Order

The seven-class FER protocol follows:

1. Surprise
2. Fear
3. Disgust
4. Happiness
5. Sadness
6. Anger
7. Neutral

If your official dataset release uses a different folder order, update the folder names or document the mapping before comparing with the reported results.

## AffectNet-7 and AffectNet-8

AffectNet-7 excludes the contempt category. AffectNet-8 includes contempt as the eighth category. The recommended eight-class order is:

1. Neutral
2. Happiness
3. Sadness
4. Surprise
5. Fear
6. Disgust
7. Anger
8. Contempt

Please keep the class order consistent with your preprocessing scripts and manuscript protocol.

## Challenge-Condition Evaluation Lists

The development code contains optional RAF-DB and FERPlus challenge-condition evaluation scripts. If the corresponding challenge-condition list files are publicly redistributable in your environment, place them under:

```text
datasets/lists/RAF_DB_dir/
datasets/lists/FERplus_dir/
```

If you are unsure about redistribution rights, do not publish the list files. Ask users to obtain them from the original providers and place them in the directories above.
