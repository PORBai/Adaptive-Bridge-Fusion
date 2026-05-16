# Adaptive Bridge Fusion for Heterogeneous Feature Learning in Visual Facial Expression Analysis


## Method Overview

ABF-Net is designed for visual facial expression analysis with heterogeneous feature learning. The model contains three main components:

- **Heterogeneous dual-branch frontend.** A MobileFaceNet branch extracts compact local facial features, while an IR50 branch provides stronger identity-aware and semantic representations.
- **Adaptive bridge fusion layer.** The bridge aligns heterogeneous branch outputs to a common spatial and channel space, adaptively estimates branch-level fusion weights, and produces a unified feature map for classification.
- **Compact Conv-SE backend.** A lightweight convolutional backend with squeeze-and-excitation recalibration performs efficient classification from the fused bridge representation.

The core implementation is directly related to the submitted manuscript. If you use this code, please cite the manuscript listed in `docs/CITATION.md`.

## Repository Structure

```text
ABFNet_TVC_OpenSource/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/              # YAML configs for RAF-DB, FERPlus, AffectNet-7, AffectNet-8
├── datasets/             # Dataset loaders, preprocessing, and optional data preparation scripts
├── models/               # ABF-Net/LABF-Net model, bridge fusion layer, branches, and backbones
├── train/                # Training entry and epoch-level training/evaluation utilities
├── eval/                 # Accuracy evaluation entry
├── tools/                # Complexity, speed, Grad-CAM, t-SNE, and post-processing tools
├── scripts/              # Convenience shell commands
├── docs/                 # Dataset, reproduction, model zoo, and citation notes
├── assets/               # Lightweight public metadata and result notes
├── checkpoints/          # Optional released checkpoints; large weights are normally hosted externally
└── pretrain/             # Optional backbone pretrained weights supplied by users
```

## Environment Setup

```bash
conda create -n abfnet python=3.10
conda activate abfnet
pip install -r requirements.txt
```

CUDA-enabled PyTorch should be installed according to your local CUDA driver. If you only need CPU sanity checks, the standard CPU PyTorch wheels are sufficient.

## Dataset Preparation

This project uses RAF-DB, FERPlus, AffectNet-7, and AffectNet-8.

Due to license restrictions, the datasets are not redistributed in this repository. Users should obtain the datasets from their official providers and organize them according to the following structure.

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

Each split should follow the `torchvision.datasets.ImageFolder` format, where every class is represented by one subdirectory. More details are provided in `docs/DATASET_PREPARATION.md`.

## Training

RAF-DB:

```bash
python train/train_abfnet.py --config configs/abfnet_rafdb.yaml
```

FERPlus, AffectNet-7, and AffectNet-8 use the corresponding config files:

```bash
python train/train_abfnet.py --config configs/abfnet_ferplus.yaml
python train/train_abfnet.py --config configs/abfnet_affectnet7.yaml
python train/train_abfnet.py --config configs/abfnet_affectnet8.yaml
```

## Evaluation

```bash
python eval/test_accuracy.py --config configs/abfnet_rafdb.yaml --checkpoint path/to/checkpoint.pth
```

For the RAF-DB checkpoint associated with the 92.37% result, the intended local path is:

```bash
python eval/test_accuracy.py --config configs/abfnet_rafdb.yaml --checkpoint checkpoints/abfnet_rafdb_9237.pth
```

## Complexity and Speed

The following command reports Params, FLOPs, model size, peak memory, latency, FPS, and throughput with batch size 1:

```bash
python tools/measure_efficiency.py --config configs/abfnet_rafdb.yaml --device auto
```

You may set `ABFNET_BENCH_WARMUP` and `ABFNET_BENCH_REPEAT` to control timing repetitions.

## Visualization

Grad-CAM:

```bash
python tools/visualize_gradcam.py --config configs/abfnet_rafdb.yaml --checkpoint checkpoints/abfnet_rafdb_9237.pth --image path/to/image.jpg
```

t-SNE:

```bash
python tools/visualize_tsne.py --config configs/abfnet_rafdb.yaml --checkpoint checkpoints/abfnet_rafdb_9237.pth --dataset-title RAF-DB
```

Confusion matrix and paper-style reports:

```bash
python tools/run_postprocess.py --config configs/abfnet_rafdb.yaml --exp-name abfnet_rafdb --confusion-title RAF-DB --dpi 300
```

## Pretrained Checkpoints

| Dataset | Accuracy (%) | Checkpoint | Notes |
|---|---:|---|---|
| RAF-DB | 92.37 |[https://pan.quark.cn/s/407c4d9b1bcf] | Best checkpoint selected on RAF-DB |
| FERPlus | 89.02 | [https://pan.quark.cn/s/407c4d9b1bcf] | Best checkpoint selected on FERPlus |
| AffectNet-7 | 81.47 | [https://pan.quark.cn/s/407c4d9b1bcf] | Best checkpoint under 7-class setting |
| AffectNet-8 | 79.33 | [https://pan.quark.cn/s/407c4d9b1bcf]| Best checkpoint under 8-class setting |

## Usage

After downloading a checkpoint, place it under:

```bash
checkpoints/

Actual results may vary slightly depending on hardware, random seed, dataset preprocessing, and software versions.

## Citation

```bibtex
@article{wu2026adaptive,
  title={Adaptive Bridge Fusion for Heterogeneous Feature Learning in Visual Facial Expression Analysis},
  author={Wu, Zixin and Huang, Aibin},
  journal={The Visual Computer},
  year={2026},
  note={Manuscript submitted}
}
```

## Contact

Corresponding author:  
Aibin Huang  
Hangzhou Dianzi University  
huangaibin@hdu.edu.cn
