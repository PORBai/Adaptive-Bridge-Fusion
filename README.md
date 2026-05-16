# Adaptive Bridge Fusion for Heterogeneous Feature Learning in Visual Facial Expression Analysis

This repository is directly associated with the manuscript **“Adaptive Bridge Fusion for Heterogeneous Feature Learning in Visual Facial Expression Analysis”**, submitted to *The Visual Computer*. The repository is released to support transparency, reproducibility, and further research on heterogeneous feature learning for visual facial expression analysis.

## Overview

Facial expression recognition in unconstrained environments remains challenging because expression-related cues are often subtle, locally distributed, and affected by inter-class similarity, intra-class variation, pose changes, illumination changes, and partial occlusion. Existing methods have achieved notable progress, but the alignment and adaptive fusion of heterogeneous backbone features with different spatial resolutions, channel dimensions, and semantic levels remain insufficiently explored.

This repository provides the implementation of an adaptive bridge fusion framework for visual facial expression analysis. The proposed method integrates compact single-scale MobileFaceNet representations and hierarchical multi-scale IR50 features through a heterogeneous dual-branch design. Instead of directly concatenating heterogeneous features, the adaptive bridge fusion layer first projects them into a unified spatial-channel space and then integrates them using branch-level gated fusion and multi-scale residual enhancement.

The framework contains three main components:

- **Heterogeneous dual-branch frontend.** The MobileFaceNet branch extracts compact single-scale facial features, while the IR50 branch provides hierarchical multi-scale facial representations.
- **Adaptive bridge fusion layer.** Heterogeneous branch features are aligned into a unified spatial-channel space and integrated through sample-adaptive branch-level gating and multi-scale residual enhancement.
- **Compact Conv-SE backend.** The fused representation is refined by a compact convolutional backend with squeeze-and-excitation recalibration for final expression classification.

The released code is intended to facilitate reproducibility of the main experiments reported in the manuscript and to provide a reusable implementation for studying heterogeneous feature fusion in visual facial expression analysis.

## Repository Structure

```text
ABFNet_TVC_OpenSource/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── main.py
├── configs/
│   ├── abfnet_rafdb.yaml
│   ├── abfnet_ferplus.yaml
│   ├── abfnet_affectnet7.yaml
│   └── abfnet_affectnet8.yaml
├── datasets/
│   └── dataset loading and preprocessing utilities
├── models/
│   ├── backbones/
│   ├── abf/
│   └── model definition files
├── train/
│   └── training entry scripts and training utilities
├── eval/
│   └── evaluation scripts
├── tools/
│   └── complexity measurement, visualization, and analysis scripts
├── scripts/
│   └── shell scripts for training and evaluation
├── docs/
│   ├── DATASET_PREPARATION.md
│   ├── REPRODUCE_RESULTS.md
│   ├── MODEL_ZOO.md
│   ├── CITATION.md
│   └── OPEN_SOURCE_AUDIT.md
├── assets/
│   └── figures, result summaries, and auxiliary files
├── pretrain/
│   └── placeholder for external pretrained backbones
└── checkpoints/
    └── placeholder for downloaded checkpoints
```

## Main Results

The main results reported in the manuscript are summarized below.

| Dataset | Accuracy (%) |
|---|---:|
| RAF-DB | 92.37 |
| FERPlus | 89.02 |
| AffectNet-7 | 81.47 |
| AffectNet-8 | 79.33 |

Actual results may vary slightly depending on dataset preprocessing, random seed, software versions, hardware environment, and checkpoint selection.

## Environment Setup

The code is implemented with PyTorch. A recommended environment can be created as follows:

```bash
conda create -n abfnet python=3.10 -y
conda activate abfnet
pip install -r requirements.txt
```

The core dependencies include:

```text
torch
torchvision
numpy
opencv-python
Pillow
scikit-learn
matplotlib
tqdm
PyYAML
pandas
thop
timm
```

Please refer to `requirements.txt` for the full dependency list.

## Dataset Preparation

This project uses the following facial expression recognition datasets:

- RAF-DB
- FERPlus
- AffectNet-7
- AffectNet-8

Due to dataset license restrictions, the original images and annotations are **not redistributed** in this repository. Users should obtain the datasets from their official providers and organize them according to the required directory structure.

A recommended ImageFolder-style structure is:

```text
data/
├── RAFDB/
│   ├── train/
│   │   ├── 1/
│   │   ├── 2/
│   │   └── ...
│   └── test/
│       ├── 1/
│       ├── 2/
│       └── ...
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

More detailed instructions are provided in:

```text
docs/DATASET_PREPARATION.md
```

Please make sure that the class order and preprocessing protocol are consistent with the experimental settings described in the manuscript.

## Pretrained Checkpoints

Pretrained checkpoints are provided to facilitate reproducibility. The checkpoint files are not stored directly in this GitHub repository due to file size considerations. Please download the corresponding checkpoint files from the links below and place them under the `checkpoints/` directory.

| Dataset | Accuracy (%) | Checkpoint | Notes |
|---|---:|---|---|
| RAF-DB | 92.37 | [Download](https://pan.quark.cn/s/407c4d9b1bcf) | Best checkpoint selected on RAF-DB |
| FERPlus | 89.02 | [Download](https://pan.quark.cn/s/407c4d9b1bcf) | Best checkpoint selected on FERPlus |
| AffectNet-7 | 81.47 | [Download](https://pan.quark.cn/s/407c4d9b1bcf) | Best checkpoint under the 7-class setting |
| AffectNet-8 | 79.33 | [Download](https://pan.quark.cn/s/407c4d9b1bcf) | Best checkpoint under the 8-class setting |

All four checkpoints are provided in the shared folder. Please download the corresponding checkpoint according to the dataset name.

After downloading a checkpoint, place it under:

```text
checkpoints/
```

For example:

```text
checkpoints/abfnet_rafdb_best.pth
checkpoints/abfnet_ferplus_best.pth
checkpoints/abfnet_affectnet7_best.pth
checkpoints/abfnet_affectnet8_best.pth
```

More information is provided in:

```text
docs/MODEL_ZOO.md
```

## Training

Training can be launched with the provided configuration files.

For RAF-DB:

```bash
python train/train_abfnet.py --config configs/abfnet_rafdb.yaml
```

For FERPlus:

```bash
python train/train_abfnet.py --config configs/abfnet_ferplus.yaml
```

For AffectNet-7:

```bash
python train/train_abfnet.py --config configs/abfnet_affectnet7.yaml
```

For AffectNet-8:

```bash
python train/train_abfnet.py --config configs/abfnet_affectnet8.yaml
```

Equivalent shell scripts may also be available under:

```text
scripts/
```

Example:

```bash
bash scripts/train_rafdb.sh
bash scripts/train_ferplus.sh
bash scripts/train_affectnet7.sh
bash scripts/train_affectnet8.sh
```

Please check the configuration files and update dataset paths before training.

## Evaluation

To evaluate a trained model, run:

```bash
python eval/test_accuracy.py \
  --config configs/abfnet_rafdb.yaml \
  --checkpoint checkpoints/abfnet_rafdb_best.pth
```

Example evaluation commands are listed below.

RAF-DB:

```bash
python eval/test_accuracy.py \
  --config configs/abfnet_rafdb.yaml \
  --checkpoint checkpoints/abfnet_rafdb_best.pth
```

FERPlus:

```bash
python eval/test_accuracy.py \
  --config configs/abfnet_ferplus.yaml \
  --checkpoint checkpoints/abfnet_ferplus_best.pth
```

AffectNet-7:

```bash
python eval/test_accuracy.py \
  --config configs/abfnet_affectnet7.yaml \
  --checkpoint checkpoints/abfnet_affectnet7_best.pth
```

AffectNet-8:

```bash
python eval/test_accuracy.py \
  --config configs/abfnet_affectnet8.yaml \
  --checkpoint checkpoints/abfnet_affectnet8_best.pth
```

## Complexity and Inference Efficiency

The number of parameters, FLOPs, latency, and FPS can be measured using the efficiency measurement script.

Example:

```bash
python tools/measure_efficiency.py \
  --config configs/abfnet_rafdb.yaml \
  --checkpoint checkpoints/abfnet_rafdb_best.pth
```

The complexity values reported in the manuscript are measured with input size `224 × 224` and batch size `1`.

Please note that latency and FPS may vary across different hardware platforms, CUDA versions, PyTorch versions, and measurement settings.

## Robustness Evaluation

The manuscript includes robustness evaluation under occlusion and large-pose conditions. If the corresponding evaluation list files are available, users can reproduce the challenge-condition evaluation using the scripts provided in `tools/`.

Example:

```bash
python tools/eval_raf_ran_subsets.py \
  --config configs/abfnet_rafdb.yaml \
  --checkpoint checkpoints/abfnet_rafdb_best.pth
```

```bash
python tools/eval_ferplus_ran_subsets.py \
  --config configs/abfnet_ferplus.yaml \
  --checkpoint checkpoints/abfnet_ferplus_best.pth
```

The original datasets are not redistributed. Users should also ensure that any challenge-condition lists used for evaluation comply with their original license and usage terms.

## Visualization

This repository includes scripts for qualitative analysis, including confusion matrix visualization, Grad-CAM visualization, and t-SNE feature visualization when the required dependencies, data, and checkpoint files are available.

Example commands:

```bash
python tools/visualize_gradcam.py \
  --config configs/abfnet_rafdb.yaml \
  --checkpoint checkpoints/abfnet_rafdb_best.pth
```

```bash
python tools/visualize_tsne.py \
  --config configs/abfnet_rafdb.yaml \
  --checkpoint checkpoints/abfnet_rafdb_best.pth
```

```bash
python tools/plot_confusion_matrix.py \
  --config configs/abfnet_rafdb.yaml \
  --checkpoint checkpoints/abfnet_rafdb_best.pth
```

Please adjust the script names and arguments according to the actual files in the repository.

## Reproducibility Notes

To reproduce the reported results, users should ensure that:

1. The datasets are obtained from the official providers.
2. The dataset directory structure follows `docs/DATASET_PREPARATION.md`.
3. The corresponding configuration file is used for each dataset.
4. The downloaded checkpoints are placed under the `checkpoints/` directory.
5. The same preprocessing and evaluation protocol is used for each benchmark.
6. The software environment is consistent with the dependencies listed in `requirements.txt`.

Detailed reproduction instructions are provided in:

```text
docs/REPRODUCE_RESULTS.md
```

## Code and Data Availability

The source code, configuration files, reproduction guidelines, and pretrained checkpoint links are provided in this repository.

Due to the license restrictions of RAF-DB, FERPlus, and AffectNet, the original datasets are not redistributed. Users should obtain the datasets from the official providers and organize them according to the instructions in:

```text
docs/DATASET_PREPARATION.md
```

The released code is directly related to the manuscript submitted to *The Visual Computer*. If you use this code, checkpoints, or benchmark protocol, please cite the associated manuscript.

## Notes on External Resources

This repository may require external pretrained backbones or dataset resources depending on the experimental setting. These files are not redistributed unless their licenses explicitly allow redistribution. Users should obtain external resources from their official sources and follow the corresponding licenses and terms of use.

The `pretrain/` and `checkpoints/` directories are intended as placeholders for locally downloaded files.

## License

This repository is released under the MIT License.

Please also follow the licenses and usage terms of all external datasets, pretrained backbones, and third-party libraries used in this project.

## Citation

If you use this code, checkpoints, or benchmark protocol, please cite the associated manuscript:

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

For questions about the code, reproduction protocol, or manuscript, please contact:

```text
Aibin Huang
Hangzhou Dianzi University
Email: huangaibin@hdu.edu.cn
```
