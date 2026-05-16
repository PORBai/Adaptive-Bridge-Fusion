# Reproduce Results

This document describes the commands used to reproduce the main ABF-Net results reported in the manuscript submitted to *The Visual Computer*.

## Training

RAF-DB:

```bash
python train/train_abfnet.py --config configs/abfnet_rafdb.yaml
```

FERPlus:

```bash
python train/train_abfnet.py --config configs/abfnet_ferplus.yaml
```

AffectNet-7:

```bash
python train/train_abfnet.py --config configs/abfnet_affectnet7.yaml
```

AffectNet-8:

```bash
python train/train_abfnet.py --config configs/abfnet_affectnet8.yaml
```

## Evaluation

```bash
python eval/test_accuracy.py --config configs/abfnet_rafdb.yaml --checkpoint checkpoints/abfnet_rafdb_9237.pth
python eval/test_accuracy.py --config configs/abfnet_ferplus.yaml --checkpoint checkpoints/abfnet_ferplus_best.pth
python eval/test_accuracy.py --config configs/abfnet_affectnet7.yaml --checkpoint checkpoints/abfnet_affectnet7_best.pth
python eval/test_accuracy.py --config configs/abfnet_affectnet8.yaml --checkpoint checkpoints/abfnet_affectnet8_best.pth
```

## Complexity and Speed

```bash
python tools/measure_efficiency.py --config configs/abfnet_rafdb.yaml --device auto
```

The benchmark uses batch size 1. FLOPs are measured by `thop`. Latency, FPS, and throughput should be reported with the hardware and software environment.

## Visualization

Grad-CAM:

```bash
python tools/visualize_gradcam.py --config configs/abfnet_rafdb.yaml --checkpoint checkpoints/abfnet_rafdb_9237.pth --image path/to/image.jpg
```

t-SNE:

```bash
python tools/visualize_tsne.py --config configs/abfnet_rafdb.yaml --checkpoint checkpoints/abfnet_rafdb_9237.pth --dataset-title RAF-DB
```

Confusion matrix and per-class report:

```bash
python tools/run_postprocess.py --config configs/abfnet_rafdb.yaml --exp-name abfnet_rafdb --confusion-title RAF-DB --dpi 300
```

## Reported Results

| Dataset | Accuracy |
| --- | ---: |
| RAF-DB | 92.37% |
| FERPlus | 89.02% |
| AffectNet-7 | 81.47% |
| AffectNet-8 | 79.33% |

Actual results may vary slightly depending on hardware, random seed, dataset preprocessing, and software versions.
