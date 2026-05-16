# Open-Source Packaging Audit

## Core Implementation Locations

| Component | Location |
| --- | --- |
| ABF-Net / LABF-Net main model | `models/Baiyr.py` |
| MobileFaceNet branch | `models/mobile_branch.py`, `models/third_party/poster_v2_backbone/mobilefacenet.py` |
| IR50 branch | `models/ir50_branch.py`, `models/third_party/poster_v2_backbone/ir50.py` |
| Adaptive Bridge Fusion layer | `models/bridge.py` |
| Conv-SE backend | `models/lightweight_backend.py` |
| Data loading | `datasets/build.py`, `datasets/predata.py` |
| Training | `train/train_abfnet.py`, `main.py`, `train/train_one_epoch.py` |
| Evaluation | `eval/test_accuracy.py`, `train/evaluate.py` |
| Params/FLOPs/FPS | `tools/measure_efficiency.py` |
| Grad-CAM | `tools/visualize_gradcam.py`, `tools/visualize_gradcam_rafdb_panel.py` |
| t-SNE | `tools/visualize_tsne.py`, `tools/plot_abfnet_four_dataset_tsne.py` |
| Confusion matrix | `tools/run_postprocess.py` |
| Checkpoint loading | `utils/checkpoint.py`, `eval/test_accuracy.py`, `main.py` |

## Suitable for Open Source

The model definitions, training/evaluation scripts, configuration templates, lightweight visualization scripts, and documentation are suitable for open-source release.

## Not Suitable for Open Source

Raw datasets, private local paths, personal experiment logs, temporary results, `wandb/`, `__pycache__/`, large checkpoints without explicit release intent, and third-party challenge-condition lists with unclear redistribution rights should not be published.
