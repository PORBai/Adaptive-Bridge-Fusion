# Model Zoo

This page provides pretrained checkpoint information associated with the manuscript:

**Adaptive Bridge Fusion for Heterogeneous Feature Learning in Visual Facial Expression Analysis**

The checkpoints are provided to facilitate reproducibility of the main results reported in the manuscript submitted to *The Visual Computer*.

## Pretrained Checkpoints

| Dataset | Accuracy (%) | Checkpoint | Notes |
|---|---:|---|---|
| RAF-DB | 92.37 | [Download](https://pan.quark.cn/s/407c4d9b1bcf) | Best checkpoint selected on RAF-DB |
| FERPlus | 89.02 | [Download](https://pan.quark.cn/s/407c4d9b1bcf) | Best checkpoint selected on FERPlus |
| AffectNet-7 | 81.47 | [Download](https://pan.quark.cn/s/407c4d9b1bcf) | Best checkpoint under the 7-class setting |
| AffectNet-8 | 79.33 | [Download](https://pan.quark.cn/s/407c4d9b1bcf) | Best checkpoint under the 8-class setting |

All four checkpoints are provided in the shared folder. Please download the corresponding checkpoint according to the dataset name.

## Usage

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

Then evaluate the model using:

```bash
python eval/test_accuracy.py \
  --config configs/abfnet_rafdb.yaml \
  --checkpoint checkpoints/abfnet_rafdb_best.pth
```

## Dataset Notice

Due to the license restrictions of RAF-DB, FERPlus, and AffectNet, the original datasets are not redistributed in this repository. Users should obtain the datasets from their official providers and organize them according to `docs/DATASET_PREPARATION.md`.

## Citation Notice

These checkpoints are directly associated with the manuscript submitted to *The Visual Computer*. If you use the checkpoints, code, or benchmark protocol, please cite the associated manuscript.
