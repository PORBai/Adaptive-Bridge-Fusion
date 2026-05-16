#!/usr/bin/env bash
set -euo pipefail
python train/train_abfnet.py --config configs/abfnet_rafdb.yaml
