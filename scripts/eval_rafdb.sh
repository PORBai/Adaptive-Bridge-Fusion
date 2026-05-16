#!/usr/bin/env bash
set -euo pipefail
python eval/test_accuracy.py --config configs/abfnet_rafdb.yaml --checkpoint checkpoints/abfnet_rafdb_9237.pth
