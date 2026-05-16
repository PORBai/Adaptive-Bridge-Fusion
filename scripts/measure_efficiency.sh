#!/usr/bin/env bash
set -euo pipefail
python tools/measure_efficiency.py --config configs/abfnet_rafdb.yaml --device auto
