#!/usr/bin/env python3
"""Measure Params, FLOPs, latency, FPS, throughput, model size, and peak memory for ABF-Net."""
from __future__ import annotations

import argparse
import copy
import io
import os
from pathlib import Path
import sys
import time

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.build import build_model


def load_model(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = build_model(config).eval()
    return model, int(config["DATA"].get("IMG_SIZE", 224))


def thop_gflops(model, x):
    from thop import profile
    with torch.no_grad():
        ops, _ = profile(model, inputs=(x,), verbose=False)
    return float(ops) / 1e9


def state_dict_size_mb(model):
    buf = io.BytesIO()
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    torch.save(sd, buf)
    return buf.tell() / (1024 * 1024)


def proc_rss_mb():
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def bench(model, x, device, warmup, repeat):
    model = model.to(device).eval()
    x = x.to(device)
    peak = None
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
            if device == "cpu":
                rss = proc_rss_mb()
                peak = rss if peak is None else max(peak, rss or peak)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(repeat):
            model(x)
            if device == "cpu":
                rss = proc_rss_mb()
                peak = rss if peak is None else max(peak, rss or peak)
        if device == "cuda":
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0 / repeat
    fps = 1000.0 / latency_ms
    return latency_ms, fps, peak


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/abfnet_rafdb.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=os.environ.get("ABFNET_BENCH_DEVICE", "auto"))
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("ABFNET_BENCH_WARMUP", "20")))
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("ABFNET_BENCH_REPEAT", "200")))
    args = parser.parse_args()

    model, img_size = load_model(args.config)
    x = torch.randn(1, 3, img_size, img_size)
    gflops = thop_gflops(copy.deepcopy(model), x)
    cuda_ok = torch.cuda.is_available()
    device = "cuda" if args.device == "auto" and cuda_ok else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not cuda_ok:
        raise RuntimeError("CUDA was requested but is not available.")

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    model_size = state_dict_size_mb(model)
    latency, fps, peak = bench(model, x, device, args.warmup, args.repeat)
    print("=== ABF-Net efficiency ===")
    print(f"config:        {args.config}")
    print(f"device:        {device}")
    print(f"input:         1x3x{img_size}x{img_size}")
    print(f"Param (M):     {params_m:.4f}")
    print(f"FLOPs (G):     {gflops:.4f}  # thop")
    print(f"Model Size MB: {model_size:.2f}  # state_dict serialization")
    print(f"Peak Memory MB:{peak:.2f}" if peak is not None else "Peak Memory MB:N/A")
    print(f"Latency (ms):  {latency:.2f}")
    print(f"FPS:           {fps:.1f}")
    print(f"Throughput:    {fps:.1f} img/s  # batch=1")


if __name__ == "__main__":
    main()
