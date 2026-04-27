#!/usr/bin/env python3
"""
tools/profile_model.py  —  Standalone profiler for the LitePT model family.

Logs to WandB project "LitePT-Profiling" (separate from the training project).
Each run records a full system fingerprint (GPU, CUDA, PyTorch, driver, packages)
so results across different machines can be compared cleanly.

═══════════════════════════════════════════════════════════════════════════════
NeurIPS-grade metrics recorded
═══════════════════════════════════════════════════════════════════════════════
PARAMETERS
  params_total_M / params_trainable_M / params_buffer_M
  model_size_mb       — disk footprint of saved weights (0 if no checkpoint)
  compression_ratio   — params_baseline / params_model (needs --baseline-params)

COMPUTE
  gflops_total + per-stage breakdown  (theoretical backbone-encoder estimate)
  gflops_measured                     (torch.profiler with_flops=True, bs=1)
  compute_efficiency_pp               — GFLOPs per parameter (M)

MEMORY  (GPU peak, batch-size-1, single scene)
  mem_infer_alloc_gb / mem_infer_reserv_gb   (eval forward)
  mem_train_alloc_gb / mem_train_reserv_gb   (train forward + backward, AMP)

LATENCY  (bs=1, CUDA events, n_warmup + n_measure passes)
  latency_mean_ms / latency_std_ms / latency_p50_ms / latency_p95_ms
  latency_min_ms  / latency_max_ms
  throughput_sps  — samples per second  (= 1000 / latency_mean_ms)
  fps             — alias for throughput_sps (clearer in paper tables)
  speedup_vs_baseline — latency_baseline / latency_model (needs --baseline-ms)

SYSTEM FINGERPRINT
  GPU name/memory/compute-cap, CUDA/PyTorch/cuDNN versions, driver,
  key package versions (spconv, flash_attn, timm …)

═══════════════════════════════════════════════════════════════════════════════
FLOPs formula (backbone encoder only, for RELATIVE comparison)
═══════════════════════════════════════════════════════════════════════════════
All counts = multiply-accumulate (MAC) × 2.
Token count: N₀ = n_input_points; Nₛ = Nₛ₋₁ // 4 (GridPool stride-2 → ÷4/stage).

Conv block:   FLOPs = N × (27×OCC×C² + C²) × 2   [OCC=0.25]
Attn block:   FLOPs = N × [(4 + 2·ratio)·C² + 2·P·C] × 2
              derived from: QKV(3C²)+Q@K(PC)+attn@V(PC)+OutProj(C²)+MLP(2·ratio·C²)

NOTE: This estimate covers only the backbone encoder. LayerNorm, PointROPE,
GridPooling, the decoder, and the task head are excluded. Use gflops_measured
(torch.profiler) for a more complete figure. Both numbers are reported.

═══════════════════════════════════════════════════════════════════════════════
PointROPE constraint
═══════════════════════════════════════════════════════════════════════════════
The PointROPE CUDA kernel requires:  head_dim % 6 == 0
(3D RoPE needs pairs for x,y,z axes → must be divisible by 6)
The original LitePT uses head_dim=18 everywhere (18%6=0 ✓).

Usage
─────
Before training (random weights — architecture cost):
    python tools/profile_model.py \\
        --config-file configs/scannet/semseg-litept-lw-e.py \\
        --run-name scannet/lw-e-random \\
        --wandb-project LitePT-Profiling

After training (load checkpoint — real-world cost):
    python tools/profile_model.py \\
        --config-file configs/scannet/semseg-litept-lw-e.py \\
        --weight exp/scannet/semseg-litept-lw-e/model/model_best.pth \\
        --run-name scannet/lw-e-trained \\
        --wandb-project LitePT-Profiling

KD checkpoint (student weights embedded in distillation checkpoint):
    python tools/profile_model.py \\
        --config-file configs/scannet/semseg-lw-c-kd-1200epoch.py \\
        --weight exp/scannet/semseg-lw-c-kd-1200epoch/model/model_best.pth \\
        --no-strict-load \\
        --run-name scannet/lw-c-kd-trained \\
        --wandb-project LitePT-Profiling

WandB login tip
───────────────
If wandb keeps asking you to log in interactively, run once in terminal:
    wandb login
Or pass your key directly:
    --wandb-key YOUR_API_KEY
"""

from __future__ import annotations

import os
import sys
import platform
import subprocess
import argparse
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

# ── Repo root on sys.path ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import wandb
from utils.config import Config, DictAction
from datasets import build_dataset, collate_fn
from models import build_model


# ══════════════════════════════════════════════════════════════════════════════
#  Argument parser
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LitePT standalone model profiler")
    p.add_argument("--config-file", required=True,
                   help="Path to model config (.py)")
    p.add_argument("--weight", default=None,
                   help="Checkpoint path (.pth). If omitted, random weights are used.")
    p.add_argument("--no-strict-load", action="store_true",
                   help="Use strict=False when loading the checkpoint. "
                        "Required for KD checkpoints (which carry teacher and projector "
                        "keys that do not exist in the inference-time student model).")
    p.add_argument("--run-name", required=True,
                   help="WandB run name, e.g. 'scannet/lw-e-random'")
    p.add_argument("--wandb-project", default="LitePT-Profiling",
                   help="WandB project (default: LitePT-Profiling)")
    p.add_argument("--wandb-key", default=None,
                   help="WandB API key. If None, uses current `wandb login` session.")
    p.add_argument("--extra-tags", default="",
                   help="Comma-separated additional WandB tags (e.g. 'semseg,lw-c,kd').")
    p.add_argument("--n-warmup", type=int, default=10,
                   help="Latency warmup iterations (default: 10)")
    p.add_argument("--n-measure", type=int, default=50,
                   help="Latency measurement iterations (default: 50)")
    p.add_argument("--n-profile-batches", type=int, default=50,
                   help="Val batches for memory/latency measurements (default: 50)")
    p.add_argument("--skip-train-memory", action="store_true",
                   help="Skip training-mode memory measurement (saves ~30 s)")
    p.add_argument("--baseline-params-m", type=float, default=None,
                   help="Baseline model params (M) for compression_ratio logging. "
                        "E.g. 12.71 for the full LitePT-small.")
    p.add_argument("--baseline-latency-ms", type=float, default=None,
                   help="Baseline model latency (ms) for speedup_vs_baseline logging. "
                        "E.g. the latency of semseg-litept-small-v1m1.")
    p.add_argument("--baseline-gflops", type=float, default=None,
                   help="Baseline model GFLOPs for efficiency_vs_baseline logging. "
                        "E.g. 25.42 for the full LitePT-small.")
    p.add_argument("--options", nargs="+", action=DictAction,
                   help="Override config options (same syntax as train.py)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  System information
# ══════════════════════════════════════════════════════════════════════════════

def collect_system_info() -> Dict[str, object]:
    """Full system fingerprint for WandB — distinguishes runs across machines."""
    info: Dict[str, object] = {}
    info["python_version"] = platform.python_version()
    info["os"]             = platform.platform()
    info["cpu_model"]      = platform.processor() or "unknown"
    info["torch_version"]  = torch.__version__
    info["cuda_version"]   = torch.version.cuda or "N/A"
    info["cudnn_version"]  = str(torch.backends.cudnn.version())

    n_gpus = torch.cuda.device_count()
    info["n_gpus"] = n_gpus
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        info[f"gpu_{i}_name"]         = props.name
        info[f"gpu_{i}_total_mem_gb"] = round(props.total_memory / 1024 ** 3, 2)
        info[f"gpu_{i}_compute_cap"]  = f"{props.major}.{props.minor}"
        info[f"gpu_{i}_sm_count"]     = props.multi_processor_count

    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0]
        info["nvidia_driver"] = out
    except Exception:
        info["nvidia_driver"] = "N/A"

    for pkg in ("spconv", "flash_attn", "timm", "torch_scatter", "pointops", "fvcore"):
        try:
            mod = __import__(pkg)
            info[f"pkg_{pkg}"] = getattr(mod, "__version__", "installed")
        except ImportError:
            info[f"pkg_{pkg}"] = "not_found"

    return info


# ══════════════════════════════════════════════════════════════════════════════
#  Parameter counting
# ══════════════════════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> Dict[str, float]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers   = sum(b.numel() for b in model.buffers())
    return {
        "params_total_M":     round(total     / 1e6, 4),
        "params_trainable_M": round(trainable / 1e6, 4),
        "params_buffer_M":    round(buffers   / 1e6, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Theoretical FLOPs estimator
# ══════════════════════════════════════════════════════════════════════════════

def estimate_gflops(cfg: Config, n_input_points: int = 60_000) -> Dict[str, object]:
    """
    Backbone-encoder-only theoretical FLOPs estimate.
    Use for RELATIVE comparison across ablation configs only.

    Token count: N₀=n_input_points; Nₛ=Nₛ₋₁//4 (GridPool stride-2 → ÷4/stage).
    Conv block:  FLOPs = N×(27×0.25×C²+C²)×2
    Attn block:  FLOPs = N×[(4+2·ratio)·C²+2·P·C]×2
    """
    try:
        bcfg = cfg.model.backbone
    except AttributeError:
        return {"gflops_total": float("nan"),
                "gflops_note": "cfg.model.backbone not found — cannot estimate."}

    enc_depths   = list(bcfg.enc_depths)
    enc_channels = list(bcfg.enc_channels)
    enc_patch_sz = list(bcfg.enc_patch_size)
    enc_conv     = list(bcfg.enc_conv)
    enc_attn     = list(bcfg.enc_attn)
    mlp_ratio    = int(getattr(bcfg, "mlp_ratio", 4))
    num_stages   = len(enc_depths)

    CONV_OCC = 0.25

    per_stage: Dict[str, float] = {}
    total_flops = 0.0
    n = n_input_points

    for s in range(num_stages):
        if s > 0:
            n = max(n // 4, 1)
        C = enc_channels[s]
        P = enc_patch_sz[s]
        D = enc_depths[s]
        sf = 0.0
        for _ in range(D):
            if enc_conv[s]:
                sf += n * (27 * CONV_OCC * C * C  +  C * C) * 2
            if enc_attn[s]:
                sf += n * ((4 + 2 * mlp_ratio) * C * C  +  2 * P * C) * 2
        per_stage[f"gflops_stage{s}"] = round(sf / 1e9, 4)
        total_flops += sf

    result: Dict[str, object] = {"gflops_total": round(total_flops / 1e9, 4)}
    result.update(per_stage)
    result["gflops_note"] = (
        "Backbone-encoder-only theoretical estimate. "
        "Conv occ=0.25, n_input=60000, stride-4 pooling per stage. "
        "Excludes LayerNorm, PointROPE, GridPooling, decoder, task head. "
        "Use gflops_measured for a more complete figure."
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Measured FLOPs via torch.profiler
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_flops_profiler(
    model: nn.Module,
    batch: dict,
    device: torch.device,
) -> Dict[str, object]:
    """
    Measure actual FLOPs using torch.profiler with_flops=True.

    Covers all operators traced by PyTorch's FlopCounterMode, including
    the decoder and task head, but EXCLUDING custom CUDA ops (PointROPE,
    flash_attn, spconv) that are not registered with the profiler.

    For NeurIPS: report alongside gflops_total (theoretical) and note that
    both are lower bounds — the full cost lies between them. The theoretical
    number is preferred for comparison because it is reproducible and formula-
    based, while this number shows the broader model cost.
    """
    model.eval()
    b = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
         for k, v in batch.items()}

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            with_flops=True,
            record_shapes=False,
        ) as prof:
            _ = model(b)
            torch.cuda.synchronize(device)

        total = sum(
            e.flops for e in prof.key_averages() if hasattr(e, "flops") and e.flops > 0
        )
        return {
            "gflops_measured": round(total / 1e9, 4),
            "gflops_measured_note": (
                "torch.profiler with_flops=True, full model (encoder+decoder+head). "
                "Excludes custom CUDA ops: PointROPE, flash_attn, spconv SubMConv3d. "
                "Actual cost is higher. Use for complementary reporting only."
            ),
        }
    except Exception as exc:
        return {"gflops_measured": float("nan"),
                "gflops_measured_note": f"torch.profiler FLOPs failed: {exc}"}


# ══════════════════════════════════════════════════════════════════════════════
#  GPU helper
# ══════════════════════════════════════════════════════════════════════════════

def move_batch_to_gpu(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Memory measurements
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_memory_inference(
    model: nn.Module,
    batches: List[dict],
    device: torch.device,
) -> Dict[str, float]:
    """Peak GPU memory allocated/reserved during inference over all batches."""
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    for batch in batches:
        _ = model(move_batch_to_gpu(batch, device))
    torch.cuda.synchronize(device)
    return {
        "mem_infer_alloc_gb":  round(torch.cuda.max_memory_allocated(device) / 1024**3, 4),
        "mem_infer_reserv_gb": round(torch.cuda.max_memory_reserved(device)  / 1024**3, 4),
    }


def measure_memory_training(
    model: nn.Module,
    batches: List[dict],
    device: torch.device,
    amp: bool = True,
) -> Dict[str, float]:
    """Peak GPU memory during one training forward+backward (no optimiser step)."""
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    batch = move_batch_to_gpu(batches[0], device)
    loss: Optional[torch.Tensor] = None

    try:
        try:
            ctx = torch.amp.autocast(device_type="cuda", enabled=amp, dtype=torch.float16)
        except TypeError:
            import torch.cuda.amp as _amp_compat
            ctx = _amp_compat.autocast(enabled=amp, dtype=torch.float16)

        with ctx:
            output = model(batch)
            if isinstance(output, dict):
                loss = output.get("loss", None)
                if loss is None:
                    loss = sum(
                        v.mean() for v in output.values()
                        if isinstance(v, torch.Tensor) and v.requires_grad
                    )
                    if not isinstance(loss, torch.Tensor):
                        loss = None
        if loss is not None:
            loss.backward()
    except Exception as exc:
        print(f"  [warn] Training forward/backward raised: {exc}")
        print("         mem_train_* values may be underestimated.")
    finally:
        torch.cuda.synchronize(device)
        model.zero_grad(set_to_none=True)

    alloc_gb  = torch.cuda.max_memory_allocated(device) / 1024**3
    reserv_gb = torch.cuda.max_memory_reserved(device)  / 1024**3
    model.eval()
    return {
        "mem_train_alloc_gb":  round(alloc_gb,  4),
        "mem_train_reserv_gb": round(reserv_gb, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Latency measurement
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_latency(
    model: nn.Module,
    batches: List[dict],
    device: torch.device,
    n_warmup: int = 10,
    n_measure: int = 50,
) -> Dict[str, float]:
    """CUDA-event latency (bs=1 single-sample). Batches are cycled as needed."""
    model.eval()

    def _get(i: int) -> dict:
        return batches[i % len(batches)]

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    # Warmup
    for i in range(n_warmup):
        _ = model(move_batch_to_gpu(_get(i), device))
    torch.cuda.synchronize(device)

    times: List[float] = []
    for i in range(n_measure):
        b = move_batch_to_gpu(_get(i), device)
        start_evt.record()
        _ = model(b)
        end_evt.record()
        torch.cuda.synchronize(device)
        times.append(start_evt.elapsed_time(end_evt))

    t = np.array(times, dtype=np.float64)
    mean_ms = float(np.mean(t))
    fps     = round(1000.0 / mean_ms, 2) if mean_ms > 0 else float("nan")

    return {
        "latency_mean_ms": round(mean_ms,                          2),
        "latency_std_ms":  round(float(np.std(t)),                 2),
        "latency_min_ms":  round(float(np.min(t)),                 2),
        "latency_p50_ms":  round(float(np.percentile(t, 50)),      2),
        "latency_p95_ms":  round(float(np.percentile(t, 95)),      2),
        "latency_max_ms":  round(float(np.max(t)),                 2),
        "throughput_sps":  fps,
        "fps":             fps,   # alias — clearer in paper tables
        "latency_n_runs":  n_measure,
        "latency_batch_size": 1,  # explicitly record; all our runs use bs=1
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Model disk footprint
# ══════════════════════════════════════════════════════════════════════════════

def measure_model_size_mb(weight_path: Optional[str]) -> Dict[str, float]:
    """File size of the checkpoint in MB. 0 when no checkpoint is used."""
    if weight_path is None or not os.path.isfile(weight_path):
        return {"model_size_mb": 0.0, "model_size_note": "random weights (no checkpoint)"}
    size_bytes = os.path.getsize(weight_path)
    size_mb    = round(size_bytes / (1024 ** 2), 2)
    return {
        "model_size_mb":   size_mb,
        "model_size_note": f"checkpoint file: {os.path.basename(weight_path)}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Derived efficiency / comparison metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_efficiency_metrics(
    results: dict,
    baseline_params_m: Optional[float],
    baseline_latency_ms: Optional[float],
    baseline_gflops: Optional[float],
) -> Dict[str, object]:
    """
    Compute paper-table metrics that require a baseline reference.

    All ratios are ≥1.0 means the model is better than baseline on that axis.
    Report these in Table 1 to make the efficiency story immediately readable.
    """
    metrics: Dict[str, object] = {}

    # Compute efficiency = mIoU-per-GFLOPs placeholder
    # (actual mIoU comes from training logs; here we just record GFLOPs)
    gflops = results.get("gflops_total", None)

    # Compression ratio (parameter reduction)
    params_m = results.get("params_total_M", None)
    if baseline_params_m and params_m:
        metrics["compression_ratio_params"] = round(baseline_params_m / params_m, 3)
        metrics["params_reduction_pct"]     = round(
            100.0 * (1.0 - params_m / baseline_params_m), 1)

    # Latency speedup
    lat = results.get("latency_mean_ms", None)
    if baseline_latency_ms and lat and lat > 0:
        metrics["speedup_vs_baseline"] = round(baseline_latency_ms / lat, 3)
        metrics["latency_reduction_pct"] = round(
            100.0 * (1.0 - lat / baseline_latency_ms), 1)

    # FLOPs reduction
    if baseline_gflops and gflops and gflops > 0:
        metrics["flops_reduction_ratio"] = round(baseline_gflops / gflops, 3)
        metrics["flops_reduction_pct"]   = round(
            100.0 * (1.0 - gflops / baseline_gflops), 1)

    # Compute efficiency: GFLOPs per parameter
    if gflops and params_m and params_m > 0:
        metrics["gflops_per_param_m"] = round(gflops / params_m, 4)

    # Memory efficiency: throughput per GB of inference memory
    fps = results.get("fps", None)
    mem = results.get("mem_infer_alloc_gb", None)
    if fps and mem and mem > 0:
        metrics["throughput_per_gb"] = round(fps / mem, 3)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Model builder
# ══════════════════════════════════════════════════════════════════════════════

def build_and_load(
    cfg: Config,
    weight_path: Optional[str],
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    model = build_model(cfg.model)
    model = model.to(device)
    print(f"  Model built  ({sum(p.numel() for p in model.parameters())/1e6:.4f} M params)")

    if weight_path is not None:
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Checkpoint not found: {weight_path}")
        print(f"  Loading checkpoint: {weight_path}  (strict={strict})")
        try:
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(weight_path, map_location="cpu")

        state = ckpt.get("state_dict", ckpt)
        new_state: Dict[str, torch.Tensor] = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v)
            for k, v in state.items()
        )

        info = model.load_state_dict(new_state, strict=strict)

        if info.missing_keys:
            n = len(info.missing_keys)
            print(f"  [{'warn' if strict else 'info'}] Missing keys ({n}): "
                  f"{info.missing_keys[:3]} ...")
        if info.unexpected_keys:
            n = len(info.unexpected_keys)
            print(f"  [{'warn' if strict else 'info'}] Unexpected keys ({n}): "
                  f"{info.unexpected_keys[:3]} ...")
            if not strict:
                print("  [info] Unexpected keys are normal for KD checkpoints; "
                      "teacher/projector weights are safely ignored.")

        epoch_str = str(ckpt.get("epoch", "?"))
        print(f"  Checkpoint loaded (epoch {epoch_str})")

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Val data loader
# ══════════════════════════════════════════════════════════════════════════════

def build_profile_batches(cfg: Config, n_batches: int) -> List[dict]:
    """CPU-resident collated batches from the val dataset (GPU transfer is lazy)."""
    print(f"  Loading up to {n_batches} val batches ...")
    val_dataset = build_dataset(cfg.data.val)
    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    batches: List[dict] = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batches.append(batch)
    print(f"  Collected {len(batches)} val batches")
    return batches


# ══════════════════════════════════════════════════════════════════════════════
#  Config serialiser
# ══════════════════════════════════════════════════════════════════════════════

def cfg_to_dict(obj: object) -> object:
    """
    Recursively convert Config/namespace to JSON-serialisable dict.
    isinstance(dict) checked FIRST because Pointcept Config subclasses dict AND
    has __dict__ — checking __dict__ first would miss most keys.
    """
    try:
        if isinstance(obj, dict):
            return {k: cfg_to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cfg_to_dict(v) for v in obj]
        if hasattr(obj, "__dict__"):
            return {k: cfg_to_dict(v)
                    for k, v in vars(obj).items() if not k.startswith("_")}
        return obj
    except Exception:
        return str(obj)


# ══════════════════════════════════════════════════════════════════════════════
#  WandB login helper
# ══════════════════════════════════════════════════════════════════════════════

def ensure_wandb_login(api_key: Optional[str]) -> bool:
    if api_key:
        wandb.login(key=api_key, relogin=True)
        return True

    env_key = os.environ.get("WANDB_API_KEY", "")
    if env_key:
        return True

    try:
        api = wandb.Api(timeout=5)
        _ = api.viewer
        return True
    except Exception:
        pass

    print("\n  [warn] WandB is not logged in and no --wandb-key was provided.")
    print("  To fix permanently, run in your terminal:  wandb login")
    print("  Or pass your key:  --wandb-key YOUR_API_KEY")
    print("  Results will be printed to stdout instead.\n")
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    cfg = Config.fromfile(args.config_file)
    if args.options:
        cfg.merge_from_dict(args.options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available — memory and latency measurements skipped.")

    print("\n" + "=" * 68)
    print(f"  LitePT Profiler  —  {args.run_name}")
    print(f"  Config  : {args.config_file}")
    print(f"  Weight  : {args.weight or 'None (random / untrained weights)'}")
    print(f"  Strict  : {not args.no_strict_load}")
    print(f"  Device  : {device}")
    print("=" * 68 + "\n")

    # ── System info ───────────────────────────────────────────────────────────
    print("▶ Collecting system info ...")
    sys_info = collect_system_info()
    for k, v in sys_info.items():
        print(f"    {k}: {v}")

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n▶ Building model ...")
    model = build_and_load(cfg, args.weight, device, strict=not args.no_strict_load)

    # ── Model disk size ───────────────────────────────────────────────────────
    print("\n▶ Measuring model disk footprint ...")
    size_info = measure_model_size_mb(args.weight)
    for k, v in size_info.items():
        print(f"    {k}: {v}")

    # ── Parameters ────────────────────────────────────────────────────────────
    print("\n▶ Counting parameters ...")
    param_info = count_params(model)
    for k, v in param_info.items():
        print(f"    {k}: {v}")

    # ── Theoretical FLOPs ─────────────────────────────────────────────────────
    print("\n▶ Estimating theoretical GFLOPs (backbone encoder only) ...")
    flop_info = estimate_gflops(cfg, n_input_points=60_000)
    for k, v in flop_info.items():
        print(f"    {k}: {v}")

    # ── Load profiling batches ─────────────────────────────────────────────────
    batches: List[dict] = []
    skip_runtime = device.type != "cuda"
    if not skip_runtime:
        try:
            batches = build_profile_batches(cfg, args.n_profile_batches)
        except Exception as exc:
            print(f"\n[warn] Could not load val data: {exc}")
            print("       Memory, latency, and measured-FLOPs skipped.")
            skip_runtime = True

    # ── Measured FLOPs (torch.profiler) ───────────────────────────────────────
    flop_measured: Dict[str, object] = {}
    if not skip_runtime and batches:
        print("\n▶ Measuring FLOPs via torch.profiler (full model, 1 batch) ...")
        flop_measured = measure_flops_profiler(model, batches[0], device)
        for k, v in flop_measured.items():
            print(f"    {k}: {v}")

    # ── Inference memory ──────────────────────────────────────────────────────
    mem_infer: Dict[str, float] = {}
    if not skip_runtime and batches:
        print(f"\n▶ Measuring inference memory ({len(batches)} batches) ...")
        model.eval()
        mem_infer = measure_memory_inference(model, batches, device)
        for k, v in mem_infer.items():
            print(f"    {k}: {v} GB")

    # ── Training memory ───────────────────────────────────────────────────────
    mem_train: Dict[str, float] = {}
    if not skip_runtime and batches and not args.skip_train_memory:
        print("\n▶ Measuring training memory (1 forward + backward) ...")
        mem_train = measure_memory_training(
            model, batches, device,
            amp=bool(getattr(cfg, "enable_amp", True)),
        )
        for k, v in mem_train.items():
            print(f"    {k}: {v} GB")
        torch.cuda.empty_cache()

    # ── Inference latency ─────────────────────────────────────────────────────
    latency: Dict[str, float] = {}
    if not skip_runtime and batches:
        print(f"\n▶ Measuring latency "
              f"(warmup={args.n_warmup}, measure={args.n_measure}, bs=1) ...")
        model.eval()
        latency = measure_latency(model, batches, device,
                                  n_warmup=args.n_warmup, n_measure=args.n_measure)
        for k, v in latency.items():
            print(f"    {k}: {v}")

    # ── Assemble results ──────────────────────────────────────────────────────
    results: Dict[str, object] = {}
    results.update(param_info)
    results.update(size_info)
    results.update(flop_info)
    results.update(flop_measured)
    results.update(mem_infer)
    results.update(mem_train)
    results.update(latency)
    results["config_file"]       = args.config_file
    results["weight_path"]       = args.weight or "random"
    results["weight_loaded"]     = args.weight is not None
    results["strict_load"]       = not args.no_strict_load
    results["n_warmup"]          = args.n_warmup
    results["n_measure"]         = args.n_measure
    results["n_profile_batches"] = args.n_profile_batches

    # Backbone config summary for easy table population
    try:
        bcfg = cfg.model.backbone
        results["cfg_enc_channels"]  = str(list(bcfg.enc_channels))
        results["cfg_enc_depths"]    = str(list(bcfg.enc_depths))
        results["cfg_enc_patch_sz"]  = str(list(bcfg.enc_patch_size))
        results["cfg_enc_attn"]      = str(list(bcfg.enc_attn))
        results["cfg_num_stages"]    = len(list(bcfg.enc_depths))
    except AttributeError:
        results["cfg_note"] = "Could not read backbone config fields."

    # ── Derived efficiency / comparison metrics ───────────────────────────────
    print("\n▶ Computing efficiency metrics ...")
    eff_metrics = compute_efficiency_metrics(
        results,
        baseline_params_m    = args.baseline_params_m,
        baseline_latency_ms  = args.baseline_latency_ms,
        baseline_gflops      = args.baseline_gflops,
    )
    results.update(eff_metrics)
    for k, v in eff_metrics.items():
        print(f"    {k}: {v}")
    if not eff_metrics:
        print("    (no baseline values provided — skipped; use --baseline-* flags)")

    # ── WandB logging ─────────────────────────────────────────────────────────
    print("\n▶ Logging to WandB ...")
    wandb_ready = ensure_wandb_login(args.wandb_key)

    # Parse extra tags
    base_tags = ["profile"]
    if args.extra_tags:
        base_tags += [t.strip() for t in args.extra_tags.split(",") if t.strip()]

    if wandb_ready:
        try:
            try:
                backbone_cfg_dict = cfg_to_dict(cfg.model.backbone)
            except Exception:
                backbone_cfg_dict = {"serialise_error": "could not convert backbone cfg"}

            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config={
                    "backbone_cfg": backbone_cfg_dict,
                    "system":       sys_info,
                    "profiling": {
                        "n_warmup":           args.n_warmup,
                        "n_measure":          args.n_measure,
                        "n_profile_batches":  args.n_profile_batches,
                        "flop_n_input_pts":   60_000,
                        "skip_train_memory":  args.skip_train_memory,
                        "strict_load":        not args.no_strict_load,
                        "batch_size":         1,
                    },
                },
                tags=base_tags,
            )

            # Numeric metrics → single WandB step
            numeric = {k: v for k, v in results.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            wandb.log(numeric)

            # String / bool / NaN → summary (visible in WandB runs table)
            for k, v in results.items():
                if isinstance(v, (str, bool)):
                    wandb.run.summary[k] = v
                elif isinstance(v, float) and (v != v):  # NaN check
                    wandb.run.summary[k] = "N/A"

            # System info → summary with sys/ prefix
            for k, v in sys_info.items():
                wandb.run.summary[f"sys/{k}"] = v

            wandb.finish()
            print("  WandB run logged successfully ✓")

        except Exception as exc:
            print(f"  [warn] WandB logging failed: {exc}")
            wandb_ready = False

    if not wandb_ready:
        print("  Printing full results to stdout:")
        for k, v in results.items():
            print(f"    {k}: {v}")

    # ── Console summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  SUMMARY  —  {args.run_name}")
    print("=" * 68)
    key_metrics = [
        ("Params total (M)",         "params_total_M"),
        ("GFLOPs (theoretical)",     "gflops_total"),
        ("GFLOPs (measured, full)",  "gflops_measured"),
        ("Model size (MB)",          "model_size_mb"),
        ("Latency mean (ms)",        "latency_mean_ms"),
        ("Latency p95 (ms)",         "latency_p95_ms"),
        ("FPS (samples/s)",          "fps"),
        ("Mem infer alloc (GB)",     "mem_infer_alloc_gb"),
        ("Mem train alloc (GB)",     "mem_train_alloc_gb"),
        ("Compression ratio (param)","compression_ratio_params"),
        ("Speedup vs baseline",      "speedup_vs_baseline"),
        ("FLOPs reduction (%)",      "flops_reduction_pct"),
    ]
    for label, key in key_metrics:
        val = results.get(key, "N/A")
        if isinstance(val, float):
            val = f"{val:.4f}" if val < 1000 else f"{val:.2f}"
        print(f"  {label:<30} {val}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()