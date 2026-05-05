#!/usr/bin/env python3
"""
tools/profile_model.py  —  Standalone profiler for the LitePT / TopoPT model family.

Logs to WandB project "LitePT-Profiling" (separate from the training project).
Each run captures a full system fingerprint (GPU, CUDA, PyTorch, driver, packages)
so results from different machines can be compared cleanly.

═══════════════════════════════════════════════════════════════════════════════
NeurIPS-grade metrics recorded
═══════════════════════════════════════════════════════════════════════════════
PARAMETERS
  params_total_M / params_trainable_M / params_buffer_M
  model_size_mb         — on-disk checkpoint size (0 when random weights)
  compression_ratio     — baseline_params / model_params (needs --baseline-params-m)
  params_reduction_pct  — percentage reduction vs baseline

COMPUTE  (backbone encoder only — theoretical; reproducible formula)
  gflops_total                — total across all encoder stages
  gflops_stage{0..4}          — per-stage breakdown
  gflops_measured             — torch.profiler with_flops=True (full model, bs=1)
  gflops_per_param_m          — compute efficiency

MEMORY  (GPU peak, bs=1, single scene)
  mem_infer_alloc_gb / mem_infer_reserv_gb   — eval forward
  mem_train_alloc_gb / mem_train_reserv_gb   — train forward + backward (AMP)

LATENCY  (bs=1, CUDA events, n_warmup + n_measure passes)
  latency_mean_ms / latency_std_ms / latency_p50_ms / latency_p95_ms
  latency_min_ms  / latency_max_ms
  fps             — 1000 / latency_mean_ms
  speedup_vs_baseline — latency_baseline / latency_model (needs --baseline-latency-ms)

SYSTEM FINGERPRINT
  GPU name/memory/compute-cap, CUDA/PyTorch/cuDNN versions, driver,
  key package versions (spconv, flash_attn, timm, torch_scatter …)

═══════════════════════════════════════════════════════════════════════════════
FLOPs formula  (backbone encoder only — relative comparison across ablations)
═══════════════════════════════════════════════════════════════════════════════
All counts are multiply-accumulate (MAC) × 2 → FLOPs.
Token count per stage: N₀ = n_input_points; Nₛ = Nₛ₋₁ // 4  (stride-2 GridPool → ÷4).

Conv block  (SubMConv3d(C,C,3) + pointwise residual projection):
    FLOPs = N × (27 × OCC × C² + C²) × 2     [OCC = 0.25, sparse occupancy]

Attn block  (QKV proj + windowed attention + output proj + MLP):
    FLOPs = N × [(4 + 2·ratio) · C² + 2·P·C] × 2
    Breakdown: QKV(3C²) + OutProj(C²) + MLP(2·ratio·C²) + Attn-scores(P·C) + Attn-values(P·C)

NOTE: This estimate covers only the backbone encoder.  LayerNorm, PointROPE,
GridPooling, the decoder, and the task head are excluded.
Use gflops_measured (torch.profiler) for a broader figure that includes those
components, but note it excludes custom CUDA ops (PointROPE, spconv, flash_attn).

PointROPE constraint:  head_dim % 6 == 0
(3D RoPE pairs frequencies for x, y, z axes → must be divisible by 6)
LitePT-S uses head_dim = 18 everywhere (18 % 6 = 0 ✓).

═══════════════════════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════════════════════
# Random weights — architecture cost only (no data required if you add --skip-batches):
    python tools/profile_model.py \\
        --config-file configs/scannet/semseg-lw-c-100epoch.py \\
        --run-name    scannet/lw-c-random \\
        --wandb-project LitePT-Profiling \\
        --baseline-params-m 12.71 --baseline-gflops 25.42

# Trained checkpoint — real-world cost:
    python tools/profile_model.py \\
        --config-file configs/scannet/semseg-lw-c-100epoch.py \\
        --weight      exp/scannet-semseg-lw-c-100epoch/model/model_best.pth \\
        --run-name    scannet/lw-c-trained \\
        --wandb-project LitePT-Profiling \\
        --baseline-params-m 12.71 --baseline-gflops 25.42

# KD checkpoint — student weights embedded in distillation checkpoint;
# pass the BASE student config (not the KD config) to build_model():
    python tools/profile_model.py \\
        --config-file configs/scannet/semseg-lw-c-100epoch.py \\
        --weight      exp/scannet-semseg-lw-c-kd-100epoch/model/model_best.pth \\
        --no-strict-load \\
        --run-name    scannet/lw-c-kd-trained \\
        --wandb-project LitePT-Profiling

# Skip data-dependent measurements (params + FLOPs only, much faster):
    python tools/profile_model.py \\
        --config-file configs/scannet/semseg-lw-c-100epoch.py \\
        --skip-batches \\
        --run-name    scannet/lw-c-arch-only
"""

from __future__ import annotations

import os
import sys
import math
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
    p = argparse.ArgumentParser(
        description="LitePT / TopoPT standalone model profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument("--config-file", required=True,
                   help="Path to model config (.py). "
                        "For KD trained checkpoints, pass the BASE STUDENT config "
                        "so build_model() only constructs the student graph.")
    p.add_argument("--run-name", required=True,
                   help="WandB run name, e.g. 'scannet/lw-c-random'.")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    p.add_argument("--weight", default=None,
                   help="Checkpoint path (.pth). Omit for random (untrained) weights.")
    p.add_argument("--no-strict-load", action="store_true",
                   help="Use strict=False when loading the checkpoint.  "
                        "Required for KD checkpoints (teacher + projector keys are "
                        "unexpected in the inference-time student model).")

    # ── WandB ─────────────────────────────────────────────────────────────────
    p.add_argument("--wandb-project", default="LitePT-Profiling",
                   help="WandB project name (default: LitePT-Profiling).")
    p.add_argument("--wandb-key", default=None,
                   help="WandB API key. If None, uses the current `wandb login` session.")
    p.add_argument("--extra-tags", default="",
                   help="Comma-separated additional WandB tags, e.g. 'semseg,lw-c,kd'.")
    p.add_argument("--wandb-offline", action="store_true",
                   help="Run WandB in offline mode (useful on compute nodes without internet).")

    # ── Measurement settings ──────────────────────────────────────────────────
    p.add_argument("--n-warmup", type=int, default=10,
                   help="Latency warmup iterations, discarded (default: 10).")
    p.add_argument("--n-measure", type=int, default=50,
                   help="Latency measurement iterations (default: 50).")
    p.add_argument("--n-profile-batches", type=int, default=50,
                   help="Val batches loaded for memory/latency measurements (default: 50).")
    p.add_argument("--n-input-points", type=int, default=60_000,
                   help="Assumed input point count for theoretical FLOPs estimate (default: 60000).")
    p.add_argument("--skip-batches", action="store_true",
                   help="Skip all data-dependent measurements (memory, latency, measured FLOPs). "
                        "Only params and theoretical FLOPs are computed. Much faster.")
    p.add_argument("--skip-train-memory", action="store_true",
                   help="Skip training-mode memory measurement (saves ~30s per model).")

    # ── Baseline comparison ───────────────────────────────────────────────────
    p.add_argument("--baseline-params-m", type=float, default=None,
                   help="Baseline model parameter count (M) for compression_ratio. "
                        "Example: 12.71 for LitePT-S.")
    p.add_argument("--baseline-latency-ms", type=float, default=None,
                   help="Baseline model mean latency (ms) for speedup_vs_baseline. "
                        "Obtained from a prior profiler run of the teacher model.")
    p.add_argument("--baseline-gflops", type=float, default=None,
                   help="Baseline theoretical GFLOPs for flops_reduction_ratio. "
                        "Example: 25.42 for LitePT-S.")

    # ── Config overrides ──────────────────────────────────────────────────────
    p.add_argument("--options", nargs="+", action=DictAction,
                   help="Override config options (same key=value syntax as train.py).")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  System information
# ══════════════════════════════════════════════════════════════════════════════

def collect_system_info() -> Dict[str, object]:
    """Full system fingerprint — distinguishes runs across machines."""
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
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip().split("\n")[0]
        info["nvidia_driver"] = driver
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
#  Checkpoint guard
# ══════════════════════════════════════════════════════════════════════════════

def check_kd_config_warning(cfg: Config, strict: bool) -> None:
    """
    Warn if a KD config (has teacher_backbone) is used with strict=True.
    This is almost certainly a mistake: the student checkpoint will have
    unexpected teacher / projector keys and load_state_dict will raise.
    """
    has_teacher = hasattr(cfg, "model") and hasattr(cfg.model, "teacher_backbone")
    if has_teacher and strict:
        print(
            "\n  [WARN] The config has a 'teacher_backbone' field, which means it is a "
            "KD/distillation config.\n"
            "         For profiling trained KD checkpoints you should:\n"
            "           (a) pass the BASE STUDENT config (not the KD config) via --config-file\n"
            "               so that build_model() only constructs the student graph, OR\n"
            "           (b) add --no-strict-load to suppress unexpected-key errors from\n"
            "               teacher / projector weights embedded in the checkpoint.\n"
            "         Continuing with strict=True — this will likely raise a RuntimeError "
            "if a weight is supplied.\n"
        )


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
#  Theoretical FLOPs estimator  (backbone encoder only)
# ══════════════════════════════════════════════════════════════════════════════

def estimate_gflops(cfg: Config, n_input_points: int = 60_000) -> Dict[str, object]:
    """
    Backbone-encoder-only theoretical FLOPs estimate.

    Intended for RELATIVE comparison across ablation configs only.
    Excludes: LayerNorm, PointROPE, GridPooling, decoder, task head.
    Use gflops_measured (torch.profiler) for a broader figure.

    Token count:  N₀ = n_input_points;  Nₛ = Nₛ₋₁ // 4  (stride-2 pool → ÷4/stage).

    Conv block (SubMConv3d(C,C,3) + pointwise projection per block):
        FLOPs = N × (27 × OCC × C² + C²) × 2
        — 27×OCC×C²: sparse 3×3×3 conv at occupancy OCC (empirical: ~0.25)
        — C²:        pointwise (1×1×1) projection / residual linear
        — ×2:        MACs → FLOPs

    Attn block (QKV + windowed self-attn + output proj + MLP):
        FLOPs = N × [(4 + 2·ratio) · C² + 2·P·C] × 2
        — 3C²:       Q, K, V projections
        — C²:        output projection
        — 2·ratio·C²: MLP (two linear layers)
        — 2·P·C:     attention scores Q@K^T and weighted sum attn@V
                     (patch P points × C channels, simplified for serialized windows)
    """
    try:
        bcfg = cfg.model.backbone
    except AttributeError:
        return {
            "gflops_total": float("nan"),
            "gflops_note": "cfg.model.backbone not found — cannot estimate.",
        }

    enc_depths   = list(bcfg.enc_depths)
    enc_channels = list(bcfg.enc_channels)
    enc_patch_sz = list(bcfg.enc_patch_size)
    enc_conv     = list(bcfg.enc_conv)
    enc_attn     = list(bcfg.enc_attn)
    mlp_ratio    = int(getattr(bcfg, "mlp_ratio", 4))
    num_stages   = len(enc_depths)

    CONV_OCC = 0.25   # empirical sparse occupancy for indoor / outdoor scenes

    per_stage: Dict[str, float] = {}
    total_flops = 0.0
    n = n_input_points

    for s in range(num_stages):
        if s > 0:
            n = max(n // 4, 1)   # stride-2 GridPool in each spatial dim → ÷4 points
        C = enc_channels[s]
        P = enc_patch_sz[s]
        D = enc_depths[s]
        stage_flops = 0.0
        for _ in range(D):
            if enc_conv[s]:
                # SubMConv3d(C,C,3) + pointwise projection
                stage_flops += n * (27.0 * CONV_OCC * C * C + C * C) * 2.0
            if enc_attn[s]:
                # QKV + OutProj + MLP + attention scores + weighted sum
                stage_flops += n * ((4.0 + 2.0 * mlp_ratio) * C * C + 2.0 * P * C) * 2.0
        per_stage[f"gflops_stage{s}"] = round(stage_flops / 1e9, 4)
        total_flops += stage_flops

    result: Dict[str, object] = {"gflops_total": round(total_flops / 1e9, 4)}
    result.update(per_stage)
    result["gflops_n_input_pts"] = n_input_points
    result["gflops_note"] = (
        f"Backbone-encoder-only theoretical estimate. "
        f"Conv occ={CONV_OCC}, n_input={n_input_points:,}, stride-4 pooling per stage. "
        "Excludes: LayerNorm, PointROPE, GridPooling, decoder, task head. "
        "Use gflops_measured for a broader figure."
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
    Measure FLOPs using torch.profiler with_flops=True.

    Covers all operators traced by PyTorch's FlopCounterMode — including
    the decoder and task head — but EXCLUDES custom CUDA ops (PointROPE,
    flash_attn, spconv SubMConv3d) that are not registered with the profiler.

    For NeurIPS: report alongside gflops_total (theoretical) and note that
    both figures are lower bounds.  The theoretical number is preferred for
    relative comparison because it is reproducible and formula-based.
    """
    model.eval()
    b = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
         for k, v in batch.items()}
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            with_flops=True,
            record_shapes=False,
        ) as prof:
            _ = model(b)
            torch.cuda.synchronize(device)

        total = sum(
            e.flops
            for e in prof.key_averages()
            if hasattr(e, "flops") and e.flops > 0
        )
        return {
            "gflops_measured": round(total / 1e9, 4),
            "gflops_measured_note": (
                "torch.profiler with_flops=True, full model (encoder+decoder+head). "
                "Excludes custom CUDA ops: PointROPE, flash_attn, spconv SubMConv3d. "
                "Actual cost is higher. Complements the theoretical encoder estimate."
            ),
        }
    except Exception as exc:
        return {
            "gflops_measured": float("nan"),
            "gflops_measured_note": f"torch.profiler FLOPs measurement failed: {exc}",
        }


# ══════════════════════════════════════════════════════════════════════════════
#  GPU batch helper
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
    """Peak GPU memory allocated/reserved during eval-mode forward passes."""
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    for batch in batches:
        _ = model(move_batch_to_gpu(batch, device))
    torch.cuda.synchronize(device)
    return {
        "mem_infer_alloc_gb":  round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 4),
        "mem_infer_reserv_gb": round(torch.cuda.max_memory_reserved(device)  / 1024 ** 3, 4),
    }


def measure_memory_training(
    model: nn.Module,
    batches: List[dict],
    device: torch.device,
    amp: bool = True,
) -> Dict[str, float]:
    """
    Peak GPU memory during one training forward + backward (no optimiser step).

    Handles multiple model output formats:
      - dict with "loss" key
      - dict with multiple tensor values (sums them)
      - direct Tensor output
    """
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    batch = move_batch_to_gpu(batches[0], device)
    loss: Optional[torch.Tensor] = None

    try:
        # AMP context — handle both PyTorch >= 2.0 and older APIs
        try:
            ctx = torch.amp.autocast(device_type="cuda", enabled=amp, dtype=torch.float16)
        except TypeError:
            import torch.cuda.amp as _amp_compat  # type: ignore
            ctx = _amp_compat.autocast(enabled=amp, dtype=torch.float16)

        with ctx:
            output = model(batch)

        # Extract a scalar loss from various output formats
        if isinstance(output, torch.Tensor) and output.requires_grad:
            loss = output.mean()
        elif isinstance(output, dict):
            if "loss" in output and isinstance(output["loss"], torch.Tensor):
                loss = output["loss"]
            else:
                # Sum all grad-requiring tensors (covers PG-v1m2 and similar)
                candidates = [
                    v.mean() for v in output.values()
                    if isinstance(v, torch.Tensor) and v.requires_grad and v.numel() > 0
                ]
                if candidates:
                    loss = sum(candidates)  # type: ignore[arg-type]

        if loss is not None and isinstance(loss, torch.Tensor):
            loss.backward()
        else:
            print("  [info] Could not find a differentiable loss — "
                  "backward skipped; train memory may be underestimated.")

    except Exception as exc:
        print(f"  [warn] Training forward/backward raised: {exc}")
        print("         mem_train_* values may be underestimated.")
    finally:
        torch.cuda.synchronize(device)
        model.zero_grad(set_to_none=True)
        model.eval()
        torch.cuda.empty_cache()

    return {
        "mem_train_alloc_gb":  round(torch.cuda.max_memory_allocated(device) / 1024 ** 3, 4),
        "mem_train_reserv_gb": round(torch.cuda.max_memory_reserved(device)  / 1024 ** 3, 4),
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
    """
    CUDA-event-timed inference latency at bs=1.
    Batches are cycled so n_measure > len(batches) works correctly.
    """
    model.eval()
    torch.cuda.empty_cache()

    def _get_batch(i: int) -> dict:
        return move_batch_to_gpu(batches[i % len(batches)], device)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    # Discard warmup passes
    for i in range(n_warmup):
        _ = model(_get_batch(i))
    torch.cuda.synchronize(device)

    times: List[float] = []
    for i in range(n_measure):
        b = _get_batch(i)
        start_evt.record()
        _ = model(b)
        end_evt.record()
        torch.cuda.synchronize(device)
        times.append(start_evt.elapsed_time(end_evt))

    t = np.array(times, dtype=np.float64)
    mean_ms = float(np.mean(t))
    fps = round(1000.0 / mean_ms, 2) if mean_ms > 0 else float("nan")

    return {
        "latency_mean_ms":    round(mean_ms,                         2),
        "latency_std_ms":     round(float(np.std(t)),                2),
        "latency_min_ms":     round(float(np.min(t)),                2),
        "latency_p50_ms":     round(float(np.percentile(t, 50)),     2),
        "latency_p95_ms":     round(float(np.percentile(t, 95)),     2),
        "latency_max_ms":     round(float(np.max(t)),                2),
        "fps":                fps,
        "latency_n_runs":     n_measure,
        "latency_batch_size": 1,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Model disk footprint
# ══════════════════════════════════════════════════════════════════════════════

def measure_model_size_mb(weight_path: Optional[str]) -> Dict[str, object]:
    if weight_path is None or not os.path.isfile(weight_path):
        return {
            "model_size_mb":   0.0,
            "model_size_note": "random weights (no checkpoint)",
        }
    size_mb = round(os.path.getsize(weight_path) / (1024 ** 2), 2)
    return {
        "model_size_mb":   size_mb,
        "model_size_note": f"checkpoint: {os.path.basename(weight_path)}",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Derived efficiency metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_efficiency_metrics(
    results: dict,
    baseline_params_m:   Optional[float],
    baseline_latency_ms: Optional[float],
    baseline_gflops:     Optional[float],
) -> Dict[str, object]:
    """
    Compute paper-table efficiency metrics that require a baseline reference.
    Ratios > 1 mean the model is better (smaller/faster) than the baseline.
    """
    metrics: Dict[str, object] = {}

    params_m = results.get("params_total_M")
    gflops   = results.get("gflops_total")
    lat_ms   = results.get("latency_mean_ms")
    fps      = results.get("fps")
    mem_gb   = results.get("mem_infer_alloc_gb")

    # ── Compression (parameters) ──────────────────────────────────────────────
    if baseline_params_m and params_m and not math.isnan(float(params_m)):
        metrics["compression_ratio"]    = round(baseline_params_m / float(params_m), 3)
        metrics["params_reduction_pct"] = round(
            100.0 * (1.0 - float(params_m) / baseline_params_m), 1)

    # ── FLOPs reduction ───────────────────────────────────────────────────────
    if baseline_gflops and gflops and not math.isnan(float(gflops)) and float(gflops) > 0:
        metrics["flops_reduction_ratio"] = round(baseline_gflops / float(gflops), 3)
        metrics["flops_reduction_pct"]   = round(
            100.0 * (1.0 - float(gflops) / baseline_gflops), 1)

    # ── Latency speedup ───────────────────────────────────────────────────────
    if baseline_latency_ms and lat_ms and not math.isnan(float(lat_ms)) and float(lat_ms) > 0:
        metrics["speedup_vs_baseline"]  = round(baseline_latency_ms / float(lat_ms), 3)
        metrics["latency_reduction_pct"] = round(
            100.0 * (1.0 - float(lat_ms) / baseline_latency_ms), 1)

    # ── Compute efficiency: GFLOPs / param(M) ────────────────────────────────
    if gflops and params_m and not math.isnan(float(gflops)) and float(params_m) > 0:
        metrics["gflops_per_param_m"] = round(float(gflops) / float(params_m), 4)

    # ── Memory throughput: FPS / GB ───────────────────────────────────────────
    if fps and mem_gb and not math.isnan(float(fps)) and float(mem_gb) > 0:
        metrics["fps_per_gb"] = round(float(fps) / float(mem_gb), 3)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Model builder + checkpoint loader
# ══════════════════════════════════════════════════════════════════════════════

def build_and_load(
    cfg: Config,
    weight_path: Optional[str],
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    model = build_model(cfg.model)
    model = model.to(device)
    total_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model built  ({total_m:.4f} M parameters)")

    if weight_path is None:
        print("  No checkpoint provided — using random (untrained) weights.")
        return model

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {weight_path}\n"
            "  Check that the experiment directory exists and the model has been trained."
        )

    print(f"  Loading checkpoint : {weight_path}")
    print(f"  strict={strict}  {'(KD mode: teacher/projector keys will be ignored)' if not strict else ''}")

    # weights_only=False needed for checkpoints that embed non-tensor metadata
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    except TypeError:
        # PyTorch < 2.0 does not have weights_only argument
        ckpt = torch.load(weight_path, map_location="cpu")

    state = ckpt.get("state_dict", ckpt)

    # Strip DataParallel / DDP "module." prefix
    new_state: Dict[str, torch.Tensor] = OrderedDict(
        (k[len("module."):] if k.startswith("module.") else k, v)
        for k, v in state.items()
    )

    info = model.load_state_dict(new_state, strict=strict)

    if info.missing_keys:
        n = len(info.missing_keys)
        sample = info.missing_keys[:5]
        level = "warn" if strict else "info"
        print(f"  [{level}] {n} missing key(s) (showing up to 5): {sample}")

    if info.unexpected_keys:
        n = len(info.unexpected_keys)
        sample = info.unexpected_keys[:5]
        level = "warn" if strict else "info"
        print(f"  [{level}] {n} unexpected key(s) (showing up to 5): {sample}")
        if not strict:
            print("  [info] Unexpected keys are expected for KD checkpoints "
                  "(teacher backbone + projectors are discarded at inference).")

    epoch_str = str(ckpt.get("epoch", "?"))
    best_val  = ckpt.get("best_metric_value", None)
    best_str  = f"  |  best_val={best_val:.4f}" if isinstance(best_val, float) else ""
    print(f"  Checkpoint loaded  (epoch {epoch_str}{best_str})")

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Validation data loader
# ══════════════════════════════════════════════════════════════════════════════

def build_profile_batches(cfg: Config, n_batches: int) -> List[dict]:
    """CPU-resident collated batches from the val split (GPU transfer is lazy)."""
    print(f"  Loading up to {n_batches} val batches …")
    val_dataset = build_dataset(cfg.data.val)
    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 4),
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
#  Config serialiser  (for WandB config logging)
# ══════════════════════════════════════════════════════════════════════════════

def cfg_to_dict(obj: object) -> object:
    """
    Recursively convert a Pointcept Config (which subclasses dict) or any
    namespace/object to a JSON-serialisable structure.

    IMPORTANT: isinstance(dict) is checked BEFORE hasattr(__dict__) because
    Pointcept's Config subclasses dict AND has __dict__.  Without this order,
    most keys would be missed.
    """
    try:
        if isinstance(obj, dict):
            return {k: cfg_to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cfg_to_dict(v) for v in obj]
        if hasattr(obj, "__dict__"):
            return {k: cfg_to_dict(v)
                    for k, v in vars(obj).items()
                    if not k.startswith("_")}
        return obj
    except Exception:
        return str(obj)


# ══════════════════════════════════════════════════════════════════════════════
#  WandB login helper
# ══════════════════════════════════════════════════════════════════════════════

def ensure_wandb_login(api_key: Optional[str], offline: bool) -> bool:
    if offline:
        os.environ["WANDB_MODE"] = "offline"
        return True

    if api_key:
        wandb.login(key=api_key, relogin=True)
        return True

    if os.environ.get("WANDB_API_KEY", ""):
        return True

    try:
        api = wandb.Api(timeout=5)
        _ = api.viewer
        return True
    except Exception:
        pass

    print(
        "\n  [warn] WandB is not logged in and no --wandb-key was provided.\n"
        "  Fix permanently:  wandb login\n"
        "  Or per-run:       --wandb-key YOUR_API_KEY\n"
        "  Or offline mode:  --wandb-offline\n"
        "  Results will be printed to stdout instead.\n"
    )
    return False


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(val: object) -> str:
    """Format a metric value for the console summary table."""
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, float):
        if math.isnan(val):
            return "NaN"
        return f"{val:.4f}" if abs(val) < 10_000 else f"{val:.2f}"
    return str(val)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = Config.fromfile(args.config_file)
    if args.options:
        cfg.merge_from_dict(args.options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[warn] CUDA not available — memory and latency measurements will be skipped.")

    print("\n" + "=" * 72)
    print(f"  TopoPT / LitePT Profiler  —  {args.run_name}")
    print(f"  Config   : {args.config_file}")
    print(f"  Weight   : {args.weight or '(none — random weights)'}")
    print(f"  Strict   : {not args.no_strict_load}")
    print(f"  Device   : {device}")
    print(f"  N input  : {args.n_input_points:,} points  (theoretical FLOPs)")
    if args.baseline_params_m:
        print(f"  Baseline : {args.baseline_params_m} M params / "
              f"{args.baseline_gflops or '?'} GFLOPs")
    print("=" * 72 + "\n")

    # ── Check for KD config + strict load ─────────────────────────────────────
    check_kd_config_warning(cfg, strict=not args.no_strict_load)

    # ── System info ───────────────────────────────────────────────────────────
    print("▶ Collecting system info …")
    sys_info = collect_system_info()
    for k, v in sys_info.items():
        print(f"    {k}: {v}")

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n▶ Building model …")
    model = build_and_load(
        cfg, args.weight, device, strict=not args.no_strict_load
    )

    # ── Model disk footprint ──────────────────────────────────────────────────
    print("\n▶ Measuring model disk footprint …")
    size_info = measure_model_size_mb(args.weight)
    for k, v in size_info.items():
        print(f"    {k}: {v}")

    # ── Parameter count ───────────────────────────────────────────────────────
    print("\n▶ Counting parameters …")
    param_info = count_params(model)
    for k, v in param_info.items():
        print(f"    {k}: {v}")

    # ── Theoretical FLOPs ─────────────────────────────────────────────────────
    print(f"\n▶ Estimating theoretical GFLOPs "
          f"(backbone encoder only, n_input={args.n_input_points:,}) …")
    flop_info = estimate_gflops(cfg, n_input_points=args.n_input_points)
    for k, v in flop_info.items():
        print(f"    {k}: {v}")

    # ── Load profiling batches ─────────────────────────────────────────────────
    batches: List[dict] = []
    skip_runtime = device.type != "cuda" or args.skip_batches

    if not skip_runtime:
        try:
            batches = build_profile_batches(cfg, args.n_profile_batches)
        except Exception as exc:
            print(f"\n[warn] Could not load val data: {exc}")
            print("       Memory, latency, and measured-FLOPs measurements skipped.")
            skip_runtime = True

    if args.skip_batches:
        print("\n  [info] --skip-batches set — skipping all data-dependent measurements.")

    # ── Measured FLOPs (torch.profiler) ───────────────────────────────────────
    flop_measured: Dict[str, object] = {}
    if not skip_runtime and batches:
        print("\n▶ Measuring FLOPs via torch.profiler (full model, 1 batch) …")
        flop_measured = measure_flops_profiler(model, batches[0], device)
        for k, v in flop_measured.items():
            print(f"    {k}: {v}")

    # ── Inference memory ──────────────────────────────────────────────────────
    mem_infer: Dict[str, float] = {}
    if not skip_runtime and batches:
        print(f"\n▶ Measuring inference memory ({len(batches)} batches) …")
        mem_infer = measure_memory_inference(model, batches, device)
        for k, v in mem_infer.items():
            print(f"    {k}: {v:.4f} GB")

    # ── Training memory ───────────────────────────────────────────────────────
    mem_train: Dict[str, float] = {}
    if not skip_runtime and batches and not args.skip_train_memory:
        print("\n▶ Measuring training memory (1 forward + backward, AMP) …")
        mem_train = measure_memory_training(
            model, batches, device,
            amp=bool(getattr(cfg, "enable_amp", True)),
        )
        for k, v in mem_train.items():
            print(f"    {k}: {v:.4f} GB")

    # ── Latency ───────────────────────────────────────────────────────────────
    latency: Dict[str, float] = {}
    if not skip_runtime and batches:
        print(f"\n▶ Measuring latency "
              f"(warmup={args.n_warmup}, measure={args.n_measure}, bs=1) …")
        latency = measure_latency(
            model, batches, device,
            n_warmup=args.n_warmup,
            n_measure=args.n_measure,
        )
        for k, v in latency.items():
            print(f"    {k}: {v}")

    # ── Assemble full results dict ─────────────────────────────────────────────
    results: Dict[str, object] = {}
    results.update(param_info)
    results.update(size_info)
    results.update(flop_info)
    results.update(flop_measured)
    results.update(mem_infer)
    results.update(mem_train)
    results.update(latency)

    # Metadata
    results["config_file"]         = args.config_file
    results["weight_path"]         = args.weight or "random"
    results["weight_loaded"]       = args.weight is not None
    results["strict_load"]         = not args.no_strict_load
    results["n_warmup"]            = args.n_warmup
    results["n_measure"]           = args.n_measure
    results["n_profile_batches"]   = args.n_profile_batches
    results["n_input_points"]      = args.n_input_points
    results["skip_batches"]        = args.skip_batches
    results["skip_train_memory"]   = args.skip_train_memory

    # Backbone config summary (makes WandB table self-contained)
    try:
        bcfg = cfg.model.backbone
        results["cfg_enc_channels"]  = str(list(bcfg.enc_channels))
        results["cfg_enc_depths"]    = str(list(bcfg.enc_depths))
        results["cfg_enc_patch_sz"]  = str(list(bcfg.enc_patch_size))
        results["cfg_enc_attn"]      = str(list(bcfg.enc_attn))
        results["cfg_in_channels"]   = int(getattr(bcfg, "in_channels", -1))
        results["cfg_num_stages"]    = len(list(bcfg.enc_depths))
    except AttributeError as e:
        results["cfg_note"] = f"Could not read backbone config fields: {e}"

    # ── Derived efficiency metrics ─────────────────────────────────────────────
    print("\n▶ Computing efficiency metrics …")
    eff_metrics = compute_efficiency_metrics(
        results,
        baseline_params_m    = args.baseline_params_m,
        baseline_latency_ms  = args.baseline_latency_ms,
        baseline_gflops      = args.baseline_gflops,
    )
    results.update(eff_metrics)
    if eff_metrics:
        for k, v in eff_metrics.items():
            print(f"    {k}: {v}")
    else:
        print("    (no --baseline-* flags provided; relative metrics skipped)")

    # ── WandB logging ─────────────────────────────────────────────────────────
    print("\n▶ Logging to WandB …")
    wandb_ready = ensure_wandb_login(args.wandb_key, args.wandb_offline)

    base_tags = ["profile"]
    if args.extra_tags:
        base_tags += [t.strip() for t in args.extra_tags.split(",") if t.strip()]

    if wandb_ready:
        try:
            try:
                backbone_cfg_dict = cfg_to_dict(cfg.model.backbone)
            except Exception as e:
                backbone_cfg_dict = {"error": str(e)}

            run = wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config={
                    "backbone_cfg": backbone_cfg_dict,
                    "system":       sys_info,
                    "profiling": {
                        "n_warmup":           args.n_warmup,
                        "n_measure":          args.n_measure,
                        "n_profile_batches":  args.n_profile_batches,
                        "n_input_points":     args.n_input_points,
                        "skip_batches":       args.skip_batches,
                        "skip_train_memory":  args.skip_train_memory,
                        "strict_load":        not args.no_strict_load,
                        "batch_size":         1,
                    },
                    "baselines": {
                        "params_m":    args.baseline_params_m,
                        "latency_ms":  args.baseline_latency_ms,
                        "gflops":      args.baseline_gflops,
                    },
                },
                tags=base_tags,
            )

            # Numeric scalars → single WandB step (appears in charts)
            numeric = {
                k: v for k, v in results.items()
                if isinstance(v, (int, float))
                and not isinstance(v, bool)
                and not math.isnan(float(v))
            }
            wandb.log(numeric)

            # All values → run summary (visible in the runs table)
            for k, v in results.items():
                if isinstance(v, (str, bool)):
                    wandb.run.summary[k] = v
                elif isinstance(v, float) and math.isnan(v):
                    wandb.run.summary[k] = "N/A"
                elif isinstance(v, (int, float)):
                    wandb.run.summary[k] = v

            # System info with sys/ prefix in summary
            for k, v in sys_info.items():
                wandb.run.summary[f"sys/{k}"] = v

            wandb.finish()
            print("  WandB run logged successfully ✓")

        except Exception as exc:
            print(f"  [warn] WandB logging failed: {exc}")
            print("  Printing full results to stdout instead.")
            wandb_ready = False

    if not wandb_ready:
        print("  Full results:")
        for k, v in results.items():
            print(f"    {k}: {v}")

    # ── Console summary table ─────────────────────────────────────────────────
    # Per-stage GFLOPs (insert dynamically based on how many stages exist)
    num_stages = results.get("cfg_num_stages", 5)
    stage_rows = [
        (f"  GFLOPs stage {s}", f"gflops_stage{s}")
        for s in range(int(num_stages) if isinstance(num_stages, (int, float)) else 5)
    ]

    key_metrics = [
        ("Params total (M)",            "params_total_M"),
        ("Params trainable (M)",        "params_trainable_M"),
        ("Model size (MB)",             "model_size_mb"),
        ("─────────── FLOPs ──────────────────────────────────────", None),
        ("GFLOPs theoretical (encoder)","gflops_total"),
        *stage_rows,
        ("GFLOPs measured (full model)","gflops_measured"),
        ("─────────── Latency ─────────────────────────────────────", None),
        ("Latency mean (ms)",           "latency_mean_ms"),
        ("Latency std  (ms)",           "latency_std_ms"),
        ("Latency p95  (ms)",           "latency_p95_ms"),
        ("FPS",                         "fps"),
        ("─────────── Memory ──────────────────────────────────────", None),
        ("Mem infer alloc (GB)",        "mem_infer_alloc_gb"),
        ("Mem train alloc (GB)",        "mem_train_alloc_gb"),
        ("─────────── vs Baseline ─────────────────────────────────", None),
        ("Compression ratio (params)",  "compression_ratio"),
        ("Params reduction (%)",        "params_reduction_pct"),
        ("FLOPs reduction ratio",       "flops_reduction_ratio"),
        ("FLOPs reduction (%)",         "flops_reduction_pct"),
        ("Speedup vs baseline",         "speedup_vs_baseline"),
    ]

    print("\n" + "=" * 72)
    print(f"  SUMMARY  —  {args.run_name}")
    print("=" * 72)
    for label, key in key_metrics:
        if key is None:
            print(f"  {label}")
            continue
        val = results.get(key, "N/A")
        print(f"  {label:<40} {_fmt(val)}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()