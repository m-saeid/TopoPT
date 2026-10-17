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

COMPUTE  (two complementary figures)
  gflops_total                — theoretical backbone-encoder-only estimate (formula below)
  gflops_stage{0..4}          — per-stage breakdown of the theoretical estimate
  gflops_measured             — torch.profiler with_flops=True (full model, bs=1;
                                uses PyTorch's built-in flop-counting infrastructure,
                                NOT torch.utils.flop_counter.FlopCounterMode)
  gflops_per_param_m          — compute efficiency (theoretical / params)

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
  key package versions (spconv, flash_attn, timm, torch_scatter, pointops, fvcore)

═══════════════════════════════════════════════════════════════════════════════
FLOPs formula  (backbone encoder only — relative comparison across ablations)
═══════════════════════════════════════════════════════════════════════════════
All counts are multiply-accumulate (MAC) × 2 → FLOPs.
Token count per stage: N₀ = n_input_points; Nₛ = Nₛ₋₁ // 4  (stride-2 GridPool → ÷4).

Conv block  (SubMConv3d(C,C,3) + pointwise residual projection):
    FLOPs = N × (27 × OCC × C² + C²) × 2     [OCC = 0.25, sparse occupancy]
    — 27×OCC×C²: sparse 3×3×3 convolution at empirical occupancy 0.25
    — C²:        pointwise (1×1) projection / residual linear
    — ×2:        MACs → FLOPs

Attn block  (QKV proj + windowed attention + output proj + MLP):
    FLOPs = N × [(4 + 2·ratio) · C² + 2·P·C] × 2
    Breakdown:
      QKV(3C²) + OutProj(C²) + MLP(2·ratio·C²) + Attn-scores(P·C) + Attn-values(P·C)

NOTE: This estimate covers only the backbone encoder.  LayerNorm, PointROPE,
GridPooling, the decoder, and the task head are excluded — the estimate is designed
for reproducible relative comparison across ablation configs, not as an absolute
wall-clock predictor.

Use gflops_measured (torch.profiler with_flops=True) for a figure that covers the
decoder and task head, but be aware that it excludes custom CUDA kernels
(PointROPE, flash_attn, spconv SubMConv3d) which are not registered with
PyTorch's flop-counting infrastructure.  Both figures are lower bounds; report them
as complementary estimates and note their respective exclusions.

PointROPE constraint:  head_dim % 6 == 0
(3D RoPE pairs frequencies for x, y, z axes → must be divisible by 6)
LitePT-S uses head_dim = 18 everywhere (18 % 6 = 0 ✓).

═══════════════════════════════════════════════════════════════════════════════
Usage
═══════════════════════════════════════════════════════════════════════════════
# Random weights — architecture cost only (fast; no dataset required with --skip-batches):
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

# KD checkpoint — student weights are embedded in the distillation checkpoint;
# ALWAYS pass the BASE STUDENT config (not the KD config) so build_model()
# only constructs the student graph.  Use --no-strict-load to suppress
# unexpected teacher/projector key warnings.
    python tools/profile_model.py \\
        --config-file configs/scannet/semseg-lw-c-100epoch.py \\
        --weight      exp/scannet-semseg-lw-c-kd-100epoch/model/model_best.pth \\
        --no-strict-load \\
        --run-name    scannet/lw-c-kd-trained \\
        --wandb-project LitePT-Profiling

# Skip all data-dependent measurements (params + theoretical FLOPs only, much faster):
    python tools/profile_model.py \\
        --config-file configs/scannet/semseg-lw-c-100epoch.py \\
        --skip-batches \\
        --run-name    scannet/lw-c-arch-only

# LitePT baseline (teacher) checkpoint — stored under exp/litept/:
    python tools/profile_model.py \\
        --config-file configs/scannet/semseg-litept-small-v1m1.py \\
        --weight      exp/litept/scannet-semseg-litept-small-v1m1/model/model_best.pth \\
        --run-name    scannet/litept-small-trained \\
        --wandb-project LitePT-Profiling
"""

from __future__ import annotations

import os
import sys
import math
import platform
import subprocess
import argparse
from collections import OrderedDict
from typing import Dict, List, Optional

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
    p.add_argument(
        "--config-file", required=True,
        help=(
            "Path to model config (.py). "
            "For KD trained checkpoints, ALWAYS pass the BASE STUDENT config "
            "so build_model() only constructs the student inference graph. "
            "For LitePT baselines, pass configs/<dataset>/semseg-litept-*.py directly."
        ),
    )
    p.add_argument(
        "--run-name", required=True,
        help="WandB run name, e.g. 'scannet/lw-c-random'.",
    )

    # ── Checkpoint ────────────────────────────────────────────────────────────
    p.add_argument(
        "--weight", default=None,
        help=(
            "Checkpoint path (.pth). Omit for random (untrained) weights. "
            "LitePT baselines live under exp/litept/<dataset>-<stem>/model/model_best.pth. "
            "Student / KD checkpoints live under exp/<dataset>-<stem>/model/model_best.pth."
        ),
    )
    p.add_argument(
        "--no-strict-load", action="store_true",
        help=(
            "Use strict=False when loading the checkpoint. "
            "Required for KD checkpoints: the saved state dict contains teacher "
            "backbone and projector keys which are unexpected in the inference-time "
            "student model. Those keys are safely ignored."
        ),
    )

    # ── WandB ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--wandb-project", default="LitePT-Profiling",
        help="WandB project name (default: LitePT-Profiling).",
    )
    p.add_argument(
        "--wandb-key", default=None,
        help="WandB API key. If None, uses the current `wandb login` session or WANDB_API_KEY env var.",
    )
    p.add_argument(
        "--extra-tags", default="",
        help="Comma-separated additional WandB tags, e.g. 'semseg,lw-c,kd'.",
    )
    p.add_argument(
        "--wandb-offline", action="store_true",
        help="Run WandB in offline mode (useful on compute nodes without internet).",
    )

    # ── Measurement settings ──────────────────────────────────────────────────
    p.add_argument(
        "--n-warmup", type=int, default=10,
        help="Latency warmup iterations, discarded (default: 10).",
    )
    p.add_argument(
        "--n-measure", type=int, default=50,
        help="Latency measurement iterations (default: 50).",
    )
    p.add_argument(
        "--n-profile-batches", type=int, default=50,
        help="Val batches loaded for memory/latency measurements (default: 50).",
    )
    p.add_argument(
        "--n-input-points", type=int, default=60_000,
        help=(
            "Assumed input point count for theoretical FLOPs estimate (default: 60000). "
            "Use 60000 for ScanNet/ScanNet200 indoor scenes; 80000 for nuScenes outdoor."
        ),
    )
    p.add_argument(
        "--skip-batches", action="store_true",
        help=(
            "Skip all data-dependent measurements (memory, latency, measured FLOPs). "
            "Only params and theoretical FLOPs are computed. Much faster; "
            "suitable for architecture-only comparisons."
        ),
    )
    p.add_argument(
        "--skip-train-memory", action="store_true",
        help="Skip training-mode memory measurement (saves ~30s per model).",
    )

    # ── Baseline comparison ───────────────────────────────────────────────────
    p.add_argument(
        "--baseline-params-m", type=float, default=None,
        help=(
            "Baseline model parameter count (M) for compression_ratio and "
            "params_reduction_pct. Example: 12.71 for LitePT-S."
        ),
    )
    p.add_argument(
        "--baseline-latency-ms", type=float, default=None,
        help=(
            "Baseline model mean latency (ms) for speedup_vs_baseline. "
            "Obtained from a prior profiler run of the teacher / reference model."
        ),
    )
    p.add_argument(
        "--baseline-gflops", type=float, default=None,
        help=(
            "Baseline theoretical GFLOPs for flops_reduction_ratio. "
            "Example: 25.42 for LitePT-S."
        ),
    )

    # ── Config overrides ──────────────────────────────────────────────────────
    p.add_argument(
        "--options", nargs="+", action=DictAction,
        help="Override config options (same key=value syntax as train.py).",
    )

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  System information
# ══════════════════════════════════════════════════════════════════════════════

def collect_system_info() -> Dict[str, object]:
    """Full system fingerprint — disambiguates runs across machines and driver versions."""
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
    Emit a clear warning if a KD/distillation config (one that contains a
    teacher_backbone field) is paired with strict=True.

    This combination will almost certainly raise a RuntimeError at load time
    because the student checkpoint embeds teacher + projector keys that are
    unexpected in the base student model graph.

    The correct workflow is either:
      (a) pass the BASE STUDENT config via --config-file, OR
      (b) add --no-strict-load to suppress unexpected-key errors.
    """
    has_teacher = hasattr(cfg, "model") and hasattr(cfg.model, "teacher_backbone")
    if has_teacher and strict:
        print(
            "\n  [WARN] The config contains a 'teacher_backbone' field — this is a "
            "KD/distillation config.\n"
            "         For profiling a trained KD checkpoint you should either:\n"
            "           (a) pass the BASE STUDENT config via --config-file\n"
            "               so that build_model() only constructs the student graph, OR\n"
            "           (b) add --no-strict-load to suppress unexpected-key errors from\n"
            "               teacher backbone + projector weights in the checkpoint.\n"
            "         Continuing with strict=True — this will raise RuntimeError if "
            "a --weight is supplied.\n"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Parameter counting
# ══════════════════════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> Dict[str, float]:
    """Count total, trainable, and buffer parameters in millions."""
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

    Purpose: RELATIVE comparison across ablation configs.
    Excludes: LayerNorm, PointROPE, GridPooling, decoder, task head.
    Use gflops_measured (torch.profiler) for a broader absolute figure.

    Token count:  N₀ = n_input_points;  Nₛ = Nₛ₋₁ // 4  (stride-2 pool → ÷4/stage).

    Conv block (SubMConv3d(C,C,3) + pointwise projection per block):
        FLOPs = N × (27 × OCC × C² + C²) × 2
        — 27×OCC×C²: sparse 3×3×3 conv at occupancy OCC (empirical: ~0.25)
        — C²:        pointwise (1×1) projection / residual linear
        — ×2:        MACs → FLOPs

    Attn block (QKV + windowed self-attn + output proj + MLP):
        FLOPs = N × [(4 + 2·ratio) · C² + 2·P·C] × 2
        — 3C²:       Q, K, V projections
        — C²:        output projection
        — 2·ratio·C²: MLP (two linear layers, expansion ratio=mlp_ratio)
        — 2·P·C:     attention scores Q@K^T and weighted sum attn@V
                     (P = patch_size points, serialized window attention)
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

    # Empirical sparse occupancy for both indoor (ScanNet) and outdoor (nuScenes) scenes.
    # Indoor scenes tend to be ~0.20-0.30; outdoor LiDAR tends to be ~0.15-0.25.
    # 0.25 is a reasonable mid-point for relative comparisons.
    CONV_OCC = 0.25

    per_stage: Dict[str, float] = {}
    total_flops = 0.0
    n = n_input_points

    for s in range(num_stages):
        if s > 0:
            # stride-2 GridPool in each of 2 spatial dims → point count ÷ 4
            n = max(n // 4, 1)
        C = enc_channels[s]
        P = enc_patch_sz[s]
        D = enc_depths[s]
        stage_flops = 0.0

        for _ in range(D):
            if enc_conv[s]:
                # SubMConv3d(C, C, 3) + pointwise projection
                stage_flops += n * (27.0 * CONV_OCC * C * C + C * C) * 2.0
            if enc_attn[s]:
                # QKV(3C²) + OutProj(C²) + MLP(2·ratio·C²) + Attn-scores(P·C) + Attn-values(P·C)
                stage_flops += n * ((4.0 + 2.0 * mlp_ratio) * C * C + 2.0 * P * C) * 2.0

        per_stage[f"gflops_stage{s}"] = round(stage_flops / 1e9, 4)
        total_flops += stage_flops

    result: Dict[str, object] = {"gflops_total": round(total_flops / 1e9, 4)}
    result.update(per_stage)
    result["gflops_n_input_pts"] = n_input_points
    result["gflops_note"] = (
        f"Backbone-encoder-only theoretical estimate (relative comparison only). "
        f"Conv occ={CONV_OCC}, n_input={n_input_points:,}, stride-4 pooling per stage. "
        "Excludes: LayerNorm, PointROPE, GridPooling, decoder, task head. "
        "Complement with gflops_measured (torch.profiler) for a broader figure."
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

    This uses PyTorch's built-in operator-level flop counting (registered via
    torch.jit._overload / _ProfilerFunction), NOT the separate
    torch.utils.flop_counter.FlopCounterMode API.  Both count standard linear
    algebra ops, but neither captures custom CUDA extensions.

    Coverage:  encoder + decoder + task head (all standard PyTorch ops).
    Excluded:  PointROPE, flash_attn, spconv SubMConv3d (custom CUDA kernels
               not registered with torch.profiler's flop counter).

    For NeurIPS reporting: present both gflops_total (theoretical, encoder-only,
    reproducible) and gflops_measured (broader but lower-bound) and note their
    respective exclusions.  The theoretical figure is preferred for relative
    ablation comparison because it is formula-based and implementation-independent.
    """
    model.eval()
    b = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
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
                "torch.profiler with_flops=True — covers encoder + decoder + task head "
                "(all standard PyTorch ops). Excludes custom CUDA kernels: "
                "PointROPE, flash_attn, spconv SubMConv3d. "
                "Actual cost is higher. Complements gflops_total (theoretical encoder estimate)."
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
    """Transfer all tensor values in a batch dict to the given device."""
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
    """
    Peak GPU memory allocated and reserved during eval-mode forward passes.

    Iterates all supplied batches so peak reflects a realistic scene distribution
    rather than a single lucky (small) scene.
    """
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
    Peak GPU memory during one training forward + backward pass (no optimiser step).

    Handles multiple model output formats:
      - dict with "loss" key                        (most semseg models)
      - dict with multiple tensor values            (PG-v1m2 instance seg and similar)
      - direct Tensor output

    AMP context is selected for compatibility with PyTorch ≥ 2.0 and older:
      - PyTorch ≥ 2.0: torch.amp.autocast(device_type="cuda", ...)
      - PyTorch < 2.0: torch.cuda.amp.autocast(...)
    """
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    batch = move_batch_to_gpu(batches[0], device)
    loss: Optional[torch.Tensor] = None

    try:
        # Select AMP context compatible with both PyTorch ≥ 2.0 and older APIs
        try:
            ctx = torch.amp.autocast(device_type="cuda", enabled=amp, dtype=torch.float16)
        except TypeError:
            # PyTorch < 2.0
            import torch.cuda.amp as _amp_compat  # type: ignore[import]
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
                # Sum all grad-requiring tensors (covers PG-v1m2 and similar heads)
                candidates = [
                    v.mean()
                    for v in output.values()
                    if isinstance(v, torch.Tensor) and v.requires_grad and v.numel() > 0
                ]
                if candidates:
                    loss = sum(candidates)  # type: ignore[arg-type]

        if loss is not None and isinstance(loss, torch.Tensor):
            loss.backward()
        else:
            print(
                "  [info] Could not find a differentiable loss in the model output — "
                "backward skipped; mem_train_* values may be underestimated."
            )

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

    Uses torch.cuda.Event for timing rather than time.time() to avoid CPU-GPU
    synchronisation artefacts.  Batches are cycled when n_measure > len(batches).
    Warmup passes are fully discarded before recording begins.
    """
    model.eval()
    torch.cuda.empty_cache()

    def _get_batch(i: int) -> dict:
        return move_batch_to_gpu(batches[i % len(batches)], device)

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    # Discard warmup passes to allow GPU clocks and caches to stabilise
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
        "latency_mean_ms":    round(mean_ms,                        2),
        "latency_std_ms":     round(float(np.std(t)),               2),
        "latency_min_ms":     round(float(np.min(t)),               2),
        "latency_p50_ms":     round(float(np.percentile(t, 50)),    2),
        "latency_p95_ms":     round(float(np.percentile(t, 95)),    2),
        "latency_max_ms":     round(float(np.max(t)),               2),
        "fps":                fps,
        "latency_n_runs":     n_measure,
        "latency_batch_size": 1,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Model disk footprint
# ══════════════════════════════════════════════════════════════════════════════

def measure_model_size_mb(weight_path: Optional[str]) -> Dict[str, object]:
    """Return on-disk checkpoint size in MB, or 0 for random weights."""
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

    All ratios > 1 mean the model is better (smaller / faster) than the baseline.
    All pct values are positive when the model improves over baseline.

    These metrics are only computed when the corresponding --baseline-* flag is set.
    """
    metrics: Dict[str, object] = {}

    params_m = results.get("params_total_M")
    gflops   = results.get("gflops_total")
    lat_ms   = results.get("latency_mean_ms")
    fps      = results.get("fps")
    mem_gb   = results.get("mem_infer_alloc_gb")

    def _valid(x: object) -> bool:
        """True if x is a finite number."""
        try:
            return x is not None and not math.isnan(float(x))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    # ── Compression (parameters) ──────────────────────────────────────────────
    if baseline_params_m and _valid(params_m):
        metrics["compression_ratio"]    = round(baseline_params_m / float(params_m), 3)
        metrics["params_reduction_pct"] = round(
            100.0 * (1.0 - float(params_m) / baseline_params_m), 1)

    # ── FLOPs reduction ───────────────────────────────────────────────────────
    if baseline_gflops and _valid(gflops) and float(gflops) > 0:
        metrics["flops_reduction_ratio"] = round(baseline_gflops / float(gflops), 3)
        metrics["flops_reduction_pct"]   = round(
            100.0 * (1.0 - float(gflops) / baseline_gflops), 1)

    # ── Latency speedup ───────────────────────────────────────────────────────
    if baseline_latency_ms and _valid(lat_ms) and float(lat_ms) > 0:
        metrics["speedup_vs_baseline"]    = round(baseline_latency_ms / float(lat_ms), 3)
        metrics["latency_reduction_pct"]  = round(
            100.0 * (1.0 - float(lat_ms) / baseline_latency_ms), 1)

    # ── Compute efficiency: GFLOPs / param(M) ────────────────────────────────
    if _valid(gflops) and _valid(params_m) and float(params_m) > 0:
        metrics["gflops_per_param_m"] = round(float(gflops) / float(params_m), 4)

    # ── Memory throughput: FPS / GB ───────────────────────────────────────────
    if _valid(fps) and _valid(mem_gb) and float(mem_gb) > 0:
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
    """
    Build the model from config and optionally load a checkpoint.

    Checkpoint loading logic:
    1. Attempt torch.load with weights_only=False (needed for checkpoints that
       embed non-tensor metadata such as epoch number and best_metric_value).
    2. Fall back to the PyTorch < 2.0 signature (no weights_only argument).
    3. Extract state_dict from the "state_dict" key if present; otherwise treat
       the raw checkpoint dict as the state dict.
    4. Strip "module." prefix produced by DataParallel / DistributedDataParallel.
    5. Call load_state_dict with the requested strict setting.

    For KD checkpoints: use strict=False and pass the BASE STUDENT config to
    build_model(). The teacher backbone and projector keys will appear as
    unexpected keys and are safely ignored when strict=False.

    Checkpoint paths:
      LitePT baselines:  exp/litept/<dataset>-<stem>/model/model_best.pth
      Student / TopoPT:  exp/<dataset>-<stem>/model/model_best.pth
    """
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
            f"  LitePT baselines: exp/litept/<dataset>-<stem>/model/model_best.pth\n"
            f"  Student runs:     exp/<dataset>-<stem>/model/model_best.pth\n"
            f"  Check that the experiment directory exists and training has completed."
        )

    print(f"  Loading checkpoint : {weight_path}")
    print(
        f"  strict={strict}"
        + ("  (KD mode: teacher/projector keys will be ignored)" if not strict else "")
    )

    # weights_only=False is required for checkpoints that embed non-tensor metadata.
    # Fall back to the PyTorch < 2.0 API if the argument is not accepted.
    try:
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(weight_path, map_location="cpu")

    # Support both wrapped ({"state_dict": {...}}) and unwrapped checkpoints
    state = ckpt.get("state_dict", ckpt)

    # Strip DataParallel / DistributedDataParallel "module." prefix
    new_state: Dict[str, torch.Tensor] = OrderedDict(
        (k[len("module."):] if k.startswith("module.") else k, v)
        for k, v in state.items()
    )

    info = model.load_state_dict(new_state, strict=strict)

    if info.missing_keys:
        n      = len(info.missing_keys)
        sample = info.missing_keys[:5]
        level  = "warn" if strict else "info"
        print(f"  [{level}] {n} missing key(s) (showing up to 5): {sample}")

    if info.unexpected_keys:
        n      = len(info.unexpected_keys)
        sample = info.unexpected_keys[:5]
        level  = "warn" if strict else "info"
        print(f"  [{level}] {n} unexpected key(s) (showing up to 5): {sample}")
        if not strict:
            print(
                "  [info] Unexpected keys are expected for KD checkpoints — "
                "teacher backbone + projectors are discarded at inference."
            )

    epoch_str = str(ckpt.get("epoch", "?"))
    best_val  = ckpt.get("best_metric_value", None)
    best_str  = f"  |  best_val={best_val:.4f}" if isinstance(best_val, float) else ""
    print(f"  Checkpoint loaded  (epoch {epoch_str}{best_str})")

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Validation data loader
# ══════════════════════════════════════════════════════════════════════════════

def build_profile_batches(cfg: Config, n_batches: int) -> List[dict]:
    """
    Pre-load n_batches validation samples as CPU-resident collated dicts.

    GPU transfer is deferred to measurement time (lazy / non-blocking) to
    avoid inflating the memory baseline before profiling begins.
    """
    print(f"  Loading up to {n_batches} val batches …")
    val_dataset = build_dataset(cfg.data.val)
    loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 4),
        pin_memory=False,       # pinning happens at transfer time, not here
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
    namespace/object into a JSON-serialisable structure.

    IMPORTANT: isinstance(obj, dict) is checked BEFORE hasattr(obj, "__dict__")
    because Pointcept's Config subclasses dict AND has __dict__.  Without this
    ordering, all dict-level keys would be missed (only __dict__ keys returned).
    """
    try:
        if isinstance(obj, dict):
            return {k: cfg_to_dict(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [cfg_to_dict(v) for v in obj]
        if hasattr(obj, "__dict__"):
            return {
                k: cfg_to_dict(v)
                for k, v in vars(obj).items()
                if not k.startswith("_")
            }
        return obj
    except Exception:
        return str(obj)


# ══════════════════════════════════════════════════════════════════════════════
#  WandB login helper
# ══════════════════════════════════════════════════════════════════════════════

def ensure_wandb_login(api_key: Optional[str], offline: bool) -> bool:
    """
    Ensure WandB is ready.  Returns True if logging will succeed.

    Priority:
      1. --wandb-offline flag → set WANDB_MODE=offline and return True
      2. --wandb-key flag     → call wandb.login(key=...)
      3. WANDB_API_KEY env var → already picked up by the wandb library
      4. Existing wandb login session (wandb.Api().viewer)
      5. No credentials found  → print instructions, return False
         (results are printed to stdout instead)
    """
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
        "  Or env var:       export WANDB_API_KEY=YOUR_API_KEY\n"
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
        print(
            "[warn] CUDA is not available — memory and latency measurements will be "
            "skipped.  Only parameter counts and theoretical FLOPs will be reported."
        )

    print("\n" + "=" * 72)
    print(f"  TopoPT / LitePT Profiler  —  {args.run_name}")
    print(f"  Config   : {args.config_file}")
    print(f"  Weight   : {args.weight or '(none — random weights)'}")
    print(f"  Strict   : {not args.no_strict_load}")
    print(f"  Device   : {device}")
    print(f"  N input  : {args.n_input_points:,} points  (theoretical FLOPs estimate)")
    if args.baseline_params_m:
        print(
            f"  Baseline : {args.baseline_params_m} M params"
            + (f" / {args.baseline_gflops} GFLOPs" if args.baseline_gflops else "")
            + (f" / {args.baseline_latency_ms} ms" if args.baseline_latency_ms else "")
        )
    print("=" * 72 + "\n")

    # ── Check for KD config + strict load (common user error) ─────────────────
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
    print(
        f"\n▶ Estimating theoretical GFLOPs "
        f"(backbone encoder only, n_input={args.n_input_points:,}) …"
    )
    flop_info = estimate_gflops(cfg, n_input_points=args.n_input_points)
    for k, v in flop_info.items():
        print(f"    {k}: {v}")

    # ── Decide whether to run data-dependent measurements ─────────────────────
    skip_runtime = device.type != "cuda" or args.skip_batches
    batches: List[dict] = []

    if args.skip_batches:
        print("\n  [info] --skip-batches set — skipping all data-dependent measurements.")

    if not skip_runtime:
        print(f"\n▶ Loading val batches ({args.n_profile_batches} max) …")
        try:
            batches = build_profile_batches(cfg, args.n_profile_batches)
        except Exception as exc:
            print(f"\n  [warn] Could not load val data: {exc}")
            print("         Memory, latency, and measured-FLOPs measurements will be skipped.")
            skip_runtime = True

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
        print(
            f"\n▶ Measuring latency "
            f"(warmup={args.n_warmup}, measure={args.n_measure}, bs=1) …"
        )
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
    results["config_file"]          = args.config_file
    results["weight_path"]          = args.weight or "random"
    results["weight_loaded"]        = args.weight is not None
    results["strict_load"]          = not args.no_strict_load
    results["n_warmup"]             = args.n_warmup
    results["n_measure"]            = args.n_measure
    results["n_profile_batches"]    = args.n_profile_batches
    results["n_input_points"]       = args.n_input_points
    results["skip_batches"]         = args.skip_batches
    results["skip_train_memory"]    = args.skip_train_memory

    # Backbone config summary (makes WandB run table self-contained without opening configs)
    try:
        bcfg = cfg.model.backbone
        results["cfg_enc_channels"] = str(list(bcfg.enc_channels))
        results["cfg_enc_depths"]   = str(list(bcfg.enc_depths))
        results["cfg_enc_patch_sz"] = str(list(bcfg.enc_patch_size))
        results["cfg_enc_attn"]     = str(list(bcfg.enc_attn))
        results["cfg_in_channels"]  = int(getattr(bcfg, "in_channels", -1))
        results["cfg_num_stages"]   = len(list(bcfg.enc_depths))
    except AttributeError as e:
        results["cfg_note"] = f"Could not read backbone config fields: {e}"

    # ── Derived efficiency metrics ─────────────────────────────────────────────
    print("\n▶ Computing efficiency metrics …")
    eff_metrics = compute_efficiency_metrics(
        results,
        baseline_params_m   = args.baseline_params_m,
        baseline_latency_ms = args.baseline_latency_ms,
        baseline_gflops     = args.baseline_gflops,
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
                    "system": sys_info,
                    "profiling": {
                        "n_warmup":          args.n_warmup,
                        "n_measure":         args.n_measure,
                        "n_profile_batches": args.n_profile_batches,
                        "n_input_points":    args.n_input_points,
                        "skip_batches":      args.skip_batches,
                        "skip_train_memory": args.skip_train_memory,
                        "strict_load":       not args.no_strict_load,
                        "batch_size":        1,
                    },
                    "baselines": {
                        "params_m":   args.baseline_params_m,
                        "latency_ms": args.baseline_latency_ms,
                        "gflops":     args.baseline_gflops,
                    },
                },
                tags=base_tags,
            )

            # Numeric scalars → single WandB step (visible in charts)
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

            # System info with sys/ prefix so it's grouped in the summary
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
    num_stages = results.get("cfg_num_stages", 5)
    stage_rows = [
        (f"  GFLOPs stage {s}", f"gflops_stage{s}")
        for s in range(int(num_stages) if isinstance(num_stages, (int, float)) else 5)
    ]

    key_metrics = [
        ("Params total (M)",             "params_total_M"),
        ("Params trainable (M)",         "params_trainable_M"),
        ("Model size (MB)",              "model_size_mb"),
        ("─────────── FLOPs ────────────────────────────────────────", None),
        ("GFLOPs theoretical (encoder)", "gflops_total"),
        *stage_rows,
        ("GFLOPs measured (full model)", "gflops_measured"),
        ("─────────── Latency ───────────────────────────────────────", None),
        ("Latency mean (ms)",            "latency_mean_ms"),
        ("Latency std  (ms)",            "latency_std_ms"),
        ("Latency p95  (ms)",            "latency_p95_ms"),
        ("FPS",                          "fps"),
        ("─────────── Memory ────────────────────────────────────────", None),
        ("Mem infer alloc (GB)",         "mem_infer_alloc_gb"),
        ("Mem train alloc (GB)",         "mem_train_alloc_gb"),
        ("─────────── vs Baseline ───────────────────────────────────", None),
        ("Compression ratio (params)",   "compression_ratio"),
        ("Params reduction (%)",         "params_reduction_pct"),
        ("FLOPs reduction ratio",        "flops_reduction_ratio"),
        ("FLOPs reduction (%)",          "flops_reduction_pct"),
        ("Speedup vs baseline",          "speedup_vs_baseline"),
    ]

    print("\n" + "=" * 72)
    print(f"  SUMMARY  —  {args.run_name}")
    print("=" * 72)
    for label, key in key_metrics:
        if key is None:
            print(f"  {label}")
            continue
        val = results.get(key, "N/A")
        print(f"  {label:<43} {_fmt(val)}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()