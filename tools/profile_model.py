#!/usr/bin/env python3
"""
tools/profile_model.py  —  Standalone profiler for the LitePT model family.

Logs to WandB project "LitePT-Profiling" (separate from the training project).
Each run records a full system fingerprint (GPU, CUDA, PyTorch, driver, packages)
so results across different machines can be compared cleanly.

Measured quantities:
  params_total_M / params_trainable_M / params_buffer_M
  gflops_total + per-stage breakdown
  mem_infer_alloc_gb / mem_infer_reserv_gb   (inference peak)
  mem_train_alloc_gb / mem_train_reserv_gb   (one forward+backward, no optim step)
  latency_mean/std/p50/p95 _ms, throughput_sps

FLOPs formula (backbone encoder only)
--------------------------------------
All counts = multiply-accumulate (MAC) × 2.
Token count: N₀ = n_input_points; Nₛ = Nₛ₋₁ // 4 (GridPool stride-2 → ÷4/stage).

Conv block:   FLOPs = N × (27×OCC×C² + C²) × 2   [OCC=0.25]
Attn block:   FLOPs = N × [(4 + 2·ratio)·C² + 2·P·C] × 2
              derived from: QKV(3C²)+Q@K(PC)+attn@V(PC)+OutProj(C²)+MLP(2·ratio·C²)

PointROPE constraint
--------------------
The PointROPE CUDA kernel requires:  head_dim % 6 == 0
(3D RoPE needs pairs for x,y,z axes → must be divisible by 6)
The original LitePT uses head_dim=18 everywhere (18%6=0 ✓).
All ablation configs with reduced channels must use channels = 18 × k
(i.e. channels must be multiples of 18) to satisfy this constraint.

Usage
-----
Before training (random weights — architecture cost):
    python tools/profile_model.py \\
        --config-file configs/scannet/semseg-litept-lw-e.py \\
        --run-name s1/litept-lw-e-random \\
        --wandb-project LitePT-Profiling

After training (load checkpoint — real-world cost):
    python tools/profile_model.py \\
        --config-file configs/scannet/semseg-litept-lw-e.py \\
        --weight exp/scannet/semseg-litept-lw-e/model/model_best.pth \\
        --run-name s1/litept-lw-e-trained \\
        --wandb-project LitePT-Profiling

WandB login tip
---------------
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
    p = argparse.ArgumentParser(description="LitePT standalone model profiler")
    p.add_argument("--config-file", required=True,
                   help="Path to model config (.py)")
    p.add_argument("--weight", default=None,
                   help="Checkpoint path (.pth). If omitted, random weights are used.")
    p.add_argument("--run-name", required=True,
                   help="WandB run name, e.g. 's1/litept-lw-e-random'")
    p.add_argument("--wandb-project", default="LitePT-Profiling",
                   help="WandB project (default: LitePT-Profiling)")
    p.add_argument("--wandb-key", default=None,
                   help="WandB API key. If None, uses current `wandb login` session. "
                        "If not logged in, run `wandb login` once in your terminal.")
    p.add_argument("--n-warmup", type=int, default=5,
                   help="Latency warmup iterations (default: 5)")
    p.add_argument("--n-measure", type=int, default=30,
                   help="Latency measurement iterations (default: 30)")
    p.add_argument("--n-profile-batches", type=int, default=50,
                   help="Val batches for memory/latency measurements (default: 50)")
    p.add_argument("--skip-train-memory", action="store_true",
                   help="Skip training-mode memory measurement (saves ~30 s)")
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

    for pkg in ("spconv", "flash_attn", "timm", "torch_scatter", "pointops"):
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
        "params_total_M":     round(total     / 1e6, 3),
        "params_trainable_M": round(trainable / 1e6, 3),
        "params_buffer_M":    round(buffers   / 1e6, 3),
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
    bcfg = cfg.model.backbone

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
        "Conv occ=0.25, n_input=60000, stride-4 pooling. "
        "Excludes LayerNorm, PointROPE, GridPooling, decoder. "
        "RELATIVE comparison only."
    )
    return result


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
        "mem_infer_alloc_gb":  round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
        "mem_infer_reserv_gb": round(torch.cuda.max_memory_reserved(device)  / 1024**3, 3),
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
        # FIX: use torch.amp.autocast (stable since PyTorch 2.0, avoids FutureWarning
        # from the deprecated torch.cuda.amp.autocast).
        # Fallback to the old API for PyTorch < 2.0.
        try:
            ctx = torch.amp.autocast(device_type="cuda", enabled=amp, dtype=torch.float16)
        except TypeError:
            # PyTorch < 2.0: torch.amp.autocast doesn't accept device_type as positional
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
        "mem_train_alloc_gb":  round(alloc_gb,  3),
        "mem_train_reserv_gb": round(reserv_gb, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Latency measurement
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_latency(
    model: nn.Module,
    batches: List[dict],
    device: torch.device,
    n_warmup: int = 5,
    n_measure: int = 30,
) -> Dict[str, float]:
    """CUDA-event latency. Batches are cycled if fewer than n_warmup+n_measure."""
    model.eval()

    def _get(i: int) -> dict:
        return batches[i % len(batches)]

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

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
    return {
        "latency_mean_ms": round(float(np.mean(t)),           2),
        "latency_std_ms":  round(float(np.std(t)),            2),
        "latency_p50_ms":  round(float(np.percentile(t, 50)), 2),
        "latency_p95_ms":  round(float(np.percentile(t, 95)), 2),
        "throughput_sps":  round(1000.0 / float(np.mean(t)),  2),
        "latency_n_runs":  n_measure,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Model builder
# ══════════════════════════════════════════════════════════════════════════════

def build_and_load(
    cfg: Config,
    weight_path: Optional[str],
    device: torch.device,
) -> nn.Module:
    model = build_model(cfg.model)
    model = model.to(device)
    print(f"  Model built  ({sum(p.numel() for p in model.parameters())/1e6:.2f} M params)")

    if weight_path is not None:
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Checkpoint not found: {weight_path}")
        print(f"  Loading checkpoint: {weight_path}")
        try:
            ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(weight_path, map_location="cpu")

        state = ckpt.get("state_dict", ckpt)
        new_state: Dict[str, torch.Tensor] = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v)
            for k, v in state.items()
        )
        info = model.load_state_dict(new_state, strict=True)
        if info.missing_keys:
            print(f"  [warn] Missing keys ({len(info.missing_keys)}): "
                  f"{info.missing_keys[:3]} ...")
        if info.unexpected_keys:
            print(f"  [warn] Unexpected keys ({len(info.unexpected_keys)}): "
                  f"{info.unexpected_keys[:3]} ...")
        print(f"  Checkpoint loaded (epoch {ckpt.get('epoch','?')})")

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
    """
    Ensure WandB is logged in without triggering an interactive prompt.

    If an API key is provided, log in with it.
    If already logged in (netrc or WANDB_API_KEY env var), do nothing.
    If not logged in and no key given, print a helpful message and return False.

    Returns True if login is ready, False if WandB should be skipped.
    """
    if api_key:
        wandb.login(key=api_key, relogin=True)
        return True

    # Check if already authenticated (env var or valid netrc).
    env_key = os.environ.get("WANDB_API_KEY", "")
    if env_key:
        return True

    # Try a silent API check — if it raises, we're not logged in.
    try:
        api = wandb.Api(timeout=5)
        _ = api.viewer  # raises if not authenticated
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

    print("\n" + "=" * 64)
    print(f"  LitePT Profiler  —  {args.run_name}")
    print(f"  Config  : {args.config_file}")
    print(f"  Weight  : {args.weight or 'None (random / untrained weights)'}")
    print(f"  Device  : {device}")
    print("=" * 64 + "\n")

    # ── System info ───────────────────────────────────────────────────────────
    print("▶ Collecting system info ...")
    sys_info = collect_system_info()
    for k, v in sys_info.items():
        print(f"    {k}: {v}")

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n▶ Building model ...")
    model = build_and_load(cfg, args.weight, device)

    # ── Parameters ────────────────────────────────────────────────────────────
    print("\n▶ Counting parameters ...")
    param_info = count_params(model)
    for k, v in param_info.items():
        print(f"    {k}: {v}")

    # ── Theoretical FLOPs ─────────────────────────────────────────────────────
    print("\n▶ Estimating theoretical GFLOPs ...")
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
            print("       Memory and latency measurements skipped.")
            skip_runtime = True

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
        print(f"\n▶ Measuring latency (warmup={args.n_warmup}, measure={args.n_measure}) ...")
        model.eval()
        latency = measure_latency(model, batches, device,
                                  n_warmup=args.n_warmup, n_measure=args.n_measure)
        for k, v in latency.items():
            print(f"    {k}: {v}")

    # ── Assemble results ──────────────────────────────────────────────────────
    results: Dict[str, object] = {}
    results.update(param_info)
    results.update(flop_info)
    results.update(mem_infer)
    results.update(mem_train)
    results.update(latency)
    results["config_file"]   = args.config_file
    results["weight_path"]   = args.weight or "random"
    results["weight_loaded"] = args.weight is not None

    bcfg = cfg.model.backbone
    results["cfg_enc_channels"] = str(list(bcfg.enc_channels))
    results["cfg_enc_depths"]   = str(list(bcfg.enc_depths))
    results["cfg_enc_patch_sz"] = str(list(bcfg.enc_patch_size))
    results["cfg_enc_attn"]     = str(list(bcfg.enc_attn))

    # ── WandB logging ─────────────────────────────────────────────────────────
    print("\n▶ Logging to WandB ...")
    wandb_ready = ensure_wandb_login(args.wandb_key)

    if wandb_ready:
        try:
            try:
                backbone_cfg_dict = cfg_to_dict(bcfg)
            except Exception:
                backbone_cfg_dict = {"serialise_error": "could not convert backbone cfg"}

            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config={
                    "backbone_cfg": backbone_cfg_dict,
                    "system":       sys_info,
                    "profiling": {
                        "n_warmup":          args.n_warmup,
                        "n_measure":         args.n_measure,
                        "n_profile_batches": args.n_profile_batches,
                        "flop_n_input_pts":  60_000,
                        "skip_train_memory": args.skip_train_memory,
                    },
                },
                tags=["profile", "strategy1"],
            )

            # Numeric metrics → single WandB step
            numeric = {k: v for k, v in results.items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            wandb.log(numeric)

            # String / bool → summary (visible in WandB runs table)
            for k, v in results.items():
                if isinstance(v, (str, bool)):
                    wandb.run.summary[k] = v

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

    print("\n" + "=" * 64)
    print("  Profiling complete.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
