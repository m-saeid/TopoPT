# Strategy 2 — LaRA3D Integration Guide

## What was changed / added

```
NEW files to copy into LitePT repo:
  models/litept/group_latent_attn.py   ← LaRA3D module (new class)
  configs/scannet/semseg-litept-lw-e-lara3d-g32.py  ← KEY experiment
  configs/scannet/semseg-litept-lw-e-lara3d-g64.py  ← G ablation
  configs/scannet/semseg-litept-lw-c-lara3d-g32.py  ← generality test
  scripts/run_strategy2.sh             ← training orchestration

MODIFIED file (apply diff to existing):
  models/litept/litept.py              ← use litept_v2.py as the new version
```

## Step 1 — Copy new files

```bash
# From the LitePT-Lightening repo root:

# 1. New module
cp strategy2/models/litept/group_latent_attn.py  models/litept/

# 2. Replace litept.py with updated version
cp strategy2/models/litept/litept_v2.py  models/litept/litept.py

# 3. Configs
cp strategy2/configs/scannet/semseg-litept-lw-e-lara3d-g32.py  configs/scannet/
cp strategy2/configs/scannet/semseg-litept-lw-e-lara3d-g64.py  configs/scannet/
cp strategy2/configs/scannet/semseg-litept-lw-c-lara3d-g32.py  configs/scannet/

# 4. Run script
cp strategy2/scripts/run_strategy2.sh  scripts/
chmod +x scripts/run_strategy2.sh
```

## Step 2 — Verify the integration

```bash
sh scripts/run_strategy2.sh check
```

Expected output:
```
  OK  models/litept/group_latent_attn.py
  OK  configs/scannet/semseg-litept-lw-e-lara3d-g32.py
  ...
  OK  models/litept/litept.py  (GroupLatentAttention found)
```

## Step 3 — Quick smoke test (optional, before full training)

```bash
# Test that the model builds and does a forward pass:
python - <<'EOF'
import sys; sys.path.insert(0, '.')
from utils.config import Config
from models import build_model

cfg = Config.fromfile('configs/scannet/semseg-litept-lw-e-lara3d-g32.py')
model = build_model(cfg.model)
n = sum(p.numel() for p in model.parameters())
print(f"Model built: {n/1e6:.2f}M params")

# Expected: ~7.01M params
EOF
```

## Step 4 — Profile before training (run on each machine)

```bash
sh scripts/run_strategy2.sh profile
```

## Step 5 — Launch training in parallel

```bash
# On Machine 1 (RTX 5090):
sh scripts/run_strategy2.sh machine1

# On Machine 2 (RTX 3090 #1):
sh scripts/run_strategy2.sh machine2

# On Machine 3 (RTX 3090 #2):
sh scripts/run_strategy2.sh machine3
```

## Step 6 — Monitor and print table

```bash
sh scripts/run_strategy2.sh table
```

---

## What the modified litept.py adds (summary of changes)

1. **Import**: `from models.litept.group_latent_attn import GroupLatentAttention`

2. **Block class** gains two new parameters:
   - `attn_type: str = "flat"` — `"flat"` (original) or `"group"` (LaRA3D)
   - `group_size: int = 32` — G, number of groups per patch

3. **LitePT class** gains three new parameters:
   - `enc_attn_type: tuple = None` — per-stage attn type, defaults to all `"flat"`
   - `dec_attn_type: tuple = None` — same for decoder
   - `group_size: int = 32` — G used wherever attn_type=`"group"`

4. All new parameters have defaults matching the original behaviour → **fully backward-compatible**.
   Old configs without `enc_attn_type` continue to work unchanged.

---

## Architecture at a glance (lw-e + LaRA3D, G=32)

```
Stage 0: 36ch / 2h  = head_dim 18  SubMConv3d × 2   flat   (attn disabled)
Stage 1: 54ch / 3h  = head_dim 18  SubMConv3d × 2   flat   (attn disabled)
Stage 2: 108ch/ 6h  = head_dim 18  SubMConv3d × 2   flat   (attn disabled)
Stage 3: 180ch/10h  = head_dim 18  PointROPEAttn × 4  GROUP  G=32, K=16
Stage 4: 360ch/20h  = head_dim 18  PointROPEAttn × 2  GROUP  G=32, K=16

head_dim = 18, 18 % 6 == 0  →  PointROPE CUDA kernel satisfied ✓
patch_size // group_size = 512 // 32 = 16  (integer K) ✓
```

## GroupLatentAttention pipeline (per attention block)

```
Input: P=512 serialized tokens (Hilbert/Z-curve order, spatially compact)
  ↓
Step 1 LOCAL:   Split into G=32 groups of K=16 consecutive tokens
                Each group: PointROPE self-attention, O(K²)=O(256)
                Total: G×K² = 32×256 = 8192  ops  (vs P²=262144 flat)
  ↓
Step 2 COMPRESS: Mean-pool each group → 32 group latent vectors
                 Group centroid = mean of member grid_coord (3D position)
  ↓
Step 3 GLOBAL:  32 group latents do mutual PointROPE self-attention
                O(G²)=O(1024) — the LaRa-inspired global reasoning step
                All local groups "see" the full patch context at low cost
  ↓
Step 4 BROADCAST: Add global-updated latent to each member token (residual)
  ↓
Step 5 OUTPUT:  Linear projection + dropout → update point.feat
```

Total attention ops per block: G·K² + G² = 8192 + 1024 = **9216** vs **262144** (flat)
Attention speed-up: **~28×**

---

## Expected paper table

| Row | Config | test mIoU | Δ S0 | Params | GFLOPs |
|-----|--------|-----------|------|--------|--------|
| S0★ | rerun (baseline) | 75.27% | — | 12.71M | 25.42 |
| S4  | lw-c (D+C) | 74.95% | -0.32% | 5.84M | 12.95 |
| S5★ | lw-e (D+C+P) | 74.10% | -1.17% | 5.84M | 11.22 |
| **S6★** | **lw-e+LaRA3D(G=32)** | **>74.95%?** | **>-0.32%?** | **~7.0M** | **~9.65** |
| S7  | lw-e+LaRA3D(G=64) | ~74.5% | | ~7.0M | ~9.5 |
| S8  | lw-c+LaRA3D(G=32) | >74.95%? | | ~7.0M | ~10.5 |

**Success criterion**: S6 mIoU > S4 (74.95%) at fewer GFLOPs (9.65 < 12.95)
→ LaRA3D with small patches (P=512) **outperforms** flat attention with full
  patches (P=1024), while being **more computationally efficient**.

This would be a strong NeurIPS 2026 result.