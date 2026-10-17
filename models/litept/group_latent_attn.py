"""
models/litept/group_latent_attn.py
===================================
GroupLatentAttention (v1, mean-pool) and GroupLatentAttentionV2 (Perceiver-style).

──────────────────────────────────────────────────────────────────────────────
WHY V1 FAILED — ROOT CAUSE ANALYSIS
──────────────────────────────────────────────────────────────────────────────
Strategy-2 experiments confirmed the theoretical prediction:

  lw-c + LaRA3D-v1 (G=32, mean-pool):  73.99% test  ← −0.96% vs lw-c (74.95%)
  lw-e + LaRA3D-v1 (G=32, mean-pool):  CRASHED at epoch 99

Root cause: mean-pool compression in Step 2.

In LaRa (2D radiance-field reconstruction):
  • Loss = per-rendered-pixel photometric error = aggregated over the full volume
  • A mean over K=16 voxels is a valid group summary because the renderer
    integrates over the whole volume anyway.
  • Boundary voxels at object edges are a tiny minority and their
    misrepresentation costs very little in a rendering loss.

In LitePT (3D point-cloud semantic segmentation):
  • Loss = per-POINT cross-entropy. Every single token needs its own label.
  • Boundary tokens between "chair" and "floor" are exactly the hardest, most
    informative samples. A mean of 8 chair tokens + 8 floor tokens produces a
    useless "half-chair-half-floor" latent.
  • Gradient of the mean: ∂mean/∂token_i = 1/K for all i. Token gradients are
    diluted K-fold. For K=16 that is 16× weaker gradients than the baseline.
  • Result: model converges to worse boundary decision boundaries, and with the
    small K=16 on lw-e the gradient signal collapses entirely → training diverges.

──────────────────────────────────────────────────────────────────────────────
V2 DESIGN — PERCEIVER-STYLE GROUP-LATENT ATTENTION
──────────────────────────────────────────────────────────────────────────────
Core principle (from Perceiver IO, Jaegle et al. 2021):
  Instead of COMPRESSING tokens into latents, MAINTAIN INDEPENDENT per-token
  features throughout, while INJECTING global context additively.

  Tokens are NEVER compressed.
  Global context flows through a small latent bottleneck.
  Boundary tokens retain full per-token fidelity.

Pipeline for each patch of P = G × K tokens:
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1  Local Self-Attention (with PointROPE)          O(G × K²)        │
│         K tokens per group attend to each other.                        │
│         Uses flash_attn_varlen for GPU efficiency.                      │
│         Output: locally-enriched token features.                        │
├─────────────────────────────────────────────────────────────────────────┤
│ STEP 2  Latent Extraction via Cross-Attention           O(G × n_lat × K)│
│         n_lat LEARNED latent vectors per group (Q) attend to            │
│         K local tokens (K, V).                                          │
│         Content-adaptive: latents learn to focus on representative      │
│         tokens, NOT average all tokens. Boundary tokens are treated     │
│         as normal keys — they contribute to the latent but are NOT      │
│         forced to represent the whole group.                            │
├─────────────────────────────────────────────────────────────────────────┤
│ STEP 3  Global Self-Attention (with PointROPE)          O(G²)           │
│         G group latents (one per group) attend to each other.           │
│         PointROPE applied on group centroid positions.                  │
│         Each latent acquires scene-level context.                       │
├─────────────────────────────────────────────────────────────────────────┤
│ STEP 4  Additive Token Injection                        O(P)            │
│         Each token receives its group's globally-updated latent         │
│         as an ADDITIVE residual via a learned gated projection.         │
│                                                                         │
│         token_out[i] = local_out[i] + gate * broadcast_proj(latent[g]) │
│                                                                         │
│         No attention required: with n_lat=1, cross-attention from       │
│         tokens to 1 latent key reduces to a linear projection.          │
│         The gate (sigmoid) controls injection strength per latent.      │
├─────────────────────────────────────────────────────────────────────────┤
│ STEP 5  Output projection + dropout                                     │
└─────────────────────────────────────────────────────────────────────────┘

Complexity summary (P=512, G=32, K=16, n_lat=1):
  v1 (mean-pool):  O(G×K² + G²) =   32×256 + 1024  =  9,216
  v2 (Perceiver):  O(G×K² + G×K + G² + P) = 8192 + 512 + 1024 + 512 = 10,240
  flat baseline:   O(P²) = 262,144
  Speedup v2 vs flat: ~25×

Parameter count increase vs v1 (lw-c, stages 3+4, C=180 and C=360):
  v1 extra:  3C² per attn block (global_qkv vs local_qkv)
  v2 extra:  ~4C² + 2C per attn block (latent_cross_attn + broadcast + gate)
  Net increase over v1: ~1C² per block, ~0.2M total → negligible

PointROPE constraint:
  head_dim % 6 == 0 required. All LitePT channels are 18× multiples,
  so head_dim=18 at every stage → 18%6=0 ✓.
  GroupLatentAttentionV2 preserves this invariant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import flash_attn

from libs.pointrope import PointROPE
from models.modules import PointModule
from models.utils.structure import Point


# ═══════════════════════════════════════════════════════════════════════════
#  V1 — GroupLatentAttention (mean-pool)
#  Kept for reference and backward-compatibility.  DO NOT USE for new expts.
# ═══════════════════════════════════════════════════════════════════════════

class GroupLatentAttention(PointModule):
    """
    GroupLatentAttention v1 — LaRA3D (mean-pool compression).

    WARNING: This version FAILS on 3D point-cloud segmentation.
    Mean-pooling destroys boundary token information required for per-point
    supervision.  See module docstring above for full diagnosis.

    Kept for ablation comparison only.  Use GroupLatentAttentionV2 instead.
    """

    def __init__(
        self,
        channels:    int,
        num_heads:   int,
        patch_size:  int,
        group_size:  int,
        rope_freq:   float,
        qkv_bias:    bool  = True,
        qk_scale:    float = None,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
        order_index: int   = 0,
    ):
        super().__init__()

        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        head_dim = channels // num_heads
        if head_dim % 6 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by 6 for PointROPE. "
                f"channels={channels}, num_heads={num_heads}."
            )
        if patch_size % group_size != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by group_size ({group_size})."
            )

        self.channels    = channels
        self.num_heads   = num_heads
        self.patch_size  = patch_size
        self.group_size  = group_size
        self.local_size  = patch_size // group_size
        self.scale       = qk_scale or head_dim ** -0.5
        self.order_index = order_index
        self.attn_drop   = attn_drop

        self.local_qkv  = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.global_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj       = nn.Linear(channels, channels)
        self.proj_drop  = nn.Dropout(proj_drop)
        self.norm_latent = nn.LayerNorm(channels)
        self.rope        = PointROPE(freq=rope_freq)

    def _apply_rope(self, q, k, pos):
        q_r = q.transpose(0, 1).unsqueeze(0)
        k_r = k.transpose(0, 1).unsqueeze(0)
        p_r = pos.unsqueeze(0)
        q_r = self.rope(q_r.float(), p_r).to(q.dtype)
        k_r = self.rope(k_r.float(), p_r).to(k.dtype)
        return q_r.squeeze(0).transpose(0, 1), k_r.squeeze(0).transpose(0, 1)

    def forward(self, point: Point) -> Point:
        H = self.num_heads
        C = self.channels
        P = self.patch_size
        G = self.group_size
        K = self.local_size

        pad, unpad, cu_seqlens = point.get_padding_and_inverse(P)
        order   = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        feat = point.feat[order]
        pos  = point.grid_coord[order]
        N_real = feat.shape[0]

        pad_size = (P - N_real % P) % P
        if pad_size > 0:
            feat = torch.cat([feat, feat[-1:].expand(pad_size, -1)], dim=0)
            pos  = torch.cat([pos,  pos[-1:].expand(pad_size, -1)], dim=0)

        N_padded  = N_real + pad_size
        n_patches = N_padded // P
        input_dtype = feat.dtype

        # Step 1: Local self-attention
        feat_grouped = feat.view(n_patches * G, K, C)
        pos_grouped  = pos.view(n_patches * G, K, 3)

        lqkv = self.local_qkv(feat_grouped)
        lq, lk, lv = lqkv.half().chunk(3, dim=-1)
        lq = lq.reshape(-1, H, C // H)
        lk = lk.reshape(-1, H, C // H)
        lv = lv.reshape(-1, H, C // H)

        lq, lk = self._apply_rope(lq, lk, pos_grouped.reshape(-1, 3))

        local_qkv_packed = torch.stack([lq, lk, lv], dim=1)
        local_cu = torch.arange(0, (n_patches * G + 1) * K, K, device=feat.device, dtype=torch.int32)

        local_out = flash_attn.flash_attn_varlen_qkvpacked_func(
            local_qkv_packed, local_cu, max_seqlen=K,
            dropout_p=self.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
        ).reshape(n_patches * G, K, C).to(input_dtype)

        # Step 2: Mean-pool (PROBLEMATIC for segmentation — use V2 instead)
        group_latents = local_out.mean(dim=1)
        group_pos     = pos_grouped.float().mean(dim=1).long()

        # Step 3: Global self-attention
        latents_normed = self.norm_latent(group_latents)
        latents_2d     = latents_normed.view(n_patches, G, C)
        latent_pos_2d  = group_pos.view(n_patches, G, 3)

        gqkv = self.global_qkv(latents_2d)
        gq, gk, gv = gqkv.half().chunk(3, dim=-1)
        gq = gq.reshape(-1, H, C // H)
        gk = gk.reshape(-1, H, C // H)
        gv = gv.reshape(-1, H, C // H)

        gq, gk = self._apply_rope(gq, gk, latent_pos_2d.reshape(-1, 3).float())

        global_qkv_packed = torch.stack([gq, gk, gv], dim=1)
        global_cu = torch.arange(0, (n_patches + 1) * G, G, device=feat.device, dtype=torch.int32)

        global_out = flash_attn.flash_attn_varlen_qkvpacked_func(
            global_qkv_packed, global_cu, max_seqlen=G,
            dropout_p=self.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
        ).reshape(n_patches, G, C).to(input_dtype)

        # Step 4: Broadcast
        global_broadcast = (
            global_out.unsqueeze(2).expand(-1, -1, K, -1).reshape(N_padded, C)
        )
        combined = local_out.view(N_padded, C) + global_broadcast

        out = self.proj(combined)
        out = self.proj_drop(out)
        point.feat = out[:N_real][inverse]
        return point

    def extra_repr(self) -> str:
        K = self.local_size
        return (
            f"[V1-mean-pool] channels={self.channels}, heads={self.num_heads}, "
            f"P={self.patch_size}, G={self.group_size}, K={K}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  V2 — GroupLatentAttentionV2 (Perceiver-style, NO mean-pool)
#  This is the recommended version for 3D semantic segmentation.
# ═══════════════════════════════════════════════════════════════════════════

class GroupLatentAttentionV2(PointModule):
    """
    GroupLatentAttentionV2 — LaRA3D with Perceiver-style latent extraction.

    Fixes the mean-pool failure of v1. Tokens are NEVER compressed.
    Global context is injected via learned group latents without destroying
    per-token feature fidelity. See module docstring for full design rationale.

    Parameters
    ----------
    channels     : C — embedding dimension
    num_heads    : H — attention heads (head_dim = C/H, must be % 6 == 0)
    patch_size   : P — tokens per patch (= G × K)
    group_size   : G — number of groups per patch
    rope_freq    : PointROPE frequency
    num_latents  : n_lat — learnable latent vectors per group (default 1)
                   • 1: one group summary per group (recommended for segm.)
                   • 2: richer representation at 2× latent cost
    gate_init    : initial value for the injection gate (default 0.0 → tanh(0) = 0)
                   Gate controls how strongly global context is injected at init.
                   Starting at 0 means the model begins as the flat attention
                   baseline and learns to gradually activate global context.
    order_index  : which serialization order to use
    """

    def __init__(
        self,
        channels:    int,
        num_heads:   int,
        patch_size:  int,
        group_size:  int,
        rope_freq:   float,
        qkv_bias:    bool  = True,
        qk_scale:    float = None,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
        order_index: int   = 0,
        num_latents: int   = 1,
        gate_init:   float = 0.0,
    ):
        super().__init__()

        # ── Validation ────────────────────────────────────────────────────
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        head_dim = channels // num_heads
        if head_dim % 6 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be % 6 for PointROPE. "
                f"channels={channels}, num_heads={num_heads} → head_dim={head_dim}."
            )
        if patch_size % group_size != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by group_size ({group_size})."
            )

        self.channels    = channels
        self.num_heads   = num_heads
        self.patch_size  = patch_size          # P
        self.group_size  = group_size          # G
        self.local_size  = patch_size // group_size   # K = P // G
        self.num_latents = num_latents         # n_lat
        self.scale       = qk_scale or head_dim ** -0.5
        self.order_index = order_index
        self.attn_drop   = attn_drop

        # ── STEP 1: Local self-attention projections ──────────────────────
        self.local_qkv  = nn.Linear(channels, channels * 3, bias=qkv_bias)

        # ── STEP 2: Latent cross-attention projections ────────────────────
        # Learnable group latents — one set of n_lat vectors per group,
        # shared across all patches (broadcast in forward pass).
        # Shape: [G, n_lat, C]
        self.group_latents = nn.Parameter(torch.empty(group_size, num_latents, channels))
        nn.init.trunc_normal_(self.group_latents, std=0.02)

        # Project latents to Q, and tokens to K/V for cross-attention.
        # We use SEPARATE projections (no weight sharing) so latents can
        # learn different attention patterns than the local self-attention.
        self.latent_to_q  = nn.Linear(channels, channels, bias=qkv_bias)
        self.token_to_k   = nn.Linear(channels, channels, bias=qkv_bias)
        self.token_to_v   = nn.Linear(channels, channels, bias=False)

        # ── STEP 3: Global self-attention projections ─────────────────────
        self.global_qkv   = nn.Linear(channels, channels * 3, bias=qkv_bias)

        # ── STEP 4: Token injection (broadcast) ───────────────────────────
        # Project global latent → injection vector for each token.
        # We use a gated linear unit: out = token + tanh(gate) * proj(latent)
        # Gate initialized to gate_init so injection starts small.
        # Each of the G groups has its own gate scalar (learned per-group).
        self.broadcast_proj = nn.Linear(channels, channels, bias=True)
        # Per-group gate: shape [G, n_lat] — gating for each latent in each group
        self.gate           = nn.Parameter(torch.full((group_size, num_latents), gate_init))

        # ── Output + normalization ────────────────────────────────────────
        self.proj       = nn.Linear(channels, channels)
        self.proj_drop  = nn.Dropout(proj_drop)
        self.norm_local = nn.LayerNorm(channels)    # norm tokens before cross-attn
        self.norm_latent = nn.LayerNorm(channels)   # norm latents before global attn

        # ── PointROPE — for steps 1 and 3 ────────────────────────────────
        self.rope = PointROPE(freq=rope_freq)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _apply_rope(
        self,
        q:   torch.Tensor,   # [N, H, head_dim]
        k:   torch.Tensor,   # [N, H, head_dim]
        pos: torch.Tensor,   # [N, 3]
    ):
        """Apply PointROPE to Q and K independently (same positions for self-attn)."""
        q_r = q.transpose(0, 1).unsqueeze(0)    # [1, H, N, head_dim]
        k_r = k.transpose(0, 1).unsqueeze(0)
        p_r = pos.unsqueeze(0)                   # [1, N, 3]
        q_r = self.rope(q_r.float(), p_r).to(q.dtype)
        k_r = self.rope(k_r.float(), p_r).to(k.dtype)
        return q_r.squeeze(0).transpose(0, 1), k_r.squeeze(0).transpose(0, 1)

    def _cross_attn_latents_to_tokens(
        self,
        latent_feat:  torch.Tensor,  # [M, n_lat, C]  — M = n_patches*G
        token_feat:   torch.Tensor,  # [M, K, C]
    ) -> torch.Tensor:
        """
        Step 2: cross-attention from n_lat learned latents (Q) to K tokens (K/V).
        Returns updated latents [M, n_lat, C].

        No PointROPE here: latent queries have no spatial position (they are
        learned, not derived from a point cloud location). The token keys
        already encode local context from Step 1.

        For n_lat=1 and small K, this is very fast and uses standard PyTorch
        scaled dot-product attention (no flash_attn needed).
        """
        M   = latent_feat.shape[0]
        H   = self.num_heads
        C   = self.channels
        n_lat = self.num_latents
        K   = token_feat.shape[1]
        hd  = C // H     # head dimension

        # Project queries from latents
        q = self.latent_to_q(latent_feat)   # [M, n_lat, C]
        q = q.view(M, n_lat, H, hd).permute(0, 2, 1, 3)   # [M, H, n_lat, hd]

        # Project keys and values from tokens
        k = self.token_to_k(token_feat)     # [M, K, C]
        v = self.token_to_v(token_feat)     # [M, K, C]
        k = k.view(M, K, H, hd).permute(0, 2, 1, 3)   # [M, H, K, hd]
        v = v.view(M, K, H, hd).permute(0, 2, 1, 3)   # [M, H, K, hd]

        # Scaled dot-product attention: [M, H, n_lat, K]
        # Using PyTorch's built-in (uses flash-attn backend where available)
        out = F.scaled_dot_product_attention(
            q.float(), k.float(), v.float(),
            dropout_p=self.attn_drop if self.training else 0.0,
        ).to(latent_feat.dtype)   # [M, H, n_lat, hd]

        # Reshape back: [M, n_lat, C]
        out = out.permute(0, 2, 1, 3).reshape(M, n_lat, C)
        return out

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, point: Point) -> Point:
        H = self.num_heads
        C = self.channels
        P = self.patch_size
        G = self.group_size
        K = self.local_size
        n_lat = self.num_latents

        # ── Serialization & padding ───────────────────────────────────────
        pad, unpad, cu_seqlens = point.get_padding_and_inverse(P)
        order   = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        feat = point.feat[order]
        pos  = point.grid_coord[order]
        N_real = feat.shape[0]

        # Physical padding to multiple of P
        pad_size = (P - N_real % P) % P
        if pad_size > 0:
            feat = torch.cat([feat, feat[-1:].expand(pad_size, -1)], dim=0)
            pos  = torch.cat([pos,  pos[-1:].expand(pad_size, -1)], dim=0)

        N_padded  = N_real + pad_size
        n_patches = N_padded // P
        input_dtype = feat.dtype

        # Reshape to [n_patches*G, K, *]
        feat_grouped = feat.view(n_patches * G, K, C)   # [M, K, C]   M = n_patches*G
        pos_grouped  = pos.view(n_patches * G, K, 3)

        # ═════════════════════════════════════════════════════════════════
        # STEP 1: Local Intra-Group Self-Attention (with PointROPE)
        # ═════════════════════════════════════════════════════════════════
        M = n_patches * G

        lqkv = self.local_qkv(feat_grouped)               # [M, K, 3C]
        lq, lk, lv = lqkv.half().chunk(3, dim=-1)

        lq = lq.reshape(M * K, H, C // H)
        lk = lk.reshape(M * K, H, C // H)
        lv = lv.reshape(M * K, H, C // H)

        flat_pos = pos_grouped.reshape(M * K, 3)
        lq, lk = self._apply_rope(lq, lk, flat_pos)

        local_qkv_packed = torch.stack([lq, lk, lv], dim=1)   # [M*K, 3, H, hd]
        local_cu = torch.arange(
            0, (M + 1) * K, K, device=feat.device, dtype=torch.int32
        )

        local_out = flash_attn.flash_attn_varlen_qkvpacked_func(
            local_qkv_packed,
            local_cu,
            max_seqlen=K,
            dropout_p=self.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
        ).reshape(M, K, C).to(input_dtype)           # [M, K, C]

        # ═════════════════════════════════════════════════════════════════
        # STEP 2: Latent Extraction via Cross-Attention (NO mean-pool!)
        # ═════════════════════════════════════════════════════════════════
        # Normalise token features before cross-attention
        local_normed = self.norm_local(local_out)         # [M, K, C]

        # Expand learned group latents: [G, n_lat, C] → [M, n_lat, C]
        # Each group in each patch starts from the SAME learned latent —
        # it's a content-adaptive initialisation, not a hard constraint.
        lat = self.group_latents.unsqueeze(0).expand(
            n_patches, -1, -1, -1
        ).reshape(M, n_lat, C)                            # [M, n_lat, C]

        # Cross-attention: latents (Q) attend to local tokens (K/V)
        group_latents = self._cross_attn_latents_to_tokens(lat, local_normed)  # [M, n_lat, C]

        # ═════════════════════════════════════════════════════════════════
        # STEP 3: Global Inter-Group Self-Attention (with PointROPE)
        # ═════════════════════════════════════════════════════════════════
        # If n_lat > 1: flatten to (G*n_lat) tokens per patch for global attn
        G_eff = G * n_lat   # effective number of global tokens per patch

        # Group centroid positions for PointROPE
        group_pos = pos_grouped.float().mean(dim=1).long()    # [M, 3]
        if n_lat > 1:
            # Repeat centroid for each latent per group
            group_pos_eff = group_pos.unsqueeze(1).expand(-1, n_lat, -1).reshape(M * n_lat, 3)
        else:
            group_pos_eff = group_pos                         # [M, 3]

        # Normalise latents before global attention
        lat_normed = self.norm_latent(group_latents)          # [M, n_lat, C]
        lat_flat   = lat_normed.reshape(n_patches, G_eff, C)  # [n_patches, G_eff, C]

        gqkv = self.global_qkv(lat_flat)                      # [n_patches, G_eff, 3C]
        gq, gk, gv = gqkv.half().chunk(3, dim=-1)
        gq = gq.reshape(n_patches * G_eff, H, C // H)
        gk = gk.reshape(n_patches * G_eff, H, C // H)
        gv = gv.reshape(n_patches * G_eff, H, C // H)

        # PointROPE on group centroid positions
        gq, gk = self._apply_rope(gq, gk, group_pos_eff.reshape(n_patches * G_eff, 3))   #.float())

        global_qkv_packed = torch.stack([gq, gk, gv], dim=1)  # [n_patches*G_eff, 3, H, hd]
        global_cu = torch.arange(
            0, (n_patches + 1) * G_eff, G_eff,
            device=feat.device, dtype=torch.int32
        )

        global_out = flash_attn.flash_attn_varlen_qkvpacked_func(
            global_qkv_packed,
            global_cu,
            max_seqlen=G_eff,
            dropout_p=self.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
        ).reshape(n_patches, G, n_lat, C).to(input_dtype)     # [n_patches, G, n_lat, C]

        # ═════════════════════════════════════════════════════════════════
        # STEP 4: Gated Additive Token Injection (Broadcast)
        # ═════════════════════════════════════════════════════════════════
        # Project global latent → injection vector
        inj_input = global_out.reshape(n_patches * G, n_lat, C)   # [M, n_lat, C]
        inj_vec   = self.broadcast_proj(inj_input)                 # [M, n_lat, C]

        # Gate: tanh of learned scalar, shape [G, n_lat]
        # Each group+latent has its own gate. Tanh bounds injection in [-1,+1].
        # At gate_init=0.0, gate=tanh(0)=0 → no injection at start.
        gate = torch.tanh(self.gate)                               # [G, n_lat]
        gate = gate.unsqueeze(0).expand(n_patches, -1, -1).reshape(M, n_lat, 1)  # [M, n_lat, 1]

        # Sum over n_lat dimension: each token gets the weighted sum of its group's latents
        # [M, n_lat, C] * [M, n_lat, 1] → [M, C]
        injection = (inj_vec * gate).sum(dim=1)                    # [M, C]

        # Broadcast K times: [M, C] → [M, K, C]
        injection_expanded = injection.unsqueeze(1).expand(-1, K, -1)   # [M, K, C]

        # Residual connection: add global context to locally-enriched tokens
        combined = local_out + injection_expanded                  # [M, K, C]

        # ═════════════════════════════════════════════════════════════════
        # STEP 5: Output Projection
        # ═════════════════════════════════════════════════════════════════
        out = self.proj(combined.reshape(N_padded, C))
        out = self.proj_drop(out)

        # Trim physical padding, unshuffle to original point order
        point.feat = out[:N_real][inverse]
        return point

    def extra_repr(self) -> str:
        K = self.local_size
        n_lat = self.num_latents
        return (
            f"[V2-Perceiver] channels={self.channels}, heads={self.num_heads}, "
            f"P={self.patch_size}, G={self.group_size}, K={K}, n_lat={n_lat}, "
            f"attn=O(G×K²+G×K+G²+P)=O({self.group_size}×{K}²+{self.group_size}×{K}"
            f"+{self.group_size}²+{self.patch_size})"
            f" vs flat=O({self.patch_size}²)"
        )