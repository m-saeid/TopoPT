"""
models/litept/group_latent_attn.py
===================================
GroupLatentAttention — LaRA3D for 3D Point Clouds.

Conceptual inspiration
-----------------------
LaRa (Chen et al., ECCV 2024) proposes Group Attention Layers for 3D
radiance-field reconstruction from 2D images: divide dense feature volumes
into local groups, apply attention within and across groups, achieving
efficient local+global reasoning without O(P²) global attention.

We transfer this principle to serialized 3D point cloud processing:

  2D LaRa → Group Attention Layer
  ─────────────────────────────────────────────────────────────
  Dense 3D voxel grid ──→ Local groups via spatial partition
  Global vol. attention ──→ Group-latent self-attention (O(G²))
  Local feature matching ──→ Intra-group self-attention (O(K²))

  3D LaRA3D (ours) → GroupLatentAttention
  ─────────────────────────────────────────────────────────────
  Serialized patch of P ──→ G groups of K = P/G tokens
  Hilbert/Z-order serialization guarantees each group of K
  consecutive tokens covers a compact spatial neighbourhood.
  Intra-group attention → local fine-grained features (O(K²))
  Group latent (mean) ──→ compact group representation
  Inter-group attention → global patch context (O(G²))
  Broadcast back ────── → each token enriched with global info

Why this is needed
-------------------
Strategy-1 ablation shows that halving patch size P: 1024→512 costs
-1.10% mIoU (lw-d, patch-only comparison), while reducing depth+channel
costs only -0.32% mIoU (lw-c). Patch size is the dominant bottleneck.

Reducing P limits the attention receptive field within a patch. LaRA3D
restores long-range reasoning at O(G·K² + G²) ≪ O(P²) cost, enabling:
  • Small patches (P=512) for memory/speed efficiency
  • Global context within each patch via group-latent communication
  • Expected: recover ~0.7–1.1% mIoU lost from patch-size reduction

Complexity
-----------
  Flat PointROPEAttention  : O(P²)        e.g. 512² = 262 144
  GroupLatentAttention     : O(G·K² + G²) e.g. 32·256 + 1024 = 9 216
  Speed-up (attention ops) : ~28× for P=512, G=32, K=16

Parameter overhead
-------------------
  PointROPEAttention  : 4C² (QKV + proj)
  GroupLatentAttention: 7C² + 2C (local_qkv + global_qkv + proj + LN)
  Extra per attn block: 3C²
  Total extra (lw-e, stages 3+4): +1.17 M params → 7.01 M total

PointROPE constraint
---------------------
PointROPE CUDA kernel requires head_dim % 6 == 0 (3D RoPE needs pairs
per x,y,z axis). All LitePT channels are multiples of 18 → head_dim=18
→ 18 % 6 == 0 ✓. GroupLatentAttention preserves this invariant by
using the same head dimension in both local and global attention phases.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import flash_attn

from libs.pointrope import PointROPE
from models.modules import PointModule
from models.utils.structure import Point


class GroupLatentAttention(PointModule):
    """
    GroupLatentAttention — LaRA3D block for serialized 3D point-cloud patches.

    For each patch of P serialized tokens:

      Step 1  Local attention
              Split P tokens into G contiguous groups of K = P // G.
              Each group runs PointROPE self-attention internally.
              O(G × K²) attention ops (K << P → orders of magnitude cheaper).

      Step 2  Compress
              Mean-pool each group of K tokens → one group latent vector.
              Group centroid position = mean of member grid_coords.

      Step 3  Global attention  (the LaRa-inspired step)
              G group latents do mutual self-attention with PointROPE
              on centroid positions. O(G²) — cheap because G << P.
              This lets every local group "see" the entire patch context.

      Step 4  Broadcast
              Add the globally-updated group latent to every member token
              (residual broadcast). Tokens receive global patch context
              without any per-token global attention.

      Step 5  Output projection + dropout.

    Parameters
    ----------
    channels    : C — embedding dimension
    num_heads   : H — attention heads (head_dim = C/H must be % 6 == 0)
    patch_size  : P — tokens per patch (= G × K)
    group_size  : G — groups per patch (G << P)
    rope_freq   : frequency for PointROPE position encoding
    order_index : which serialization order to use (cycles over 4 orders)
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

        # ── Sanity checks ─────────────────────────────────────────────────
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads})"
            )
        head_dim = channels // num_heads
        if head_dim % 6 != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by 6 for PointROPE. "
                f"Got channels={channels}, num_heads={num_heads}."
            )
        if patch_size % group_size != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by group_size ({group_size}). "
                f"Got K = {patch_size} / {group_size} = {patch_size / group_size:.2f} (not integer)."
            )

        self.channels    = channels
        self.num_heads   = num_heads
        self.patch_size  = patch_size          # P
        self.group_size  = group_size          # G
        self.local_size  = patch_size // group_size  # K = P // G
        self.scale       = qk_scale or head_dim ** -0.5
        self.order_index = order_index
        self.attn_drop   = attn_drop

        # ── QKV projections ───────────────────────────────────────────────
        # Separate projections let local and global heads learn different
        # attention patterns (local spatial structure vs global context).
        self.local_qkv  = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.global_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)

        # ── Output ────────────────────────────────────────────────────────
        self.proj      = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)

        # ── LayerNorm applied to group latents before global attention ────
        # This stabilises training; unlike the per-token norm in Block,
        # this acts on the G compressed group vectors per patch.
        self.norm_latent = nn.LayerNorm(channels)

        # ── PointROPE — shared; applied independently for each phase ─────
        self.rope = PointROPE(freq=rope_freq)

    # ── Private helpers ────────────────────────────────────────────────────

    def _apply_rope(
        self,
        q:   torch.Tensor,   # [N, H, head_dim]   float16
        k:   torch.Tensor,   # [N, H, head_dim]
        pos: torch.Tensor,   # [N, 3]              float32 grid coords
    ):
        """Apply PointROPE to Q and K with given 3D positions."""
        # PointROPE API: inputs [1, H, N, head_dim] and [1, N, 3]
        q_r = q.transpose(0, 1).unsqueeze(0)   # [1, H, N, head_dim]
        k_r = k.transpose(0, 1).unsqueeze(0)
        p_r = pos.unsqueeze(0)                  # [1, N, 3]
        q_r = self.rope(q_r.float(), p_r).to(q.dtype)
        k_r = self.rope(k_r.float(), p_r).to(k.dtype)
        return q_r.squeeze(0).transpose(0, 1), k_r.squeeze(0).transpose(0, 1)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, point: Point) -> Point:
        H = self.num_heads
        C = self.channels
        P = self.patch_size    # tokens per patch
        G = self.group_size    # groups per patch
        K = self.local_size    # tokens per group  (K = P // G)

        # ── Serialization & padding ───────────────────────────────────────
        # get_padding_and_inverse pads N_real tokens to the nearest multiple
        # of P, so all patches are exactly P tokens (some may be padding).
        pad, unpad, cu_seqlens = point.get_padding_and_inverse(P)
        
        # n_patches = len(cu_seqlens) - 1    ← this is batch size, NOT patch count

        order   = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        feat = point.feat[order]
        pos  = point.grid_coord[order]
        N_real = feat.shape[0]          # actual token count; NOT guaranteed multiple of P
        input_dtype = feat.dtype

        # ── Physical pad to multiple of P = G × K ────────────────────────────────
        # flash_attn_varlen handles variable-length scenes natively via cu_seqlens,
        # so N_real across the batch may have a remainder mod P. We pad explicitly
        # here so the group reshape is always valid, then trim at the end.
        pad_size = (P - N_real % P) % P   # 0 when already aligned
        if pad_size > 0:
            feat = torch.cat([feat, feat[-1:].expand(pad_size, -1)], dim=0)
            pos  = torch.cat([pos,  pos[-1:].expand(pad_size, -1)], dim=0)

        N_padded = N_real + pad_size    # guaranteed multiple of P
        n_patches = N_padded // P

        input_dtype = feat.dtype      # preserve for final output

        # ── Step 1: Local intra-group self-attention ──────────────────────
        # Group tokens: [N_padded, C] → [n_patches×G, K, C]
        # Consecutive K tokens (in Hilbert/Z-order) form a spatially compact group.
        feat_grouped = feat.view(n_patches * G, K, C)
        pos_grouped  = pos.view(n_patches * G, K, 3)

        # QKV projection
        lqkv = self.local_qkv(feat_grouped)        # [n_patches*G, K, 3C]
        lq, lk, lv = lqkv.half().chunk(3, dim=-1)  # each [n_patches*G, K, C]

        # Flatten to [n_patches*G*K, H, head_dim] for PointROPE
        lq = lq.reshape(-1, H, C // H)
        lk = lk.reshape(-1, H, C // H)
        lv = lv.reshape(-1, H, C // H)
        flat_pos = pos_grouped.reshape(-1, 3)

        lq, lk = self._apply_rope(lq, lk, flat_pos)

        # Pack for flash_attn: [n_patches*G*K, 3, H, head_dim]
        local_qkv_packed = torch.stack([lq, lk, lv], dim=1)

        # cu_seqlens for local: each of n_patches×G segments has exactly K tokens
        local_cu = torch.arange(
            0, (n_patches * G + 1) * K, K,
            device=feat.device, dtype=torch.int32
        )

        local_out = flash_attn.flash_attn_varlen_qkvpacked_func(
            local_qkv_packed,
            local_cu,
            max_seqlen=K,
            dropout_p=self.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
        ).reshape(n_patches * G, K, C).to(input_dtype)   # [n_patches*G, K, C]

        # ── Step 2: Compress — mean-pool each group ───────────────────────
        # Group latent: mean of K member token features
        group_latents = local_out.mean(dim=1)      # [n_patches*G, C]
        # Group centroid: mean of K member 3D grid coordinates
        # group_pos = pos_grouped.mean(dim=1)        # [n_patches*G, 3]
        group_pos = pos_grouped.float().mean(dim=1).long()   # [n_patches*G, 3] Long

        # ── Step 3: Global inter-group self-attention (LaRa core idea) ────
        # LayerNorm on group latents for training stability
        latents_normed = self.norm_latent(group_latents)  # [n_patches*G, C]

        # Reshape to [n_patches, G, C] for batch-friendly QKV
        latents_2d   = latents_normed.view(n_patches, G, C)
        latent_pos_2d = group_pos.view(n_patches, G, 3)

        gqkv = self.global_qkv(latents_2d)          # [n_patches, G, 3C]
        gq, gk, gv = gqkv.half().chunk(3, dim=-1)   # each [n_patches, G, C]

        # Flatten to [n_patches*G, H, head_dim]
        gq = gq.reshape(-1, H, C // H)
        gk = gk.reshape(-1, H, C // H)
        gv = gv.reshape(-1, H, C // H)
        flat_latent_pos = latent_pos_2d.reshape(-1, 3)

        # PointROPE on group centroid positions
        gq, gk = self._apply_rope(gq, gk, flat_latent_pos)

        # Pack for flash_attn: [n_patches*G, 3, H, head_dim]
        global_qkv_packed = torch.stack([gq, gk, gv], dim=1)

        # cu_seqlens for global: each of n_patches segments has exactly G latents
        global_cu = torch.arange(
            0, (n_patches + 1) * G, G,
            device=feat.device, dtype=torch.int32
        )

        global_out = flash_attn.flash_attn_varlen_qkvpacked_func(
            global_qkv_packed,
            global_cu,
            max_seqlen=G,
            dropout_p=self.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
        ).reshape(n_patches, G, C).to(input_dtype)    # [n_patches, G, C]

        # ── Step 4: Broadcast — add global context back to member tokens ──
        # Each of the K tokens in group g receives global_out[patch, g] as
        # an additive bias (residual broadcast from the global reasoning phase).
        # [n_patches, G, C] → [n_patches, G, K, C] → [n_patches, G*K, C]
        global_broadcast = (
            global_out.unsqueeze(2)             # [n_patches, G, 1, C]
                       .expand(-1, -1, K, -1)   # [n_patches, G, K, C]
                       .reshape(N_padded, C)     # [N_padded, C]
        )

        # Local + global residual: [N_padded, C]
        combined = local_out.view(N_padded, C) + global_broadcast

        # ── Step 5: Output projection ─────────────────────────────────────
        out = self.proj(combined)
        out = self.proj_drop(out)

        # Unshuffle back to original point order: [N_real, C]
        # point.feat = out[inverse]
        # Trim physical padding, then unshuffle to original point order
        point.feat = out[:N_real][inverse]

        return point

    # ── Repr ───────────────────────────────────────────────────────────────
    def extra_repr(self) -> str:
        K = self.local_size
        return (
            f"channels={self.channels}, heads={self.num_heads}, "
            f"P={self.patch_size}, G={self.group_size}, K={K}, "
            f"attn=O({self.group_size}×{K}²+{self.group_size}²)"
            f" vs flat=O({self.patch_size}²)"
        )