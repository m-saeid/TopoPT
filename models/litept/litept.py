"""
models/litept/litept.py  — LitePT backbone with LaRA3D V2 GroupLatentAttentionV2.

Changes vs original:
  1. Imports both GroupLatentAttention (v1) and GroupLatentAttentionV2 from group_latent_attn.py
  2. Block gains `attn_type` parameter: "flat" | "group_v1" | "group_v2"
     - "flat"     → original PointROPEAttention  (default, backward-compatible)
     - "group_v1" → GroupLatentAttention (v1, mean-pool — kept for ablation)
     - "group_v2" → GroupLatentAttentionV2 (Perceiver-style — RECOMMENDED)
  3. Block gains `group_size` and `num_latents` parameters
  4. LitePT gains `enc_attn_type`, `dec_attn_type`, `group_size`, `num_latents` parameters

Backward-compatibility: All existing configs that do not set `enc_attn_type`
default to all-"flat" (original PointROPEAttention), unchanged behaviour.
"""

from functools import partial

import torch
import torch.nn as nn
import spconv.pytorch as spconv
import flash_attn
from timm.layers import DropPath

from libs.pointrope import PointROPE
from models.builder import MODELS
from models.modules import PointModule, PointSequential, Embedding, GridPooling, GridUnpooling
from models.utils.structure import Point

# ── Import both LaRA3D versions ───────────────────────────────────────────
from models.litept.group_latent_attn import GroupLatentAttention, GroupLatentAttentionV2


# ─────────────────────────────────────────────────────────────────────────────
#  Original PointROPEAttention (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class PointROPEAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        rope_freq,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels    = channels
        self.num_heads   = num_heads
        self.scale       = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.patch_size  = patch_size
        self.attn_drop   = attn_drop

        self.qkv       = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj      = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax   = torch.nn.Softmax(dim=-1)
        self.rope      = PointROPE(freq=rope_freq)

    def forward(self, point):
        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = point.get_padding_and_inverse(self.patch_size)
        order   = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]
        pos = point.grid_coord[order].reshape(-1, 3).unsqueeze(0)

        q, k, v = qkv.half().chunk(3, dim=-1)
        q = q.reshape(-1, H, C // H).transpose(0,1)[None]
        k = k.reshape(-1, H, C // H).transpose(0,1)[None]

        q = self.rope(q.float(), pos).to(q.dtype)
        k = self.rope(k.float(), pos).to(k.dtype)

        qkv_rotated = torch.stack([
            q.squeeze(0).transpose(0,1),
            k.squeeze(0).transpose(0,1),
            v.reshape(-1, H, C // H)
        ], dim=1)

        feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv_rotated,
            cu_seqlens,
            max_seqlen=self.patch_size,
            dropout_p=self.attn_drop if self.training else 0,
            softmax_scale=self.scale,
        ).reshape(-1, C)

        feat = feat.to(qkv.dtype)
        feat = feat[inverse]
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels    = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1  = nn.Linear(in_channels, hidden_channels)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  Block  — supports three attention modes
# ─────────────────────────────────────────────────────────────────────────────

class Block(PointModule):
    """
    LitePT transformer block with pluggable attention.

    attn_type options:
      "flat"     — original PointROPEAttention (default, backward-compatible)
      "group_v1" — GroupLatentAttention v1 (mean-pool; kept for ablation)
      "group_v2" — GroupLatentAttentionV2  (Perceiver-style; RECOMMENDED)
    """

    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None, 
        enable_conv=True,
        enable_attn=True,
        rope_freq=100.0,
        # ── Group-latent attention parameters ───────────────────────────
        attn_type: str  = "flat",   # "flat" | "group_v1" | "group_v2"
        group_size: int = 32,       # G (used when attn_type != "flat")
        num_latents: int = 1,       # n_lat (used only when attn_type == "group_v2")
        gate_init: float = 0.0,     # initial gate value (V2 only)
        key_prefix: str = "",
    ):
        super().__init__()
        self.channels    = channels
        self.pre_norm    = pre_norm
        self.enable_conv = enable_conv
        self.enable_attn = enable_attn
        self.attn_type   = attn_type

        if self.enable_conv:
            self.conv = PointSequential(
                spconv.SubMConv3d(
                    channels, channels, kernel_size=3, bias=True,
                    # indice_key=cpe_indice_key,
                    indice_key=f"{key_prefix}{cpe_indice_key}",   # prefixed
                ),
                nn.Linear(channels, channels),
                norm_layer(channels),
            )
        else:
            self.norm0 = PointSequential(norm_layer(channels))

        if self.enable_attn:
            self.norm1 = PointSequential(norm_layer(channels))

            # ── Select attention module ───────────────────────────────────
            if attn_type == "group_v2":
                if patch_size % group_size != 0:
                    raise ValueError(
                        f"patch_size ({patch_size}) must be divisible by "
                        f"group_size ({group_size}) for attn_type='group_v2'."
                    )
                self.attn = GroupLatentAttentionV2(
                    channels=channels,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    group_size=group_size,
                    rope_freq=rope_freq,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    order_index=order_index,
                    num_latents=num_latents,
                    gate_init=gate_init,
                )
            elif attn_type == "group_v1":
                if patch_size % group_size != 0:
                    raise ValueError(
                        f"patch_size ({patch_size}) must be divisible by "
                        f"group_size ({group_size}) for attn_type='group_v1'."
                    )
                self.attn = GroupLatentAttention(
                    channels=channels,
                    num_heads=num_heads,
                    patch_size=patch_size,
                    group_size=group_size,
                    rope_freq=rope_freq,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    order_index=order_index,
                )
            else:
                # "flat" — original PointROPEAttention (default)
                self.attn = PointROPEAttention(
                    channels=channels,
                    patch_size=patch_size,
                    rope_freq=rope_freq,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    order_index=order_index,
                )

            self.norm2     = PointSequential(norm_layer(channels))
            self.mlp       = PointSequential(
                MLP(
                    in_channels=channels,
                    hidden_channels=int(channels * mlp_ratio),
                    out_channels=channels,
                    act_layer=act_layer,
                    drop=proj_drop,
                )
            )
            self.drop_path = PointSequential(
                DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
            )

    def forward(self, point: Point):
        if self.enable_conv:
            shortcut = point.feat
            point = self.conv(point)
            point.feat = shortcut + point.feat
        else:
            point = self.norm0(point)

        if self.enable_attn:
            shortcut = point.feat
            if self.pre_norm:
                point = self.norm1(point)
            point = self.drop_path(self.attn(point))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm1(point)

            shortcut = point.feat
            if self.pre_norm:
                point = self.norm2(point)
            point = self.drop_path(self.mlp(point))
            point.feat = shortcut + point.feat
            if not self.pre_norm:
                point = self.norm2(point)

        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


# ─────────────────────────────────────────────────────────────────────────────
#  LitePT backbone
# ─────────────────────────────────────────────────────────────────────────────

@MODELS.register_module("LitePT")
class LitePT(PointModule):
    """
    LitePT backbone with optional LaRA3D V2 GroupLatentAttentionV2.

    New parameters (all optional, backward-compatible):
    ─────────────────────────────────────────────────────────────
    enc_attn_type : tuple[str]  (length = num_stages)
        Per-stage attention type for the encoder.
        "flat"     — original PointROPEAttention (default for all stages)
        "group_v1" — GroupLatentAttention (mean-pool; for ablation)
        "group_v2" — GroupLatentAttentionV2 (Perceiver; RECOMMENDED)
        Only stages where enc_attn[s]=True actually use attention modules;
        enc_attn_type for conv-only stages is ignored.

    dec_attn_type : tuple[str]  (length = num_stages-1)
        Same for decoder stages. Defaults to all "flat".

    group_size : int  (default 32)
        G — number of groups per patch.
        Must divide every patch_size where attn_type != "flat".

    num_latents : int  (default 1)
        n_lat — number of latent vectors per group (V2 only).
        1 is sufficient for most cases. 2 gives richer representation.

    gate_init : float  (default 0.0)
        Initial value of the injection gate for V2.
        0.0 → tanh(0) = 0 → model starts as flat attention baseline.
        The model learns to gradually increase injection during training.

    Typical config for LaRA3D-V2 on lw-c backbone:
    ─────────────────────────────────────────────────────────────
    enc_attn_type = ("flat", "flat", "flat", "group_v2", "group_v2")
    group_size    = 32       # G=32, K = 1024//32 = 32 for P=1024
    num_latents   = 1        # one group summary per group
    gate_init     = 0.0      # start as flat, learn to inject
    """

    def __init__(
        self,
        in_channels=4,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(36, 72, 144, 252, 504),
        enc_num_head=(2, 4, 8, 14, 28),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 72, 144, 252),
        dec_num_head=(4, 4, 8, 14),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enc_mode=False,
        # ── New parameters (backward-compatible defaults) ────────────────
        enc_attn_type=None,     # tuple[str] or None → all "flat"
        dec_attn_type=None,     # tuple[str] or None → all "flat"
        group_size: int   = 32,
        num_latents: int  = 1,
        gate_init: float  = 0.0,
        key_prefix: str = "",
    ):
        super().__init__()
        self.num_stages   = len(enc_depths)
        self.order        = [order] if isinstance(order, str) else order
        self.enc_mode     = enc_mode
        self.shuffle_orders = shuffle_orders
        self.enc_conv     = enc_conv
        self.enc_attn     = enc_attn
        self.dec_conv     = dec_conv
        self.dec_attn     = dec_attn
        self.key_prefix   = key_prefix

        # ── Resolve attn_type tuples ──────────────────────────────────────
        if enc_attn_type is None:
            enc_attn_type = tuple("flat" for _ in range(self.num_stages))
        if dec_attn_type is None:
            dec_attn_type = tuple("flat" for _ in range(self.num_stages - 1))

        self.enc_attn_type = enc_attn_type
        self.dec_attn_type = dec_attn_type
        self.group_size    = group_size
        self.num_latents   = num_latents
        self.gate_init     = gate_init

        # ── Assertions ────────────────────────────────────────────────────
        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.num_stages == len(enc_attn_type)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1

        # ── Validate group_size compatibility ─────────────────────────────
        for s in range(self.num_stages):
            if enc_attn[s] and enc_attn_type[s] in ("group_v1", "group_v2"):
                P = enc_patch_size[s]
                if P % group_size != 0:
                    raise ValueError(
                        f"enc_patch_size[{s}]={P} must be divisible by "
                        f"group_size={group_size}."
                    )

        # ── Norm / activation ─────────────────────────────────────────────
        bn_layer  = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        ln_layer  = nn.LayerNorm
        act_layer = nn.GELU

        # ── Embedding ─────────────────────────────────────────────────────
        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
            indice_key_stem=f"{key_prefix}stem",
        )

        # ── Encoder ───────────────────────────────────────────────────────
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]): sum(enc_depths[:s+1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        re_serialization=enc_attn[s],
                        serialization_order=self.order,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_conv=enc_conv[s],
                        enable_attn=enc_attn[s],
                        rope_freq=enc_rope_freq[s],
                        attn_type=enc_attn_type[s],
                        group_size=group_size,
                        num_latents=num_latents,
                        gate_init=gate_init,
                        key_prefix=key_prefix,                  # thread through
                    ),
                    name=f"block{i}",
                )

            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # ── Decoder ───────────────────────────────────────────────────────
        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels_ext = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]): sum(dec_depths[:s+1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels_ext[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels_ext[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels_ext[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_conv=dec_conv[s],
                            enable_attn=dec_attn[s],
                            rope_freq=dec_rope_freq[s],
                            attn_type=dec_attn_type[s] if s < len(dec_attn_type) else "flat",
                            group_size=group_size,
                            num_latents=num_latents,
                            gate_init=gate_init,
                            key_prefix=key_prefix,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

    def forward(self, data_dict):
        point = Point(data_dict)
        if self.enc_attn[0]:
            point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        point = self.enc(point)

        if not self.enc_mode:
            point = self.dec(point)

        return point