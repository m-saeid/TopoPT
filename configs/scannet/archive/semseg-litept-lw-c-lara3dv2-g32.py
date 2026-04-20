"""
configs/scannet/semseg-litept-lw-c-lara3dv2-g32.py
====================================================
LitePT-LW-C + LaRA3D-V2(G=32)   [Strategy-2b  Row S8-V2  —  KEY generality test]

Backbone: lw-c (S4: depth+channel reduction, P=1024 patches unchanged)
Attention: GroupLatentAttentionV2 (Perceiver-style) on encoder stages 3 and 4

  enc_channels  = (36, 54, 108, 180, 360)   — stage 3: C=180, stage 4: C=360
  enc_patch_size = (1024, 1024, 1024, 1024, 1024)   — full patches
  enc_attn_type = ("flat", "flat", "flat", "group_v2", "group_v2")
  group_size = 32   →   K = 1024 // 32 = 32 tokens per group
  num_latents = 1   →   1 latent per group
  gate_init = 0.0   →   starts as flat attention, learns injection

What this answers:
  "Does LaRA3D-V2 improve on the lw-c flat baseline (74.95%)?"

  If YES: LaRA3D-V2 is beneficial even without patch reduction.
    Hierarchical local+global reasoning > flat global attention at P=1024.
    This is a STRONGER paper claim: V2 is a general replacement, not just
    a patch-reduction compensator.

  If NEUTRAL (within ±0.1%): V2 is patch-reduction-specific (still valid).

Key comparison to V1:
  lw-c + LaRA3D-V1 (mean-pool):  73.99% test  (−0.96% vs lw-c)
  lw-c + LaRA3D-V2 (Perceiver):  expected ≥ 74.95%  (≥ baseline)

Parameter count:
  lw-c flat:     5.84M params
  lw-c + V2-G32: ~7.1M params  (+1.3M for cross-attn projections + latents)
  lw-c + V1-G32: 7.02M params

  V2 adds: latent_to_q + token_to_k + token_to_v + broadcast_proj + gate
  At stage 3 (C=180): 4 × 180² = 129,600 extra params per block × 4 blocks = 518K
  At stage 4 (C=360): 4 × 360² = 518,400 extra params per block × 2 blocks = 1,037K
  Total extra: ~1.55M → ~7.39M total

GFLOPs: same as lw-c (12.95) since group_attn FLOPs < profiler measurement error

Expected test mIoU: ≥ 74.95% (target: 75.2–75.5%)
GPU: RTX 3090 (24 GB), ~20h

PointROPE constraint:
  enc_channels = (36, 54, 108, 180, 360) = 18×(2,3,6,10,20)
  enc_num_head = (2, 3, 6, 10, 20)
  head_dim = 18 at every stage → 18 % 6 == 0 ✓
"""

_base_ = ["../_base_/default_runtime.py"]

batch_size  = 12
num_worker  = 24
mix_prob    = 0.8
empty_cache = False
enable_amp  = True
clip_grad   = 1.0

save_path = "exp/scannet/semseg-litept-lw-c-lara3dv2-g32"

model = dict(
    type="DefaultSegmentorV2",
    num_classes=20,
    backbone_out_channels=54,
    backbone=dict(
        type="LitePT",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # ── Same as lw-c (S4) — no architecture changes ───────────────────
        enc_depths=(2, 2, 2, 4, 2),
        enc_channels=(36, 54, 108, 180, 360),
        enc_num_head=(2, 3, 6, 10, 20),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),   # FULL patches P=1024
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(54, 54, 108, 180),
        dec_num_head=(3, 3, 6, 10),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        # ── LaRA3D-V2 on stages 3+4 ──────────────────────────────────────
        enc_attn_type=("flat", "flat", "flat", "group_v2", "group_v2"),
        dec_attn_type=("flat", "flat", "flat", "flat"),
        group_size=32,       # G=32 → K = 1024 // 32 = 32 tokens/group
        num_latents=1,       # 1 Perceiver latent per group
        gate_init=0.0,       # start as flat, gradually learn injection
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enc_mode=False,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

epoch      = 1200
eval_epoch = 100
optimizer  = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler  = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

dataset_type = "ScanNetDataset"
data_root    = "data/scannet"

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=[
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
        "door", "window", "bookshelf", "picture", "counter", "desk",
        "curtain", "refridgerator", "shower curtain", "toilet", "sink",
        "bathtub", "otherfurniture",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv",
                 mode="train", return_grid_coord=True),
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": 0.02}),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment", "grid_size"),
                 feat_keys=("color", "normal")),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv",
                 mode="train", return_grid_coord=True, return_inverse=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                 feat_keys=("color", "normal")),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect",
                     keys=("coord", "grid_coord", "index"),
                     feat_keys=("color", "normal")),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle",
                      angle=[0],   axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle",
                      angle=[1/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle",
                      angle=[1],   axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle",
                      angle=[3/2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle",
                      angle=[0],   axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle",
                      angle=[1/2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle",
                      angle=[1],   axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle",
                      angle=[3/2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle",
                      angle=[0],   axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle",
                      angle=[1/2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle",
                      angle=[1],   axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle",
                      angle=[3/2], axis="z", center=[0, 0, 0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomFlip", p=1)],
            ],
        ),
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=3),
    dict(type="PreciseEvaluator", test_last=False),
]