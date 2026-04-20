"""
configs/scannet/semseg-litept-lw-c-lara3d-g32.py
==================================================
LitePT-LW-C + LaRA3D(G=32)  [Strategy-2  Generality ablation]

Applies LaRA3D GroupLatentAttention on the lw-c backbone (depth+channel
reduction, but NO patch-size reduction; enc_patch_size=1024 at stages 3,4).

  group_size=32  →  K = 1024 // 32 = 32 tokens per group

This ablation answers a separate and important question:
  "Does LaRA3D help even when patches are not reduced (P=1024)?"

If lw-c+LaRA3D (P=1024, G=32, K=32) beats lw-c (flat P=1024):
  → LaRA3D is beneficial as a general replacement, not just a patch-reduction
    compensator. The local+global hierarchy is intrinsically better than flat
    global attention, even at large patch sizes.
  → This would be a STRONGER paper claim.

If lw-c+LaRA3D ≈ lw-c (flat):
  → LaRA3D's benefit is specific to the small-patch regime (patch-reduction
    compensation). This is still a valid and useful paper claim.

Attention FLOPs comparison (per block, stage 3, C=180, P=1024):
  lw-c flat:          P² = 1024² = 1,048,576 attention ops
  lw-c+LaRA3D(G=32):  G×K² + G² = 32×1024 + 1024 = 33,792  → 31× cheaper

Parameter delta vs lw-c:
  Same as lw-e case: +1.17M → ~7.01M total params

Architecture:
  All same as lw-c EXCEPT enc_attn_type and group_size
"""

_base_ = ["../_base_/default_runtime.py"]

batch_size  = 12
num_worker  = 24
mix_prob    = 0.8
empty_cache = False
enable_amp  = True
clip_grad   = 1.0

save_path = "exp/scannet/semseg-litept-lw-c-lara3d-g32"

model = dict(
    type="DefaultSegmentorV2",
    num_classes=20,
    backbone_out_channels=54,
    backbone=dict(
        type="LitePT",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # ── Same as lw-c (S4) ─────────────────────────────────────────────
        enc_depths=(2, 2, 2, 4, 2),
        enc_channels=(36, 54, 108, 180, 360),
        enc_num_head=(2, 3, 6, 10, 20),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),  # FULL patches (P=1024)
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(54, 54, 108, 180),
        dec_num_head=(3, 3, 6, 10),
        dec_patch_size=(1024, 1024, 1024, 1024),  # dec_patch[s]=enc_patch[s]
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        # ── LaRA3D on full patches (P=1024, G=32, K=32) ──────────────────
        enc_attn_type=("flat", "flat", "flat", "group", "group"),
        dec_attn_type=("flat", "flat", "flat", "flat"),
        group_size=32,   # G=32 → K = 1024//32 = 32 tokens/group
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
        type=dataset_type, split="train", data_root=data_root,
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
        type=dataset_type, split="val", data_root=data_root,
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
        type=dataset_type, split="val", data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=0.02, hash_type="fnv",
                          mode="test", return_grid_coord=True),
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