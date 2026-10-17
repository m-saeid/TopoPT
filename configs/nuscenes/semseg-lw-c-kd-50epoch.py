"""
configs/nuscenes/semseg-lw-c-kd.py
==========================================
LitePT-LW-C + SRFD KD — NuScenes Semantic Segmentation

bash scripts/train.sh -g 1 -d nuscenes -c semseg-lw-c-kd -n semseg-lw-c-kd
     ### -w exp/nuscenes/semseg-lw-c/model/model_best.pth

STUDENT : lw-c backbone + DefaultSegmentorV2
TEACHER : full LitePT from exp/nuscenes/semseg-litept-small-v1m1/model/model_best.pth

Training loss:
  L = L_CE + L_Lovász
    + 1.0 × L_pointwise_kd   (per-point cosine, stages 3+4)
    + 2.0 × L_relational_kd  (SRFD pairwise similarity, stages 3+4)

CRITICAL: in_channels=4 for BOTH student and teacher (NuScenes uses coord+strength).
"""

_base_ = ["../_base_/default_runtime.py"]

batch_size  = 3
num_worker  = 4
mix_prob    = 0.8
empty_cache = False
enable_amp  = True

save_path = "exp/nuscenes/semseg-lw-c-kd-50epoch"

_teacher_ckpt = "exp/nuscenes/semseg-litept-small-v1m1/model/model_best.pth"

model = dict(
    type="DistillationSegmentorV2",
    num_classes=16,
    backbone_out_channels=54,

    # ── Student ───────────────────────────────────────────────────────────
    backbone=dict(
        type="LitePT",
        in_channels=4,                           # ← coord + strength, NOT 6
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 4, 2),
        enc_channels=(36, 54, 108, 180, 360),
        enc_num_head=(2, 3, 6, 10, 20),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
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

    # ── Teacher ───────────────────────────────────────────────────────────
    teacher_backbone=dict(
        type="LitePT",
        in_channels=4,                           # ← must match student: coord + strength
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
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
        shuffle_orders=True,
        pre_norm=True,
        key_prefix="teacher_",
        enc_mode=True,
    ),

    teacher_ckpt=_teacher_ckpt,
    distill_stages=(3, 4),
    pointwise_weight=1.0,
    relational_weight=2.0,
    relational_n_sample=512,

    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

epoch      = 50
eval_epoch = 50
optimizer  = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler  = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

dataset_type = "NuScenesDataset"
data_root    = "data/nuscenes"
ignore_index = -1
names = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle",
    "motorcycle", "pedestrian", "traffic_cone", "trailer", "truck",
    "driveable_surface", "other_flat", "sidewalk", "terrain",
    "manmade", "vegetation",
]

data = dict(
    num_classes=16,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="GridSample", grid_size=0.05, hash_type="fnv",
                 mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": 0.05}),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment", "grid_size"),
                 feat_keys=("coord", "strength")),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSample", grid_size=0.05, hash_type="fnv",
                 mode="train", return_grid_coord=True, return_inverse=True),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                 feat_keys=("coord", "strength")),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="GridSample", grid_size=0.025, hash_type="fnv",
                 mode="train", return_inverse=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=0.05, hash_type="fnv",
                          mode="test", return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(type="Collect",
                     keys=("coord", "grid_coord", "index"),
                     feat_keys=("coord", "strength")),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9,  0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1,    1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1,  1.1])],
                [dict(type="RandomScale", scale=[0.9,  0.9]),  dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[0.95, 0.95]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1,    1]),    dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.05, 1.05]), dict(type="RandomFlip", p=1)],
                [dict(type="RandomScale", scale=[1.1,  1.1]),  dict(type="RandomFlip", p=1)],
            ],
        ),
        ignore_index=ignore_index,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=50),
    dict(type="PreciseEvaluator", test_last=False),
]