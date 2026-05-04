"""
configs/structured3d/semseg-lw-c-kd-tb-100epoch.py
===============================================
LitePT-LW-C + SRFD KD — Structured3D Semantic Segmentation
Teacher: LitePT-Base (semseg-litept-base-v1m1)
100-epoch fast-run variant (halved from the 200-epoch baseline).

bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c-kd-tb-100epoch \
                      -n semseg-lw-c-kd-tb-100epoch

STUDENT : lw-c backbone + DistillationSegmentorV2
TEACHER : LitePT-Base from exp/structured3d/semseg-litept-base-v1m1/model/model_best.pth

Training loss (unchanged):
  L = L_CE + L_Lovász
    + 1.0 × L_pointwise_kd   (per-point cosine, stages 3+4)
    + 2.0 × L_relational_kd  (SRFD pairwise similarity, stages 3+4)

Differences vs semseg-lw-c-kd-tb-200epoch.py
(MODEL ARCHITECTURE AND DISTILLATION WEIGHTS ARE UNCHANGED — only training
 hyper-parameters are modified):

  1. epoch            : 200  → 100
       Rationale: time budget cut in half for a faster Structured3D run.

  2. eval_epoch       : 200  → 50
       Rationale: the 200-epoch config evaluated only at the final epoch.
       With 100 epochs we begin validation at epoch 50 to monitor the second
       half of training and catch the best checkpoint if it appears before the
       final epoch.

  3. optimizer lr     : 0.012 → 0.018  (×1.5)
  4. scheduler max_lr : [0.012, 0.0012] → [0.018, 0.0018]  (×1.5)
  5. param_dicts lr   : 0.0012 → 0.0018  (×1.5)
       Rationale: halving training steps roughly halves the total gradient
       signal.  The linear-scaling heuristic suggests √2 ≈ 1.41×; we apply
       1.5× as a conservative round figure.
       Base-teacher note: the capacity gap between lw-c student channels
       (180/360) and Base teacher channels (216/432 at stages 3/4) is larger
       than for the Small teacher (252/504 — Small channels are wider than
       Base at stages 3–4).  The higher LR helps the student projectors
       (φ_s : R^{student_ch} → R^{teacher_ch}) converge faster to a useful
       projection, which is more important when the epoch budget is tight.
       The backbone-block LR (0.0018) and head LR (0.018) maintain 1:10 ratio.

  6. pct_start        : 0.05 → 0.10
       Rationale: restores the original 10-epoch absolute warmup.
       At pct_start=0.05 and 100 epochs the warmup would be only 5 epochs —
       too short when a stronger Base teacher is emitting high-magnitude SRFD
       relational gradients from the very first batch.  The longer warmup at
       max_lr / div_factor = 0.0018 lets the student adapt before full LR
       exposure.

  7. save_freq        : None → None  (unchanged; saves every epoch)

  8. save_path updated to reflect 100-epoch variant.

No changes to:
  - Student/teacher model architecture (backbone channels, depths, heads, etc.)
  - Distillation stages, pointwise_weight, relational_weight, relational_n_sample
  - teacher_ckpt path
  - Data pipeline / augmentation
  - Weight decay, div_factor, final_div_factor, mix_prob, clip_grad, empty_cache
"""

_base_ = ["../_base_/default_runtime.py"]

batch_size  = 3
num_worker  = 4
mix_prob    = 0.8
empty_cache = False
enable_amp  = True
clip_grad   = 1.0

# [CHANGED 1] save_path updated to reflect the 100-epoch / Base-teacher variant.
save_path = "exp/structured3d/semseg-lw-c-kd-tb-50epoch"

_teacher_ckpt = "exp/structured3d/semseg-litept-base-v1m1/model/model_best.pth"

# ── MODEL: identical to semseg-lw-c-kd-tb-200epoch.py — DO NOT MODIFY ────────
model = dict(
    type="DistillationSegmentorV2",
    num_classes=25,
    backbone_out_channels=54,

    # ── Student (lw-c — unchanged) ────────────────────────────────────────
    backbone=dict(
        type="LitePT",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
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

    # ── Teacher: LitePT-Base (unchanged) ─────────────────────────────────
    teacher_backbone=dict(
        type="LitePT",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(72, 108, 216, 432),
        dec_num_head=(4, 6, 12, 24),
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
    pointwise_weight=1.0,   # unchanged
    relational_weight=2.0,  # unchanged
    relational_n_sample=512,

    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# ── SCHEDULER ─────────────────────────────────────────────────────────────────
# [CHANGED 2] epoch: 200 → 100.
epoch      = 100

# [CHANGED 3] eval_epoch: 200 → 50.
#   Begin validation at the halfway point of the 100-epoch schedule so we
#   can track and save the best model throughout the second half of training.
eval_epoch = 50

# [CHANGED 4] optimizer lr: 0.012 → 0.018 (×1.5).
optimizer  = dict(type="AdamW", lr=0.018, weight_decay=0.05)

scheduler  = dict(
    type="OneCycleLR",
    # [CHANGED 5] max_lr: [0.012, 0.0012] → [0.018, 0.0018] (×1.5).
    #   Compensates for halved gradient steps; 1:10 block/head ratio preserved.
    max_lr=[0.018, 0.0018],
    # [CHANGED 6] pct_start: 0.05 → 0.10.
    #   0.10 × 100 = 10 epochs of warmup, matching the original 0.05 × 200 = 10.
    #   Especially important for tb: the Base teacher is larger and its SRFD
    #   relational targets span higher-dimensional spaces (stages-3/4 teacher
    #   channels: 216 and 432), so the student projectors need the extra
    #   warmup stability at the reduced initial LR.
    pct_start=0.10,
    anneal_strategy="cos",
    div_factor=10.0,          # initial_lr = max_lr / 10 — unchanged
    final_div_factor=1000.0,  # min_lr = initial_lr / 1000 — unchanged
)

# [CHANGED 4 continued] block lr: 0.0012 → 0.0018 (×1.5).
param_dicts = [dict(keyword="block", lr=0.0018)]

# ── DATASET: identical to semseg-lw-c-kd-tb-200epoch.py — DO NOT MODIFY ──────
dataset_type = "Structured3DDataset"
data_root    = "data/structured3d"

data = dict(
    num_classes=25,
    ignore_index=-1,
    names=(
        "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
        "door", "window", "picture", "desk", "shelves", "curtain",
        "dresser", "pillow", "mirror", "ceiling", "refrigerator",
        "television", "nightstand", "sink", "lamp", "otherstructure",
        "otherfurniture", "otherprop",
    ),
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
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
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
                [dict(type="RandomRotateTargetAngle", angle=[0],   axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1],   axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0,0,0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[0],   axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[1],   axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomRotateTargetAngle", angle=[0],   axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[1],   axis="z", center=[0,0,0], p=1),
                 dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis="z", center=[0,0,0], p=1),
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
    # save_freq=None: save every epoch, same as the 200-epoch KD config.
    # 100 epochs is still manageable and ensures the best checkpoint is never
    # missed under the compressed schedule.
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
