"""
configs/structured3d/semseg-lw-c-100epoch.py
===========================================
LitePT-LW-C  Semantic Segmentation — Structured3D
100-epoch fast-run variant (halved from the 200-epoch baseline).

bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c-100epoch -n semseg-lw-c-100epoch

Differences vs semseg-lw-c-200epoch.py
(MODEL ARCHITECTURE IS UNCHANGED — only training hyper-parameters are modified):

  1. epoch            : 200  → 100
       Rationale: time budget cut in half for a faster Structured3D run.

  2. eval_epoch       : 100  → 50
       Rationale: proportional reduction so validation starts at the halfway
       point rather than right at the end.  Keeps the same "evaluate the second
       half of training" behaviour as the 200-epoch schedule.

  3. optimizer lr     : 0.012 → 0.018  (×1.5)
  4. scheduler max_lr : [0.012, 0.0012] → [0.018, 0.0018]  (×1.5)
  5. param_dicts lr   : 0.0012 → 0.0018  (×1.5)
       Rationale: halving total training steps shrinks the cumulative gradient
       signal by ~2×.  The linear-scaling rule for OneCycleLR suggests scaling
       peak LR by √(200/100) = √2 ≈ 1.41 when halving steps; we use the
       slightly more conservative 1.5× to avoid instability while still
       recovering the lost capacity.  The ratio between the backbone LR
       (0.0018) and the head LR (0.018) is preserved at 1:10.
       Expected effect: the model reaches a comparable loss surface in 100
       epochs to what the 1×LR schedule finds in 200 epochs.

  6. pct_start        : 0.05 → 0.10
       Rationale: with 200 epochs, pct_start=0.05 gives 10 epochs of warmup.
       At 100 epochs, keeping 0.05 would give only 5 epochs — too short at a
       higher peak LR and risks divergence in the first few epochs.  Setting
       0.10 restores the same absolute 10-epoch warmup duration.
       Expected effect: stable loss at the start of training even with the
       higher peak LR.

  7. save_freq        : 3 → 2
       Rationale: proportional reduction (200/3 ≈ 67 checkpoints; 100/2 = 50
       checkpoints) so disk usage and checkpoint granularity stay comparable.

  8. empty_cache      : False → True   (copied from lw-c baseline; keeps GPU
       memory headroom stable when running with a small batch on a single GPU)

No changes to:
  - Model architecture (backbone, head, backbone_out_channels, criteria)
  - Data pipeline / augmentation (all transforms, GridSample, SphereCrop, etc.)
  - Weight decay, div_factor, final_div_factor, mix_prob, clip_grad
"""

_base_ = ["../_base_/default_runtime.py"]

batch_size  = 3
num_worker  = 4
mix_prob    = 0.8
empty_cache = True
enable_amp  = True
clip_grad   = 1.0

# ── [CHANGED 1/7] save_path updated to reflect 100-epoch variant ──────────────
save_path = "exp/structured3d/semseg-lw-c-50epoch"

# ── MODEL: identical to semseg-lw-c-200epoch.py — DO NOT MODIFY ───────────────
model = dict(
    type="DefaultSegmentorV2",
    num_classes=25,
    backbone_out_channels=54,
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
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# ── SCHEDULER ─────────────────────────────────────────────────────────────────
# [CHANGED 2/7] epoch: 200 → 100 (time-budget reduction).
epoch      = 100

# [CHANGED 3/7] eval_epoch: 100 → 50.
#   200-epoch schedule: validation begins at epoch 100 (second half of training).
#   100-epoch schedule: validation begins at epoch 50 (same proportional point).
eval_epoch = 50

# [CHANGED 4/7] optimizer base lr: 0.012 → 0.018 (×1.5).
#   See rationale in module docstring point 5.
optimizer  = dict(type="AdamW", lr=0.018, weight_decay=0.05)

scheduler  = dict(
    type="OneCycleLR",
    # [CHANGED 5/7] max_lr: [0.012, 0.0012] → [0.018, 0.0018] (×1.5).
    #   Backbone-block LR (0.0018) and head LR (0.018) maintain the 1:10 ratio.
    max_lr=[0.018, 0.0018],
    # [CHANGED 6/7] pct_start: 0.05 → 0.10.
    #   Restores 10 absolute warmup epochs that the original 200-epoch schedule
    #   achieved; prevents divergence with the elevated peak LR.
    pct_start=0.10,
    anneal_strategy="cos",
    div_factor=10.0,        # initial_lr = max_lr / 10 — unchanged
    final_div_factor=1000.0,  # min_lr = initial_lr / 1000 — unchanged
)

# [CHANGED 4/7 continued] block lr: 0.0012 → 0.0018 (×1.5, paired with optimizer).
param_dicts = [dict(keyword="block", lr=0.0018)]

# ── DATASET: identical to semseg-lw-c-200epoch.py — DO NOT MODIFY ─────────────
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
    # [CHANGED 7/7] save_freq: 3 → 2.
    #   Proportional reduction: 200 epochs / save_freq=3 ≈ 67 saves;
    #   100 epochs / save_freq=2 = 50 saves — similar checkpoint density.
    dict(type="CheckpointSaver", save_freq=2),
    dict(type="PreciseEvaluator", test_last=False),
]
