"""
configs/scannet/semseg-litept-lw-e-lara3d-g32.py
==================================================
LitePT-LW-E + LaRA3D(G=32)  [Strategy-2  Row S6★]

KEY EXPERIMENT — the main paper contribution.

Backbone: identical to lw-e (S5):
  enc_depths    = (2, 2, 2, 4, 2)
  enc_channels  = (36, 54, 108, 180, 360) = 18×(2,3,6,10,20)
  enc_patch_sz  = (1024, 1024, 1024, 512, 512)

NEW: at the two attention stages (3 and 4), replace flat PointROPEAttention
     with GroupLatentAttention (LaRA3D):
  enc_attn_type = ("flat", "flat", "flat", "group", "group")
  group_size    = 32   → K = patch_size // G = 512 // 32 = 16 tokens/group

GroupLatentAttention pipeline (per patch of P=512 tokens):
  ─────────────────────────────────────────────────────────
  Step 1 Local:    32 groups × 16 tokens → intra-group self-attn (O(16²)=256)
  Step 2 Compress: mean-pool each group → 32 group latents + centroid coords
  Step 3 Global:   32 latents mutual self-attn (O(32²)=1024) ← LaRa principle
  Step 4 Broadcast: group context added to all 16 member tokens
  ─────────────────────────────────────────────────────────
  Total attention ops: 32×256 + 1024 = 9216  vs  512² = 262144 (flat)
  Speed-up (attention): ~28× per block

Paper motivation:
  Strategy-1 shows patch-size reduction (S3/lw-d) costs -1.10% mIoU while
  depth+channel reduction (S4/lw-c) costs only -0.32% mIoU. Patch size is
  the dominant accuracy bottleneck in the fully-lightweight model (S5/lw-e).
  LaRA3D addresses this by restoring global-within-patch context at O(G·K²)
  cost, aiming to recover the patch-reduction penalty without increasing P.

Expected outcome:
  mIoU  : > 74.95% (lw-c), ideally ~75.0–75.2%
  Params : ~7.01M  (+1.17M vs lw-e due to global_qkv + norm_latent)
  GFLOPs : ~9.65   (< lw-c's 12.95 AND < lw-e's 11.22 in attention stages)
  Latency: similar to lw-e (~20ms), since local attn on K=16 is very fast

Architecture summary (head_dim = 18, PointROPE satisfied: 18%6=0 ✓):
  stage 0: 36ch / 2h = 18  conv  flat   ✓
  stage 1: 54ch / 3h = 18  conv  flat   ✓
  stage 2: 108ch/ 6h = 18  conv  flat   ✓
  stage 3: 180ch/10h = 18  attn  GROUP  G=32, K=16  ✓
  stage 4: 360ch/20h = 18  attn  GROUP  G=32, K=16  ✓
"""

_base_ = ["../_base_/default_runtime.py"]

batch_size  = 12
num_worker  = 24
mix_prob    = 0.8
empty_cache = False
enable_amp  = True
clip_grad   = 1.0

save_path = "exp/scannet/semseg-litept-lw-e-lara3d-g32"

model = dict(
    type="DefaultSegmentorV2",
    num_classes=20,
    backbone_out_channels=54,        # = enc_channels[1]
    backbone=dict(
        type="LitePT",
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        # ── Same as lw-e (S5) ─────────────────────────────────────────────
        enc_depths=(2, 2, 2, 4, 2),
        enc_channels=(36, 54, 108, 180, 360),       # 18×(2,3,6,10,20)
        enc_num_head=(2, 3, 6, 10, 20),              # head_dim=18 ✓
        enc_patch_size=(1024, 1024, 1024, 512, 512), # small patches at attn stages
        enc_conv=(True, True, True, False, False),
        enc_attn=(False, False, False, True, True),
        enc_rope_freq=(100.0, 100.0, 100.0, 100.0, 100.0),
        dec_depths=(0, 0, 0, 0),
        dec_channels=(54, 54, 108, 180),
        dec_num_head=(3, 3, 6, 10),
        dec_patch_size=(1024, 1024, 1024, 512),     # dec_patch[s] = enc_patch[s]
        dec_conv=(False, False, False, False),
        dec_attn=(False, False, False, False),
        dec_rope_freq=(100.0, 100.0, 100.0, 100.0),
        # ── LaRA3D parameters (NEW) ───────────────────────────────────────
        enc_attn_type=("flat", "flat", "flat", "group", "group"),
        dec_attn_type=("flat", "flat", "flat", "flat"),  # dec_depths=0, not used
        group_size=32,   # G=32 → K = 512//32 = 16 tokens/group
        # ── Shared hyperparams ───────────────────────────────────────────
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