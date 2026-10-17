# tools/warmup_teacher.py
import torch
from models.builder import build_model
from models.utils.structure import Point

teacher_cfg = dict(
    type="LitePT",
    in_channels=6,
    enc_depths=(2, 2, 2, 6, 2),
    enc_channels=(36, 72, 144, 252, 504),
    enc_num_head=(2, 4, 8, 14, 28),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    enc_conv=(True, True, True, False, False),
    enc_attn=(False, False, False, True, True),
    enc_rope_freq=(100., 100., 100., 100., 100.),
    dec_depths=(0, 0, 0, 0),
    dec_channels=(72, 72, 144, 252),
    dec_num_head=(4, 4, 8, 14),
    dec_patch_size=(1024, 1024, 1024, 1024),
    dec_conv=(False, False, False, False),
    dec_attn=(False, False, False, False),
    dec_rope_freq=(100., 100., 100., 100.),
    drop_path=0.3, mlp_ratio=4, qkv_bias=True, pre_norm=True,
    shuffle_orders=True, enc_mode=True,
    key_prefix="teacher_",
)

model = build_model(teacher_cfg).cuda().eval()

# Build a dummy sparse batch that covers the teacher's channel transitions
N = 4096
dummy = dict(
    coord=torch.rand(N, 3).cuda() * 10,
    feat=torch.rand(N, 6).cuda(),
    batch=torch.zeros(N, dtype=torch.long).cuda(),
    grid_size=torch.tensor(0.02).cuda(),
)
with torch.no_grad():
    model(dummy)

print("Teacher kernel cache warm-up complete.")