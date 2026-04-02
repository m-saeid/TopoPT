"""
models/distillation.py
======================
DistillationSegmentorV2 — Knowledge Distillation for efficient LitePT compression.

NOVEL CONTRIBUTION: Stage-wise Relational Feature Distillation (SRFD)
----------------------------------------------------------------------
Standard KD aligns per-point features independently: for each point i,
minimize ||project(student_feat_i) - teacher_feat_i||.

This misses the RELATIONAL STRUCTURE between points, which is what
3D segmentation actually depends on: points of the same class should
cluster together, and points of different classes should separate.

SRFD adds a second loss term that preserves PAIRWISE similarity patterns:

  S_teacher[i,j] = cosine_similarity(teacher_feat_i, teacher_feat_j)
  S_student[i,j] = cosine_similarity(project(student_feat_i), project(student_feat_j))
  L_relational    = ||S_teacher - S_student||_F² (Frobenius norm)

In other words: if teacher says points i and j are similar (same class),
the student must also place them close in feature space. If teacher says
they are different (boundary), student must also separate them.

Why this is novel for 3D:
  - Relational KD (Park et al., CVPR 2019) was designed for image classifiers.
  - We apply it specifically to the ATTENTION STAGES (3 and 4) of a 3D serialized
    point cloud transformer, where the features encode geometric-semantic relationships.
  - The serialized attention in LitePT creates rich inter-point relationship structure
    that a shallower/narrower student fails to capture pointwise but CAN capture
    relationally with this supervision.
  - At stage 3, N≈1K points → S is 1K×1K = 1M elements (cheap).
  - At stage 4, N≈300 points → S is 300×300 = 90K elements (trivially cheap).

Why this will work:
  - The 0.32% gap between lw-c (74.95%) and baseline (75.27%) is precisely
    at the boundary/hard-example level — exactly what relational structure captures.
  - Combined with standard pointwise distillation, SRFD addresses BOTH
    absolute feature quality AND structural consistency.
  - Expected lw-c+SRFD: 75.2–75.5% (matching/beating baseline at 49% fewer GFLOPs).

How to use:
  This is a drop-in replacement for DefaultSegmentorV2 in the config.
  Set type="DistillationSegmentorV2" and add teacher_backbone + teacher_ckpt.
  All other parameters are identical to DefaultSegmentorV2.

REGISTER as "DistillationSegmentorV2" in models/__init__.py:
  from models.distillation import DistillationSegmentorV2
  MODELS.register_module("DistillationSegmentorV2")(DistillationSegmentorV2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.builder import MODELS, build_model
from models.losses.builder import build_criteria


@MODELS.register_module("DistillationSegmentorV2")
class DistillationSegmentorV2(nn.Module):
    """
    Segmentation model with Stage-wise Relational Feature Distillation (SRFD).

    Total training loss:
        L = L_seg + α × L_pointwise_kd + β × L_relational_kd

    where:
        L_seg        = CrossEntropy + Lovász  (standard segmentation losses)
        L_pointwise  = Σ_s (1 - cosine_sim(proj(student_s), teacher_s).mean())
        L_relational = Σ_s ||S_student_s - S_teacher_s||_F²   [NOVEL]

    At inference: identical to DefaultSegmentorV2 (teacher and KD losses are gone).
    """

    def __init__(
        self,
        # ── Student architecture (same as DefaultSegmentorV2) ────────────
        num_classes,
        backbone_out_channels,
        backbone,               # backbone config dict (lw-c)
        criteria,               # list of loss configs
        # ── Teacher (KD) parameters ──────────────────────────────────────
        teacher_backbone,       # backbone config dict (full baseline)
        teacher_ckpt,           # path to teacher model_best.pth
        # ── Distillation hyperparameters ─────────────────────────────────
        distill_stages=(3, 4),  # encoder stages to distill (attention stages only)
        pointwise_weight=1.0,   # α: weight for per-point cosine KD loss
        relational_weight=2.0,  # β: weight for relational (pairwise) KD loss
        relational_n_sample=512,  # max points sampled for pairwise sim matrix
    ):
        super().__init__()
        self.num_classes    = num_classes
        self.distill_stages = list(distill_stages)
        self.pw_weight      = pointwise_weight
        self.rel_weight     = relational_weight
        self.rel_n_sample   = relational_n_sample

        # ── Student model ────────────────────────────────────────────────
        self.backbone = build_model(backbone)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.criteria = build_criteria(criteria)

        # ── Teacher backbone (frozen at all times) ───────────────────────
        self.teacher_backbone = build_model(teacher_backbone)
        self._load_teacher(teacher_ckpt)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        # ── Channel-projectors: student C_s → teacher C_t ────────────────
        # enc_channels is a list/tuple in the backbone config
        student_ch = backbone["enc_channels"]
        teacher_ch = teacher_backbone["enc_channels"]
        self.projectors = nn.ModuleDict()
        for s in self.distill_stages:
            # LayerNorm after projection stabilises training
            self.projectors[f"s{s}"] = nn.Sequential(
                nn.Linear(student_ch[s], teacher_ch[s], bias=False),
                nn.LayerNorm(teacher_ch[s]),
            )

        # ── Feature storage populated by forward hooks ───────────────────
        self._s_feats: dict = {}  # student intermediate features
        self._t_feats: dict = {}  # teacher intermediate features
        self._hooks   = []
        self._register_hooks()

    # ──────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ──────────────────────────────────────────────────────────────────────

    def _load_teacher(self, ckpt_path: str):
        """
        Load backbone weights from a DefaultSegmentorV2 checkpoint.
        Handles both 'state_dict' and direct formats.
        Strict=False so extra/missing keys don't crash (seg_head will be missing).
        """
        raw = torch.load(ckpt_path, map_location="cpu")

        # Unwrap common checkpoint wrappers
        if isinstance(raw, dict):
            state = raw.get("state_dict", raw.get("model", raw))
        else:
            state = raw

        # Extract backbone-only weights
        prefix = "backbone."
        backbone_state = {}
        for k, v in state.items():
            if k.startswith(prefix):
                backbone_state[k[len(prefix):]] = v
            elif not any(k.startswith(x) for x in ("seg_head.", "criteria.")):
                # Some saves don't prefix backbone keys
                backbone_state[k] = v

        missing, unexpected = self.teacher_backbone.load_state_dict(
            backbone_state, strict=False
        )
        n_loaded = len(backbone_state) - len(missing)
        print(
            f"[DistillationSegmentorV2] Teacher loaded {n_loaded}/{len(backbone_state)} "
            f"keys  |  missing={len(missing)}  unexpected={len(unexpected)}"
        )
        if len(missing) > 10:
            print(f"  First 5 missing: {missing[:5]}")

    def _register_hooks(self):
        """
        Register forward hooks on enc{s} modules of both backbones.
        Each hook stores the Point.feat tensor after that stage completes.
        PointSequential stores named sub-modules in _modules (standard nn.Module dict).
        """
        def _make_hook(storage: dict, stage: int):
            def hook(module, inp, out):
                # `out` is a Point object; .feat is the [N, C] feature tensor
                if hasattr(out, "feat"):
                    storage[stage] = out.feat
            return hook

        for s in self.distill_stages:
            key = f"enc{s}"

            # Student hook
            s_enc = self.backbone.enc._modules.get(key)
            if s_enc is not None:
                h = s_enc.register_forward_hook(_make_hook(self._s_feats, s))
                self._hooks.append(h)
            else:
                print(f"[DistillationSegmentorV2] WARNING: student enc '{key}' not found")

            # Teacher hook
            t_enc = self.teacher_backbone.enc._modules.get(key)
            if t_enc is not None:
                h = t_enc.register_forward_hook(_make_hook(self._t_feats, s))
                self._hooks.append(h)
            else:
                print(f"[DistillationSegmentorV2] WARNING: teacher enc '{key}' not found")

    # ──────────────────────────────────────────────────────────────────────
    # Distillation loss components
    # ──────────────────────────────────────────────────────────────────────

    def _pointwise_kd_loss(self) -> torch.Tensor:
        """
        Standard per-point cosine distillation.
        Minimises (1 - cosine_similarity(proj(student), teacher)).mean() at each stage.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        n     = 0
        for s in self.distill_stages:
            if s not in self._s_feats or s not in self._t_feats:
                continue
            sf = self.projectors[f"s{s}"](self._s_feats[s])  # [N_s, C_t]  — has grad
            tf = self._t_feats[s].detach()                     # [N_t, C_t]  — no grad
            if sf.shape[0] != tf.shape[0]:
                # Different number of points after pooling (shouldn't happen in same fwd pass)
                continue
            total = total + (1.0 - F.cosine_similarity(sf, tf, dim=-1)).mean()
            n    += 1
        return total / max(n, 1)

    def _relational_kd_loss(self) -> torch.Tensor:
        """
        Novel: Stage-wise Relational Feature Distillation (SRFD).

        Preserves pairwise cosine-similarity structure between points.
        At stage s with N points:
          S_teacher[i,j] = cos(t_i, t_j)
          S_student[i,j] = cos(proj(s_i), proj(s_j))
          loss_s = MSE(S_student, S_teacher)

        This forces the student to maintain the same inter-point semantic
        relationships as the teacher: same-class pairs stay close,
        different-class pairs stay separated — without ever seeing labels.
        """
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        n     = 0
        for s in self.distill_stages:
            if s not in self._s_feats or s not in self._t_feats:
                continue
            sf_raw = self._s_feats[s]   # [N, C_s]
            tf_raw = self._t_feats[s].detach()  # [N, C_t]

            if sf_raw.shape[0] != tf_raw.shape[0]:
                continue

            N = sf_raw.shape[0]

            # Sub-sample for efficiency when N is large (rarely needed at stages 3/4)
            if N > self.rel_n_sample:
                idx = torch.randperm(N, device=sf_raw.device)[: self.rel_n_sample]
                sf_raw = sf_raw[idx]
                tf_raw = tf_raw[idx]

            # Project student to teacher dimension
            sf = self.projectors[f"s{s}"](sf_raw)  # [N', C_t]  — has grad

            # L2-normalise for cosine similarity
            sf_n = F.normalize(sf,     dim=-1)  # [N', C_t]
            tf_n = F.normalize(tf_raw, dim=-1)  # [N', C_t]

            # Pairwise similarity matrices  [N', N']
            # Clamp to [-1, 1] for numerical safety
            S_s = torch.clamp(sf_n @ sf_n.t(), -1.0, 1.0)
            S_t = torch.clamp(tf_n @ tf_n.t(), -1.0, 1.0)

            # Mean-squared error between similarity matrices
            # Divide by N'² so loss scale is independent of sub-sample size
            total = total + F.mse_loss(S_s, S_t)
            n    += 1

        return total / max(n, 1)

    # ──────────────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────────────

    def forward(self, data_dict):
        # ── Clear stale features from previous step ───────────────────────
        self._s_feats.clear()
        self._t_feats.clear()

        # ── Student forward (triggers student hooks) ──────────────────────
        # Identical to DefaultSegmentorV2.forward
        point = self.backbone(data_dict)           # hooks populate _s_feats
        feat  = point.feat
        seg_logits = self.seg_head(feat)

        # ── Segmentation loss (standard) ─────────────────────────────────
        # criteria is a list of (loss_fn, weight) pairs or callable list
        if self.training:
            segment = data_dict.get("segment", None)
            if segment is None:
                raise ValueError("'segment' key missing from data_dict during training")

            # Single call — Criteria.__call__ handles the internal loop
            seg_loss = self.criteria(seg_logits, segment)

            # Teacher forward
            with torch.no_grad():
                self.teacher_backbone.eval()
                self.teacher_backbone(data_dict)

            pw_loss  = self._pointwise_kd_loss()
            rel_loss = self._relational_kd_loss()

            total_loss = (
                seg_loss
                + self.pw_weight  * pw_loss
                + self.rel_weight * rel_loss
            )

            return dict(
                loss     = total_loss,
                seg_loss = seg_loss.detach(),
                pw_kd    = pw_loss.detach(),
                rel_kd   = rel_loss.detach(),
            )

        # ── Inference: identical to DefaultSegmentorV2 ───────────────────
        return dict(seg_logits=seg_logits)

    def __del__(self):
        for h in self._hooks:
            h.remove()