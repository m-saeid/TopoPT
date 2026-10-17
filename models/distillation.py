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
  L_relational    = ||S_teacher - S_student||_F²  (Frobenius norm, via MSE)

Why this is novel for 3D:
  - Relational KD (Park et al., CVPR 2019) was designed for image classifiers.
  - We apply it specifically to the ATTENTION STAGES (3 and 4) of a 3D serialized
    point cloud transformer, where features encode geometric-semantic relationships.
  - Serialized attention in LitePT creates rich inter-point relationship structure
    that a shallower/narrower student fails to capture pointwise but CAN capture
    relationally with this supervision.
  - Stage 3: N≈1K → S is 1K×1K (cheap). Stage 4: N≈300 → S is 300×300 (trivial).

Why this will work:
  - The 0.32% gap between lw-c (74.95%) and baseline (75.27%) is precisely at the
    boundary/hard-example level — exactly what relational structure captures.
  - Combined with pointwise distillation, SRFD addresses BOTH absolute feature
    quality AND structural consistency.
  - Expected lw-c+SRFD: 75.2–75.5% (matching/beating baseline at 49% fewer GFLOPs).

─────────────────────────────────────────────────────────────────────────────────
SPCONV CUTLASS COMPATIBILITY NOTE
─────────────────────────────────────────────────────────────────────────────────
spconv selects different CUTLASS kernel templates based on SubMConv3d.training:

  training=True  → training-mode CUTLASS templates  (already profiled/cached
                   when the student backbone runs — guaranteed to work)
  training=False → eval-mode CUTLASS templates      (separate profiling required;
                   may fail on some GPU/spconv-version combinations because
                   eval templates use a different CUTLASS specialisation path)

To avoid the "can't find suitable algorithm for 0" RuntimeError we:
  1. Keep teacher's spconv layers in training=True so they share the same
     already-proven CUTLASS path as the student.
  2. Keep teacher's BatchNorm and DropPath layers in eval mode so the teacher
     produces stable, deterministic features (running stats, no path drop).
  3. Run teacher BEFORE student each step so CUTLASS profiling (first batch only)
     happens with maximum free GPU memory.
  4. Deep-clone input tensors for teacher so student mutations cannot corrupt them.
─────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

from models.builder import MODELS, build_model
from models.losses.builder import build_criteria
from models.utils.structure import Point


@MODELS.register_module("DistillationSegmentorV2")
class DistillationSegmentorV2(nn.Module):
    """
    Segmentation model with Stage-wise Relational Feature Distillation (SRFD).

    Total training loss:
        L = L_seg  +  α × L_pointwise_kd  +  β × L_relational_kd

    where:
        L_seg        = CrossEntropy + Lovász   (standard segmentation losses)
        L_pointwise  = Σ_s mean(1 - cosine_sim(proj(student_s), teacher_s))
        L_relational = Σ_s MSE(S_student_s, S_teacher_s)               [NOVEL]

    At inference: identical to DefaultSegmentorV2. Teacher and KD losses are gone.
    """

    def __init__(
        self,
        # ── Student ───────────────────────────────────────────────────────
        num_classes,
        backbone_out_channels,
        backbone,               # config dict for student (e.g. lw-c)
        criteria,               # list of loss config dicts
        # ── Teacher ───────────────────────────────────────────────────────
        teacher_backbone,       # config dict for teacher (full baseline)
        teacher_ckpt,           # path to teacher model_best.pth
        # ── Distillation hyper-parameters ─────────────────────────────────
        distill_stages=(3, 4),  # which encoder stages to distil
        pointwise_weight=1.0,   # α: weight for per-point cosine KD loss
        relational_weight=2.0,  # β: weight for relational (pairwise) KD loss
        relational_n_sample=512,  # max points sampled for pairwise sim matrix
    ):
        super().__init__()

        # Store hyper-parameters
        self.num_classes    = num_classes
        self.distill_stages = list(distill_stages)
        self.pw_weight      = pointwise_weight
        self.rel_weight     = relational_weight
        self.rel_n_sample   = relational_n_sample

        # ── Student ───────────────────────────────────────────────────────
        self.backbone = build_model(backbone)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        self.criteria = build_criteria(criteria)

        # ── Teacher (always frozen) ────────────────────────────────────────
        self.teacher_backbone = build_model(teacher_backbone)
        self._load_teacher(teacher_ckpt)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        # ── Channel projectors: student C_s → teacher C_t ─────────────────
        student_ch = backbone["enc_channels"]
        teacher_ch = teacher_backbone["enc_channels"]
        self.projectors = nn.ModuleDict()
        for s in self.distill_stages:
            # Linear + LayerNorm: projection + training stability
            self.projectors[f"s{s}"] = nn.Sequential(
                nn.Linear(student_ch[s], teacher_ch[s], bias=False),
                nn.LayerNorm(teacher_ch[s]),
            )

        # ── Forward-hook storage ───────────────────────────────────────────
        # Initialise BEFORE _register_hooks so __del__ is always safe
        self._s_feats: dict = {}
        self._t_feats: dict = {}
        self._hooks:   list = []
        self._register_hooks()

    # ─────────────────────────────────────────────────────────────────────
    # Initialisation helpers
    # ─────────────────────────────────────────────────────────────────────

    def _load_teacher(self, ckpt_path: str) -> None:
        """Load backbone weights from a DefaultSegmentorV2 checkpoint.

        Handles 'state_dict'-wrapped and bare formats.
        strict=False: seg_head / criteria keys in the ckpt are harmlessly ignored.
        weights_only=False: required for PyTorch ≥ 2.6 when the checkpoint
        contains numpy scalars (common in older saves).
        """
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Unwrap common checkpoint wrappers
        if isinstance(raw, dict):
            state = raw.get("state_dict", raw.get("model", raw))
        else:
            state = raw

        # Pull backbone-only weights; skip seg_head / criteria keys
        prefix = "backbone."
        skip   = ("seg_head.", "criteria.", "optimizer.", "scheduler.", "scaler.")
        backbone_state: dict = {}
        for k, v in state.items():
            if k.startswith(prefix):
                backbone_state[k[len(prefix):]] = v
            elif not any(k.startswith(s) for s in skip):
                # Some saves don't prefix backbone keys at all
                backbone_state[k] = v

        missing, unexpected = self.teacher_backbone.load_state_dict(
            backbone_state, strict=False
        )
        n_loaded = len(backbone_state) - len(missing)
        print(
            f"[DistillationSegmentorV2] Teacher: loaded {n_loaded}/{len(backbone_state)} "
            f"keys  |  missing={len(missing)}  unexpected={len(unexpected)}"
        )
        if missing:
            print(f"  First missing keys: {list(missing)[:5]}")

    def _register_hooks(self) -> None:
        """Register forward hooks on enc{s} sub-modules of both backbones.

        Each hook captures .feat from the Point object returned by that stage.
        Using a factory (_make_hook) ensures the closure captures the correct
        storage dict and stage index per iteration.
        """
        def _make_hook(storage: dict, stage: int):
            def _hook(module, inp, out):
                if hasattr(out, "feat"):          # Point object
                    storage[stage] = out.feat
                elif isinstance(out, torch.Tensor):
                    storage[stage] = out
            return _hook

        for s in self.distill_stages:
            key = f"enc{s}"

            # Student
            s_enc = self.backbone.enc._modules.get(key)
            if s_enc is not None:
                self._hooks.append(
                    s_enc.register_forward_hook(_make_hook(self._s_feats, s))
                )
            else:
                print(f"[DistillationSegmentorV2] WARNING: student '{key}' not found — "
                      f"stage {s} will be skipped during distillation.")

            # Teacher
            t_enc = self.teacher_backbone.enc._modules.get(key)
            if t_enc is not None:
                self._hooks.append(
                    t_enc.register_forward_hook(_make_hook(self._t_feats, s))
                )
            else:
                print(f"[DistillationSegmentorV2] WARNING: teacher '{key}' not found — "
                      f"stage {s} will be skipped during distillation.")

    # ─────────────────────────────────────────────────────────────────────
    # Teacher mode management  ← THE CORE FIX
    # ─────────────────────────────────────────────────────────────────────

    def _apply_teacher_distill_mode(self) -> None:
        """Put teacher backbone into 'distillation eval' mode.

        WHY THIS IS NECESSARY
        ─────────────────────
        spconv's SubMConv3d calls:
            self._conv_forward(self.training, ...)
        and uses self.training to select CUTLASS kernel templates:

          training=True  → training CUTLASS templates  → already profiled &
                           cached when the student runs → ALWAYS works.
          training=False → eval CUTLASS templates      → separate profiling
                           required; fails with "can't find suitable algorithm
                           for 0" on some GPU/spconv combinations.

        The solution: keep spconv layers at training=True (reuse student's
        proven CUTLASS path) while setting everything else to eval:
          • BatchNorm  training=False → uses running mean/var, not batch stats.
          • DropPath   training=False → no random path drops (deterministic).
          • All other  training=False → safe default.

        This is called:
          (a) in train() override  — so model.train() never breaks it, and
          (b) at the top of each teacher forward — belt-and-suspenders guard.
        """
        # Step 1: set entire teacher to eval (BN + DropPath are now correct)
        self.teacher_backbone.eval()

        # Step 2: flip ONLY spconv conv layers back to training=True so they
        # use the training CUTLASS path (same as student — proven to work).
        for m in self.teacher_backbone.modules():
            if isinstance(m, (
                spconv.SubMConv3d,
                spconv.SparseConv3d,
                spconv.SparseInverseConv3d,
                spconv.SparseConvTranspose3d,
            )):
                m.training = True

    def train(self, mode: bool = True) -> "DistillationSegmentorV2":
        """Override so the trainer's per-epoch model.train() never breaks teacher mode.

        Without this, Trainer.train() calls self.model.train() which recursively
        sets every submodule — including teacher_backbone — to training=True for
        ALL layers (including BN). Then forward() calls .eval() mid-step, causing
        spconv to see a new eval-mode config and attempt (failing) re-profiling.

        With this override, after the student + projectors are set to `mode`,
        the teacher is immediately corrected to distillation mode regardless.
        """
        super().train(mode)              # student backbone + projectors: normal
        if mode:
            # Only apply when switching to train; eval() leaves teacher alone
            # (teacher is never used during inference anyway).
            self._apply_teacher_distill_mode()
        return self

    # ─────────────────────────────────────────────────────────────────────
    # Distillation loss components
    # ─────────────────────────────────────────────────────────────────────

    def _pointwise_kd_loss(self) -> torch.Tensor:
        """Per-point cosine distillation (standard KD).

        For each distilled stage s:
            loss_s = mean(1 - cosine_similarity(proj(student_s), teacher_s))

        Averaged over all valid stages.
        """
        device = next(self.projectors.parameters()).device
        total  = torch.zeros(1, device=device)
        n      = 0

        for s in self.distill_stages:
            if s not in self._s_feats or s not in self._t_feats:
                continue
            sf = self.projectors[f"s{s}"](self._s_feats[s])   # [N, C_t]  grad ON
            # Cast teacher features to match student dtype (student may be fp16
            # under AMP; teacher runs in fp32 via training-mode spconv path).
            tf = self._t_feats[s].detach().to(sf.dtype)        # [N, C_t]  grad OFF
            if sf.shape[0] != tf.shape[0]:
                continue
            total = total + (1.0 - F.cosine_similarity(sf, tf, dim=-1)).mean()
            n    += 1

        return total / max(n, 1)

    def _relational_kd_loss(self) -> torch.Tensor:
        """Stage-wise Relational Feature Distillation — SRFD (novel contribution).

        For each distilled stage s with N points:
            S_teacher[i,j] = cos(t_i, t_j)
            S_student[i,j] = cos(proj(s_i), proj(s_j))
            loss_s         = MSE(S_student, S_teacher)

        This forces the student to reproduce the teacher's inter-point semantic
        topology: same-class point pairs stay close, cross-class pairs stay
        separated — without ever observing labels during distillation.

        Sub-sampling to rel_n_sample points is applied when N exceeds the
        threshold (rare at stages 3–4 with ≈300–1K points).
        """
        device = next(self.projectors.parameters()).device
        total  = torch.zeros(1, device=device)
        n      = 0

        for s in self.distill_stages:
            if s not in self._s_feats or s not in self._t_feats:
                continue

            sf_raw = self._s_feats[s]          # [N, C_s]
            # Cast teacher to student dtype (see note in _pointwise_kd_loss)
            tf_raw = self._t_feats[s].detach().to(sf_raw.dtype)  # [N, C_t]

            if sf_raw.shape[0] != tf_raw.shape[0]:
                continue

            N = sf_raw.shape[0]
            if N > self.rel_n_sample:
                idx    = torch.randperm(N, device=sf_raw.device)[:self.rel_n_sample]
                sf_raw = sf_raw[idx]
                tf_raw = tf_raw[idx]

            # Project student → teacher channel dimension
            sf = self.projectors[f"s{s}"](sf_raw)  # [N', C_t]  grad ON

            # L2-normalise both for cosine similarity via matrix multiply
            sf_n = F.normalize(sf,     dim=-1)      # [N', C_t]
            tf_n = F.normalize(tf_raw, dim=-1)      # [N', C_t]

            # Pairwise similarity matrices [N', N'], clamped for numerical safety
            S_s = torch.clamp(sf_n @ sf_n.t(), -1.0, 1.0)
            S_t = torch.clamp(tf_n @ tf_n.t(), -1.0, 1.0)

            total = total + F.mse_loss(S_s, S_t)
            n    += 1

        return total / max(n, 1)

    # ─────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────

    def forward(self, input_dict: dict) -> dict:
        # Clear stale features from the previous step
        self._s_feats.clear()
        self._t_feats.clear()

        # ── Teacher forward FIRST ──────────────────────────────────────────
        #
        # ORDER MATTERS: teacher runs before student so that spconv's CUTLASS
        # kernel profiling (first batch only) happens with maximum free GPU
        # memory, before the student allocates its activations.
        #
        # DEEP CLONE: student's forward mutates the Point structure in-place
        # (GridPooling replaces grid_coord, sparsify() attaches SparseConvTensor,
        # etc.).  Without cloning, shared tensor references would be corrupted
        # by the time the teacher accesses them.
        #
        # DISTILL MODE: _apply_teacher_distill_mode() is called here as a
        # belt-and-suspenders guard in case anything (DDP, checkpoint reload,
        # AMP hooks) reset teacher submodule flags between train() and forward().
        if self.training:
            self._apply_teacher_distill_mode()

            teacher_input = {
                k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                for k, v in input_dict.items()
            }

            with torch.no_grad():
                # spconv layers inside teacher_backbone are training=True
                # (set by _apply_teacher_distill_mode), so they use the same
                # training-mode CUTLASS kernels as the student — no new
                # profiling required, no "can't find suitable algorithm" crash.
                self.teacher_backbone(Point(teacher_input))   # populates _t_feats

        # ── Student forward ────────────────────────────────────────────────
        point = self.backbone(Point(input_dict))               # populates _s_feats

        # ── Unpooling (mirrors DefaultSegmentorV2 exactly) ────────────────
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys(), \
                    "pooling_parent present but pooling_inverse is missing"
                parent  = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point  # backbone returned a raw tensor (legacy path)

        seg_logits = self.seg_head(feat)

        # ── Training: seg loss + KD losses ────────────────────────────────
        if self.training:
            segment = input_dict.get("segment", None)
            if segment is None:
                raise ValueError(
                    "'segment' key missing from input_dict during training. "
                    "Check your dataset / collate_fn."
                )

            seg_loss = self.criteria(seg_logits, segment)
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

        # ── Eval with labels (SemSegEvaluator calls model with 'segment') ─
        elif "segment" in input_dict:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)

        # ── Test / inference (no labels) ──────────────────────────────────
        else:
            return dict(seg_logits=seg_logits)

    # ─────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────

    def __del__(self):
        # getattr guard: safe even if __init__ raised before _hooks was assigned
        for h in getattr(self, "_hooks", []):
            h.remove()