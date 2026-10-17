"""
models/distillation_pg.py
==========================
DistillationPointGroupV2 — Knowledge Distillation for efficient LitePT-lw-c
instance segmentation, based on PointGroup (PG-v1m2).

Reuses the full SRFD (Stage-wise Relational Feature Distillation) machinery
from DistillationSegmentorV2, adapted for the PG-v1m2 task head.

Training loss:
  L = L_seg                          (CE on semantic logits)
    + L_bias_L1                      (offset regression, L1)
    + L_bias_cosine                  (offset direction, cosine)
    + α × L_pointwise_kd             (per-point cosine KD, stages 3+4)
    + β × L_relational_kd   [SRFD]   (pairwise similarity, stages 3+4)

At inference: identical to PG-v1m2.  Teacher and KD losses are gone.

─────────────────────────────────────────────────────────────────────────────
DESIGN NOTES
─────────────────────────────────────────────────────────────────────────────
1. The teacher is the pretrained FULL LitePT backbone loaded from the
   insseg-litept-small-v1m2 checkpoint.  Only the encoder portion is used
   (enc_mode=True on the teacher backbone config) to save GPU memory.

2. Hooks are registered on backbone.enc.enc3 and backbone.enc.enc4 for both
   student and teacher, capturing Point.feat after each encoder stage.
   This is the same hook strategy as DistillationSegmentorV2.

3. The teacher backbone's spconv layers are kept in training=True to share
   the proven CUTLASS kernel path with the student (same fix as semseg KD).
   See _apply_teacher_distill_mode() for details.

4. The student backbone must have enc_mode=False (it has a real decoder,
   which PG-v1m2 needs for per-point features). The teacher backbone uses
   enc_mode=True (we only need encoder features for KD).

5. The bias_head, seg_head, and clustering logic are identical to PG-v1m2.
   This class is purely additive — it wraps PG-v1m2 behaviour and injects
   KD losses during training.
─────────────────────────────────────────────────────────────────────────────
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv

from models.builder import MODELS, build_model
from models.losses import build_criteria
from models.utils import offset2batch, batch2offset
from models.utils.structure import Point

try:
    from pointgroup_ops import ballquery_batch_p, bfs_cluster
except ImportError:
    ballquery_batch_p = bfs_cluster = None


@MODELS.register_module("DistillationPointGroupV2")
class DistillationPointGroupV2(nn.Module):
    """
    PG-v1m2 instance segmentation with Stage-wise Relational Feature
    Distillation (SRFD) applied at backbone encoder stages 3 and 4.

    Parameters
    ----------
    backbone : dict
        Config dict for the STUDENT backbone (lw-c LitePT, enc_mode=False).
    backbone_out_channels : int
        Output channels of the student backbone decoder (= enc_channels[1]
        for lw-c = 54).  Used to build bias_head and seg_head.
    teacher_backbone : dict
        Config dict for the TEACHER backbone (full LitePT, enc_mode=True).
    teacher_ckpt : str
        Path to a trained PG-v1m2 checkpoint (insseg-litept-small-v1m2).
        Only backbone weights are extracted; task head keys are ignored.
    distill_stages : tuple[int]
        Encoder stages to distil from (default: (3, 4)).
    pointwise_weight : float (α)
        Weight for per-point cosine KD loss.
    relational_weight : float (β)
        Weight for SRFD pairwise similarity loss.
    relational_n_sample : int
        Max points sampled for the N×N pairwise matrix per stage.
    semantic_num_classes, semantic_ignore_index, segment_ignore_index,
    instance_ignore_index, cluster_thresh, cluster_closed_points,
    cluster_propose_points, cluster_min_points, voxel_size, criteria :
        All forwarded to PG-v1m2 logic unchanged.
    """

    def __init__(
        self,
        # ── Student ───────────────────────────────────────────────────────
        backbone,
        # ── Teacher ───────────────────────────────────────────────────────
        teacher_backbone,
        teacher_ckpt,
        backbone_out_channels=54,
        # ── Distillation ──────────────────────────────────────────────────
        distill_stages=(3, 4),
        pointwise_weight=1.0,
        relational_weight=2.0,
        relational_n_sample=512,
        # ── Instance segmentation (PG-v1m2) ──────────────────────────────
        semantic_num_classes=20,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        cluster_thresh=1.5,
        cluster_closed_points=600,
        cluster_propose_points=200,
        cluster_min_points=50,
        voxel_size=0.02,
        criteria=None,
    ):
        super().__init__()

        # ── Store config ───────────────────────────────────────────────────
        self.semantic_num_classes   = semantic_num_classes
        self.semantic_ignore_index  = semantic_ignore_index
        self.segment_ignore_index   = segment_ignore_index
        self.instance_ignore_index  = instance_ignore_index
        self.cluster_thresh         = cluster_thresh
        self.cluster_closed_points  = cluster_closed_points
        self.cluster_propose_points = cluster_propose_points
        self.cluster_min_points     = cluster_min_points
        self.voxel_size             = voxel_size
        self.distill_stages         = list(distill_stages)
        self.pw_weight              = pointwise_weight
        self.rel_weight             = relational_weight
        self.rel_n_sample           = relational_n_sample

        # ── Student backbone ───────────────────────────────────────────────
        self.backbone = build_model(backbone)

        # ── Task heads (identical to PG-v1m2) ─────────────────────────────
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.bias_head = nn.Sequential(
            nn.Linear(backbone_out_channels, backbone_out_channels),
            norm_fn(backbone_out_channels),
            nn.ReLU(),
            nn.Linear(backbone_out_channels, 3),
        )
        self.seg_head = nn.Linear(backbone_out_channels, semantic_num_classes)
        self.seg_criteria = build_criteria(criteria)

        # ── Teacher backbone (frozen, enc_mode=True) ───────────────────────
        self.teacher_backbone = build_model(teacher_backbone)
        self._load_teacher(teacher_ckpt)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        # ── Channel projectors: student C_s → teacher C_t ─────────────────
        student_ch = backbone["enc_channels"]
        teacher_ch = teacher_backbone["enc_channels"]
        self.projectors = nn.ModuleDict()
        for s in self.distill_stages:
            self.projectors[f"s{s}"] = nn.Sequential(
                nn.Linear(student_ch[s], teacher_ch[s], bias=False),
                nn.LayerNorm(teacher_ch[s]),
            )

        # ── Hook storage (initialise BEFORE _register_hooks) ──────────────
        self._s_feats: dict = {}
        self._t_feats: dict = {}
        self._hooks:   list = []
        self._register_hooks()

    # ─────────────────────────────────────────────────────────────────────
    #  Initialisation helpers
    # ─────────────────────────────────────────────────────────────────────

    def _load_teacher(self, ckpt_path: str) -> None:
        """Load backbone weights from a PG-v1m2 (insseg) checkpoint.

        The checkpoint was saved by PointGroup whose state_dict has keys
        prefixed with 'backbone.'.  bias_head / seg_head / clustering
        parameters are silently ignored (strict=False).
        """
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(raw, dict):
            state = raw.get("state_dict", raw.get("model", raw))
        else:
            state = raw

        prefix = "backbone."
        skip   = ("bias_head.", "seg_head.", "seg_criteria.",
                  "optimizer.", "scheduler.", "scaler.")
        backbone_state: dict = {}
        for k, v in state.items():
            if k.startswith(prefix):
                backbone_state[k[len(prefix):]] = v
            elif not any(k.startswith(s) for s in skip):
                backbone_state[k] = v

        missing, unexpected = self.teacher_backbone.load_state_dict(
            backbone_state, strict=False
        )
        n_loaded = len(backbone_state) - len(missing)
        print(
            f"[DistillationPointGroupV2] Teacher loaded {n_loaded}/"
            f"{len(backbone_state)} keys  |  "
            f"missing={len(missing)}  unexpected={len(unexpected)}"
        )
        if missing:
            print(f"  First missing: {list(missing)[:5]}")

    def _register_hooks(self) -> None:
        """Register forward hooks on enc{s} sub-modules of both backbones."""

        def _make_hook(storage: dict, stage: int):
            def _hook(module, inp, out):
                if hasattr(out, "feat"):
                    storage[stage] = out.feat
                elif isinstance(out, torch.Tensor):
                    storage[stage] = out
            return _hook

        for s in self.distill_stages:
            key = f"enc{s}"

            s_enc = self.backbone.enc._modules.get(key)
            if s_enc is not None:
                self._hooks.append(
                    s_enc.register_forward_hook(_make_hook(self._s_feats, s))
                )
            else:
                print(f"[DistillationPointGroupV2] WARNING: student '{key}' "
                      f"not found — stage {s} skipped.")

            t_enc = self.teacher_backbone.enc._modules.get(key)
            if t_enc is not None:
                self._hooks.append(
                    t_enc.register_forward_hook(_make_hook(self._t_feats, s))
                )
            else:
                print(f"[DistillationPointGroupV2] WARNING: teacher '{key}' "
                      f"not found — stage {s} skipped.")

    # ─────────────────────────────────────────────────────────────────────
    #  Teacher mode management (identical logic to DistillationSegmentorV2)
    # ─────────────────────────────────────────────────────────────────────

    def _apply_teacher_distill_mode(self) -> None:
        """Keep spconv layers in training=True, everything else eval.

        Prevents "can't find suitable algorithm for 0" CUTLASS crash.
        See DistillationSegmentorV2 for full explanation.
        """
        self.teacher_backbone.eval()
        for m in self.teacher_backbone.modules():
            if isinstance(m, (
                spconv.SubMConv3d,
                spconv.SparseConv3d,
                spconv.SparseInverseConv3d,
                spconv.SparseConvTranspose3d,
            )):
                m.training = True

    def train(self, mode: bool = True) -> "DistillationPointGroupV2":
        """Override so model.train() never resets teacher mode."""
        super().train(mode)
        if mode:
            self._apply_teacher_distill_mode()
        return self

    # ─────────────────────────────────────────────────────────────────────
    #  Distillation loss components (identical to DistillationSegmentorV2)
    # ─────────────────────────────────────────────────────────────────────

    def _pointwise_kd_loss(self) -> torch.Tensor:
        device = next(self.projectors.parameters()).device
        total  = torch.zeros(1, device=device)
        n      = 0
        for s in self.distill_stages:
            if s not in self._s_feats or s not in self._t_feats:
                continue
            sf = self.projectors[f"s{s}"](self._s_feats[s])
            tf = self._t_feats[s].detach().to(sf.dtype)
            if sf.shape[0] != tf.shape[0]:
                continue
            total = total + (1.0 - F.cosine_similarity(sf, tf, dim=-1)).mean()
            n    += 1
        return total / max(n, 1)

    def _relational_kd_loss(self) -> torch.Tensor:
        device = next(self.projectors.parameters()).device
        total  = torch.zeros(1, device=device)
        n      = 0
        for s in self.distill_stages:
            if s not in self._s_feats or s not in self._t_feats:
                continue
            sf_raw = self._s_feats[s]
            tf_raw = self._t_feats[s].detach().to(sf_raw.dtype)
            if sf_raw.shape[0] != tf_raw.shape[0]:
                continue
            N = sf_raw.shape[0]
            if N > self.rel_n_sample:
                idx    = torch.randperm(N, device=sf_raw.device)[:self.rel_n_sample]
                sf_raw = sf_raw[idx]
                tf_raw = tf_raw[idx]
            sf   = self.projectors[f"s{s}"](sf_raw)
            sf_n = F.normalize(sf,     dim=-1)
            tf_n = F.normalize(tf_raw, dim=-1)
            S_s  = torch.clamp(sf_n @ sf_n.t(), -1.0, 1.0)
            S_t  = torch.clamp(tf_n @ tf_n.t(), -1.0, 1.0)
            total = total + F.mse_loss(S_s, S_t)
            n    += 1
        return total / max(n, 1)

    # ─────────────────────────────────────────────────────────────────────
    #  Forward (PG-v1m2 logic + KD losses during training)
    # ─────────────────────────────────────────────────────────────────────

    def forward(self, data_dict: dict) -> dict:
        self._s_feats.clear()
        self._t_feats.clear()

        coord             = data_dict["coord"]
        offset            = data_dict["offset"]
        segment           = data_dict.get("segment", None)
        instance          = data_dict.get("instance", None)
        instance_centroid = data_dict.get("instance_centroid", None)

        # ── Teacher forward FIRST (training only) ──────────────────────────
        if self.training:
            self._apply_teacher_distill_mode()
            teacher_input = {
                k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                for k, v in data_dict.items()
            }
            with torch.no_grad():
                self.teacher_backbone(Point(teacher_input))   # populates _t_feats

        # ── Student backbone ───────────────────────────────────────────────
        point = self.backbone(Point(data_dict))               # populates _s_feats

        # ── Unpool to original resolution (same as PG-v1m2) ───────────────
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent  = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat(
                    [parent.feat, point.feat[inverse]], dim=-1
                )
                point = parent
            feat = point.feat
        else:
            feat = point

        # ── Task head predictions ──────────────────────────────────────────
        bias_pred  = self.bias_head(feat)
        logit_pred = self.seg_head(feat)

        # ── Training loss ──────────────────────────────────────────────────
        if self.training:
            assert segment is not None and instance is not None \
                   and instance_centroid is not None, \
                "segment / instance / instance_centroid required during training."

            seg_loss = self.seg_criteria(logit_pred, segment)

            mask = (instance != self.instance_ignore_index).float()

            bias_gt   = instance_centroid - coord
            bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
            bias_l1_loss = (
                torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)
            )

            bias_pred_norm = bias_pred / (
                torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
            )
            bias_gt_norm = bias_gt / (
                torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8
            )
            cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
            bias_cosine_loss  = (
                torch.sum(cosine_similarity * mask) / (torch.sum(mask) + 1e-8)
            )

            # PG-v1m2 task loss
            task_loss = seg_loss + bias_l1_loss + bias_cosine_loss

            # SRFD KD losses
            pw_loss  = self._pointwise_kd_loss()
            rel_loss = self._relational_kd_loss()

            total_loss = (
                task_loss
                + self.pw_weight  * pw_loss
                + self.rel_weight * rel_loss
            )

            return dict(
                loss             = total_loss,
                seg_loss         = seg_loss.detach(),
                bias_l1_loss     = bias_l1_loss.detach(),
                bias_cosine_loss = bias_cosine_loss.detach(),
                pw_kd            = pw_loss.detach(),
                rel_kd           = rel_loss.detach(),
            )

        # ── Inference (identical to PG-v1m2) ──────────────────────────────
        # Compute task loss if labels available (e.g. val loop)
        return_dict = {}
        if segment is not None and instance is not None \
                and instance_centroid is not None:
            seg_loss = self.seg_criteria(logit_pred, segment)
            mask     = (instance != self.instance_ignore_index).float()
            bias_gt  = instance_centroid - coord
            bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
            bias_l1_loss = (
                torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)
            )
            bias_pred_norm = bias_pred / (
                torch.norm(bias_pred, p=2, dim=1, keepdim=True) + 1e-8
            )
            bias_gt_norm = bias_gt / (
                torch.norm(bias_gt, p=2, dim=1, keepdim=True) + 1e-8
            )
            cosine_similarity = -(bias_pred_norm * bias_gt_norm).sum(-1)
            bias_cosine_loss  = (
                torch.sum(cosine_similarity * mask) / (torch.sum(mask) + 1e-8)
            )
            return_dict["loss"] = (
                seg_loss + bias_l1_loss + bias_cosine_loss
            )
            return_dict["seg_loss"]         = seg_loss
            return_dict["bias_l1_loss"]     = bias_l1_loss
            return_dict["bias_cosine_loss"] = bias_cosine_loss

        # BFS clustering (eval only) — identical to PG-v1m2
        center_pred = coord + bias_pred
        center_pred /= self.voxel_size
        logit_pred   = F.softmax(logit_pred, dim=-1)
        segment_pred = torch.max(logit_pred, 1)[1]

        mask = (
            ~torch.concat(
                [(segment_pred == idx).unsqueeze(-1)
                 for idx in self.segment_ignore_index],
                dim=1,
            ).sum(-1).bool()
        )

        if mask.sum() == 0:
            proposals_idx    = torch.zeros(0).int()
            proposals_offset = torch.zeros(1).int()
        else:
            center_pred_  = center_pred[mask]
            segment_pred_ = segment_pred[mask]
            batch_        = offset2batch(offset)[mask]
            offset_       = nn.ConstantPad1d((1, 0), 0)(batch2offset(batch_))
            idx, start_len = ballquery_batch_p(
                center_pred_, batch_.int(), offset_.int(),
                self.cluster_thresh, self.cluster_closed_points,
            )
            proposals_idx, proposals_offset = bfs_cluster(
                segment_pred_.int().cpu(),
                idx.cpu(),
                start_len.cpu(),
                self.cluster_min_points,
            )
            proposals_idx[:, 1] = (
                mask.nonzero().view(-1)[proposals_idx[:, 1].long()].int()
            )

        proposals_pred = torch.zeros(
            (proposals_offset.shape[0] - 1, center_pred.shape[0]),
            dtype=torch.int,
        )
        proposals_pred[
            proposals_idx[:, 0].long(), proposals_idx[:, 1].long()
        ] = 1
        instance_pred = segment_pred[
            proposals_idx[:, 1][proposals_offset[:-1].long()].long()
        ]
        proposals_point_num = proposals_pred.sum(1)
        proposals_mask      = proposals_point_num > self.cluster_propose_points
        proposals_pred      = proposals_pred[proposals_mask]
        instance_pred       = instance_pred[proposals_mask]

        pred_scores  = []
        pred_classes = []
        pred_masks   = proposals_pred.detach().cpu()
        for pid in range(len(proposals_pred)):
            seg_    = proposals_pred[pid]
            conf_   = logit_pred[seg_.bool(), instance_pred[pid]].mean()
            pred_scores.append(conf_)
            pred_classes.append(instance_pred[pid])
        if pred_scores:
            pred_scores  = torch.stack(pred_scores).cpu()
            pred_classes = torch.stack(pred_classes).cpu()
        else:
            pred_scores  = torch.tensor([])
            pred_classes = torch.tensor([])

        return_dict["pred_scores"]  = pred_scores
        return_dict["pred_masks"]   = pred_masks
        return_dict["pred_classes"] = pred_classes
        return return_dict

    def __del__(self):
        for h in getattr(self, "_hooks", []):
            h.remove()