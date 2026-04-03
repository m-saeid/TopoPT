# models\default.py

import torch
import torch.nn as nn
import torch_scatter
import torch_cluster

from models.losses import build_criteria
from models.utils.structure import Point
from models.utils import offset2batch
from .builder import MODELS, build_model

from models.modules import PointModel, PointSequential
import spconv.pytorch as spconv

import torch.distributed as dist
from tqdm import tqdm
import pointops

import torch.nn.functional as F

@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)

'''
@MODELS.register_module()
class DistillationSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,              # student (lightweight) backbone config
        teacher_backbone=None,      # teacher (full) backbone config; falls back to backbone if None
        criteria=None,
        teacher_ckpt=None,
        distill_stages=(3, 4),      # which stage indices to distill
        stage_channels=None,        # {stage_idx: (student_ch, teacher_ch)}; see defaults below
        student_hook_paths=None,    # dotted submodule paths, e.g. ["enc_stages.2", "enc_stages.3"]
        teacher_hook_paths=None,    # same, for teacher backbone
        alpha=0.5,
        freeze_backbone=False,
        beta=2.0,
        rel_n_sample=512
    ):
        super().__init__()
        self.alpha = alpha
        self.distill_stages = [int(s) for s in distill_stages]
        self.beta         = beta
        self.rel_n_sample = rel_n_sample


        # ── Student ───────────────────────────────────────────────────────────
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ── Teacher (always frozen) ───────────────────────────────────────────
        teacher_backbone_cfg = teacher_backbone if teacher_backbone is not None else backbone
        self.teacher_backbone = build_model(teacher_backbone_cfg)
        self._load_teacher_checkpoint(teacher_ckpt)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        # ── Channel-matching projectors (student_ch → teacher_ch) ────────────
        # Defaults match the sketch: stage3: 180→252, stage4: 360→504
        _default_channels = {3: (180, 252), 4: (360, 504)}
        if stage_channels is not None:
            _default_channels.update({int(k): v for k, v in stage_channels.items()})

        self.projectors = nn.ModuleDict({
            f"stage{s}": nn.Linear(_default_channels[s][0], _default_channels[s][1], bias=False)
            for s in self.distill_stages
            if s in _default_channels
        })

        # ── Forward hooks ─────────────────────────────────────────────────────
        # Default: backbone exposes enc_stages; stage N maps to enc_stages[N-1]
        if student_hook_paths is None:
            student_hook_paths = [f"enc_stages.{s - 1}" for s in self.distill_stages]
        if teacher_hook_paths is None:
            teacher_hook_paths = [f"enc_stages.{s - 1}" for s in self.distill_stages]

        self._student_feats: dict = {}
        self._teacher_feats: dict = {}
        self._hook_handles = []

        self._register_hooks(self.backbone,         student_hook_paths, self._student_feats)
        self._register_hooks(self.teacher_backbone, teacher_hook_paths, self._teacher_feats)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_teacher_checkpoint(self, ckpt_path):
        """Load teacher weights, handling full-model and backbone-only checkpoints."""
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)   # unwrap Lightning/MMEngine wrapper

        # Prefer backbone-scoped keys; fall back to loading the dict directly
        prefix = "backbone."
        backbone_sd = {k[len(prefix):]: v for k, v in state_dict.items()
                       if k.startswith(prefix)}
        target_sd = backbone_sd if backbone_sd else state_dict
        missing, unexpected = self.teacher_backbone.load_state_dict(target_sd, strict=False)
        if missing:
            print(f"[DistillationSegmentorV2] Teacher ckpt — missing keys: {missing}")
        if unexpected:
            print(f"[DistillationSegmentorV2] Teacher ckpt — unexpected keys: {unexpected}")

    def _register_hooks(self, model, paths, store):
        """Register forward hooks that extract .feat from Point or a raw Tensor."""
        for stage_idx, path in zip(self.distill_stages, paths):
            key = f"stage{stage_idx}"
            try:
                submodule = model.get_submodule(path)   # supports dotted paths
            except AttributeError:
                print(f"[DistillationSegmentorV2] Warning: submodule '{path}' not found — "
                      f"stage {stage_idx} will be skipped during distillation.")
                continue

            # Fix: use a default-argument closure to capture key per iteration
            def _make_hook(k):
                def _hook(module, inp, output):
                    if isinstance(output, Point):
                        store[k] = output.feat
                    elif isinstance(output, torch.Tensor):
                        store[k] = output
                    # spconv SparseConvTensor — extract dense features
                    elif hasattr(output, "features"):
                        store[k] = output.features
                return _hook

            handle = submodule.register_forward_hook(_make_hook(key))
            self._hook_handles.append(handle)

    def remove_hooks(self):
        """Call to clean up hooks when the model is no longer needed."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, input_dict, return_point=False):
        self._student_feats.clear()
        self._teacher_feats.clear()

        # Teacher pass — no gradients, no seg_head needed
        with torch.no_grad():
            self.teacher_backbone.eval()
            self.teacher_backbone(Point(input_dict))   # hooks populate _teacher_feats

        # Student pass
        point = Point(input_dict)
        point = self.backbone(point)                   # hooks populate _student_feats

        # Unpooling — identical to DefaultSegmentorV2
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent  = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = {}
        if return_point:
            return_dict["point"] = point

        # ── Training: segmentation loss + KD loss ────────────────────────────
        if self.training:
            seg_loss = self.criteria(seg_logits, input_dict["segment"])

            kd_loss  = seg_logits.new_zeros(())
            rel_loss = seg_logits.new_zeros(())

            for stage in self.distill_stages:
                key = f"stage{stage}"
                if key not in self._student_feats or key not in self._teacher_feats:
                    continue

                s_feat = self.projectors[key](self._student_feats[key])   # [N, C_t]  — has grad
                t_feat = self._teacher_feats[key].detach()                 # [N, C_t]  — no grad

                if s_feat.shape[0] != t_feat.shape[0]:
                    continue

                # ── Pointwise cosine KD (existing) ──────────────────────────────
                kd_loss = kd_loss + (
                    1.0 - F.cosine_similarity(s_feat, t_feat, dim=-1).mean()
                )

                # ── SRFD: pairwise relational KD (novel) ────────────────────────
                N = s_feat.shape[0]
                if N > self.rel_n_sample:
                    idx    = torch.randperm(N, device=s_feat.device)[:self.rel_n_sample]
                    s_feat = s_feat[idx]
                    t_feat = t_feat[idx]

                sf_n = F.normalize(s_feat, dim=-1)
                tf_n = F.normalize(t_feat, dim=-1)
                S_s  = torch.clamp(sf_n @ sf_n.t(), -1.0, 1.0)
                S_t  = torch.clamp(tf_n @ tf_n.t(), -1.0, 1.0)
                rel_loss = rel_loss + F.mse_loss(S_s, S_t)

            return_dict["loss"]     = seg_loss + self.alpha * kd_loss + self.beta * rel_loss
            return_dict["seg_loss"] = seg_loss.detach()
            return_dict["kd_loss"]  = kd_loss.detach()
            return_dict["rel_loss"] = rel_loss.detach()

        # ── Eval with labels ─────────────────────────────────────────────────
        elif "segment" in input_dict:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"]       = loss
            return_dict["seg_logits"] = seg_logits

        # ── Test (no labels) ─────────────────────────────────────────────────
        else:
            return_dict["seg_logits"] = seg_logits

        return return_dict
'''    

@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                # parent.feat = point.feat[inverse]
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict