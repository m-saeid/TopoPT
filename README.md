<div align="center">
<h1>Relational Feature Distillation for Lightweight 3D Point Cloud Segmentation</h1>

**NeurIPS 2026** *(anonymous submission)*

<a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-TopoPT-red" alt="Paper PDF"></a>
<a href="https://github.com/anon-push/TopoPT"><img src="https://img.shields.io/badge/GitHub-TopoPT-blue" alt="GitHub"></a>
<a href="https://huggingface.co/cogniperceptai/TopoPT"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" alt="HF Model"></a>

</div>

---

**TopoPT** is a lightweight 3D point cloud segmentation model obtained by compressing [LitePT-S](https://github.com/prs-eth/LitePT) with a compact student architecture (**TrimPT**) trained via **Stage-wise Relational Feature Distillation (SRFD)**.

- **TrimPT** compresses LitePT-S by reducing channel width and stage-3 attention depth while preserving the full 1024-token attention window. It achieves **5.84 M parameters** and **12.95 GFLOPs** — **2.18× fewer parameters** and **1.96× fewer FLOPs** than LitePT-S.
- **SRFD** is a training-only objective that matches pairwise cosine-similarity matrices between teacher (LitePT-S) and student (TrimPT) features at the compressed attention stages (stages 3–4). The teacher and projection heads are discarded after training, so **TopoPT has the same inference architecture, checkpoint size, FLOPs, and latency as TrimPT**.
- On **ScanNet semantic segmentation**, TopoPT reaches **76.6% mIoU** — matching the official LitePT-S result (76.5%) with 2.18× fewer parameters and 1.96× fewer FLOPs.

![TopoPT Teaser](https://github.com/anon-push/TopoPT/blob/main/image/teaser.jpg)

## News

- **2026-XX-XX:** TopoPT is here.

## Method Overview

TopoPT consists of two components:

### TrimPT: Compressed Student Architecture

TrimPT is derived from LitePT-S via a controlled compression study across three axes — encoder depth, channel width, and attention patch size. Key findings:

| Compression axis | Effect |
|---|---|
| Stage-3 depth (6 → 4 blocks) | Minor accuracy drop, meaningful FLOP reduction |
| Channel width (504 → 360 max) | Large compute saving, modest accuracy drop |
| Attention patch (1024 → 512) | Accuracy degradation with limited measured FLOP benefit |

TrimPT combines depth and width reduction while **preserving the 1024-token attention window**, yielding channels `(36, 54, 108, 180, 360)` and stage depths `(2, 2, 2, 4, 2)`.

![TrimPT Architecture](https://github.com/anon-push/TopoPT/blob/main/image/backbone.jpg)

### SRFD: Stage-wise Relational Feature Distillation

SRFD is applied at stages 3 and 4 (the compressed attention stages). At each stage, a learned projector maps student features to the teacher's dimension. The training loss combines:

- **Pointwise loss** (L_pw): cosine alignment of projected student features with frozen teacher features.
- **Relational loss** (L_rel): Frobenius distance between teacher and student pairwise cosine-similarity matrices, computed over a sampled set of up to 512 aligned tokens per stage.

The full training loss is:

```
L = L_seg + α·L_pw + β·L_rel    (α=1.0, β=2.0)
```

After training, the teacher and projectors are discarded. **TopoPT inference is identical to TrimPT inference.**

![SRFD Method](https://github.com/anon-push/TopoPT/blob/main/image/srfd.jpg)

## Preparation

### Environment

TopoPT builds on the [LitePT](https://github.com/prs-eth/LitePT) codebase. Set up the environment following the LitePT instructions:

```shell
git clone https://github.com/m-saeid/TopoPT.git
cd TopoPT
conda create -n topopt python=3.10
conda activate topopt
# Install PyTorch — adjust for your CUDA version
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
# Install other required packages
pip install -r requirements.txt
# spconv (SparseUNet)
pip install spconv-cu124
# Flash attention
pip install git+https://github.com/Dao-AILab/flash-attention.git
```

**PointROPE.** Modify `all_cuda_archs` in `libs/pointrope/setup.py` to match your GPU architecture (e.g., 8.6 for RTX 3090, 9.0 for H100):

```shell
cd libs/pointrope
python setup.py install
cd ../..
```

A pure PyTorch fallback is available if CUDA compilation is not possible (slightly slower).

**Optional requirements** (for evaluator and PointGroup instance segmentation):

```shell
# Evaluator
cd libs/pointops && python setup.py install && cd ../..
# PointGroup
conda install -c bioconda google-sparsehash
cd libs/pointgroup_ops && python setup.py install && cd ../..
```

### Data

Data preparation follows [Pointcept](https://github.com/Pointcept/Pointcept#data-preparation). All datasets should be placed in `TopoPT/data`.

### Teacher Checkpoint

SRFD requires a pretrained LitePT-S teacher. Download the appropriate checkpoint from the [LitePT model zoo](https://huggingface.co/prs-eth/LitePT) and place it as specified in the config file.

## Model Zoo

All TopoPT and TrimPT checkpoints are available on [Hugging Face](https://huggingface.co/cogniperceptai/TopoPT).

### Semantic Segmentation

| Model | Params | Benchmark | Epochs | val mIoU | Config | Checkpoint |
|:--|--:|:--:|--:|:--:|:--:|:--:|
| TrimPT | 5.8M | ScanNet | 100 | 74.9 | [link](configs/scannet/semseg-lw-c-100epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet-semseg-lw-c-100epoch/model) |
| TrimPT | 5.8M | ScanNet | 1200 | 75.6 | [link](configs/scannet/semseg-lw-c-1200epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet-semseg-lw-c-1200epoch/model) |
| **TopoPT** | **5.8M** | **ScanNet** | **100** | **76.6** | [link](configs/scannet/semseg-lw-c-kd-100epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet-semseg-lw-c-kd-100epoch/model) |
| TrimPT | 5.8M | NuScenes | 50 | 81.2 | [link](configs/nuscenes/semseg-lw-c-50epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/nuscenes-semseg-lw-c-50epoch/model) |
| **TopoPT** | **5.8M** | **NuScenes** | **50** | **81.4** | [link](configs/nuscenes/semseg-lw-c-kd-50epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/nuscenes-semseg-lw-c-kd-50epoch/model) |
| TrimPT | 5.8M | Structured3D | 50 | 69.4 | [link](configs/structured3d/semseg-lw-c-50epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/structured3d-semseg-lw-c-50epoch/model) |
| **TopoPT** | **5.8M** | **Structured3D** | **50** | **70.3** | [link](configs/structured3d/semseg-lw-c-kd-tl-50epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/structured3d-semseg-lw-c-kd-tl-50epoch/model) |

### Instance Segmentation

| Model | Params | Benchmark | Epochs | mAP₅₀ | Config | Checkpoint |
|:--|--:|:--:|--:|:--:|:--:|:--:|
| TrimPT | 5.8M | ScanNet | 100 | 63.1 | [link](configs/scannet/insseg-lw-c-100epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet-insseg-lw-c-100epoch/model) |
| TrimPT | 5.8M | ScanNet | 800 | 65.1 | [link](configs/scannet/insseg-lw-c-800epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet-insseg-lw-c-800epoch/model) |
| **TopoPT** | **5.8M** | **ScanNet** | **100** | **63.9** | [link](configs/scannet/insseg-lw-c-kd-100epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet-insseg-lw-c-kd-100epoch/model) |
| TrimPT | 5.8M | ScanNet200 | 100 | 26.3 | [link](configs/scannet200/insseg-lw-c-100epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet200-insseg-lw-c-100epoch/model) |
| TrimPT | 5.8M | ScanNet200 | 800 | 31.7 | [link](configs/scannet200/insseg-lw-c-800epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet200-insseg-lw-c-800epoch/model) |
| **TopoPT** | **5.8M** | **ScanNet200** | **100** | **33.0** | [link](configs/scannet200/insseg-lw-c-kd-100epoch.py) | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet200-insseg-lw-c-kd-100epoch/model) |

### Ablation Checkpoints

The following checkpoints correspond to the compression ablation study in the paper (Table 2). All are trained for 100 epochs on ScanNet semantic segmentation.

| Model | Description | Params | mIoU | Checkpoint |
|:--|:--|--:|:--:|:--:|
| LitePT-S (repro) | Official arch, reproduced | 12.7M | 75.3 | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet-semseg-litept-reruun-100epoch/model) |
| LW-A | Depth reduction only | 11.2M | 75.4 | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet-semseg-lw-a-100epoch/model) |
| LW-B | Width reduction only | 6.6M | 74.9 | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet-semseg-lw-b-100epoch/model) |
| LW-C (= TrimPT) | Depth + width reduction | 5.8M | 74.9 | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet-semseg-lw-c-100epoch/model) |
| LW-D | Patch size reduction only | 12.7M | 74.2 | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet-semseg-lw-d-100epoch/model) |
| LW-E | Depth + width + patch | 5.8M | 74.1 | [Download](https://huggingface.co/cogniperceptai/TopoPT/tree/main/scannet-semseg-lw-e-100epoch/model) |

![Ablation: Accuracy-Efficiency](https://github.com/anon-push/TopoPT/blob/main/image/ablation.jpg)

## Training

### Semantic Segmentation

```shell
# ScanNet + TrimPT (no distillation)
sh scripts/train.sh -g 4 -d scannet -c semseg-lw-c-100epoch -n semseg-lw-c-100epoch

# ScanNet + TopoPT (SRFD)
sh scripts/train.sh -g 4 -d scannet -c semseg-lw-c-kd-100epoch -n semseg-lw-c-kd-100epoch

# NuScenes + TrimPT
sh scripts/train.sh -g 4 -d nuscenes -c semseg-lw-c-50epoch -n semseg-lw-c-50epoch

# NuScenes + TopoPT
sh scripts/train.sh -g 4 -d nuscenes -c semseg-lw-c-kd-50epoch -n semseg-lw-c-kd-50epoch

# Structured3D + TrimPT
sh scripts/train.sh -g 16 -d structured3d -c semseg-lw-c-50epoch -n semseg-lw-c-50epoch

# Structured3D + TopoPT
sh scripts/train.sh -g 16 -d structured3d -c semseg-lw-c-kd-tl-50epoch -n semseg-lw-c-kd-tl-50epoch
```

### Instance Segmentation

```shell
# ScanNet + TrimPT
sh scripts/train.sh -g 4 -d scannet -c insseg-lw-c-100epoch -n insseg-lw-c-100epoch

# ScanNet + TopoPT
sh scripts/train.sh -g 4 -d scannet -c insseg-lw-c-kd-100epoch -n insseg-lw-c-kd-100epoch

# ScanNet200 + TrimPT
sh scripts/train.sh -g 4 -d scannet200 -c insseg-lw-c-100epoch -n insseg-lw-c-100epoch

# ScanNet200 + TopoPT
sh scripts/train.sh -g 4 -d scannet200 -c insseg-lw-c-kd-100epoch -n insseg-lw-c-kd-100epoch
```

## Testing

The training pipeline automatically runs evaluation upon completion. For standalone testing with a pretrained checkpoint:

```shell
export PYTHONPATH=./
python tools/test.py \
    --config-file "${CONFIG_PATH}" \
    --num-gpus "${NUM_GPU}" \
    --options save_path="${SAVE_PATH}" weight="${CHECKPOINT_PATH}"

# E.g., ScanNet semantic segmentation with TopoPT:
# python tools/test.py \
#     --config-file configs/scannet/semseg-lw-c-kd-100epoch.py \
#     --num-gpus 4 \
#     --options save_path=exp/scannet/semseg-lw-c-kd-100epoch \
#               weight=exp/scannet/semseg-lw-c-kd-100epoch/model/model_best.pth
```

## Efficiency Summary

Inference profiled on NVIDIA RTX 3090 (batch size 1, 300 forward passes after 10 warm-up iterations). TopoPT and TrimPT share the same deployed architecture.

| Dataset / Task | Model | Params (M) | GFLOPs | Latency (ms) | FPS | Mem (GB) | Size (MB) |
|:--|:--|--:|--:|--:|--:|--:|--:|
| ScanNet Sem. | LitePT-S | 12.71 | 25.42 | 34.08 | 29.34 | 1.332 | 145.8 |
| ScanNet Sem. | TrimPT | 5.84 | 12.95 | 30.75 | 32.52 | 1.211 | 67.1 |
| ScanNet Sem. | **TopoPT** | **5.84** | **12.95** | **30.78** | **32.49** | **1.211** | **67.1** |
| nuScenes Sem. | LitePT-S | 12.71 | 25.42 | 35.81 | 27.93 | 0.717 | 145.7 |
| nuScenes Sem. | TrimPT | 5.84 | 12.95 | 28.72 | 34.82 | 0.432 | 67.0 |
| nuScenes Sem. | **TopoPT** | **5.84** | **12.95** | **28.84** | **34.68** | **0.432** | **67.0** |

## Acknowledgments

TopoPT builds directly on [LitePT](https://github.com/prs-eth/LitePT) and its dependencies. We thank the authors of [Pointcept](https://github.com/Pointcept/Pointcept), [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3), and [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{topopt2026,
    title={{Relational Feature Distillation for Lightweight 3D Point Cloud Segmentation}},
    author={Anonymous},
    booktitle={...},
    year={2026}
}
```

Please also cite LitePT, on which TopoPT is built:

```bibtex
@inproceedings{yuelitept2026,
    title={{LitePT: Lighter Yet Stronger Point Transformer}},
    author={Yue, Yuanwen and Robert, Damien and Wang, Jianyuan and Hong, Sunghwan and Wegner, Jan Dirk and Rupprecht, Christian and Schindler, Konrad},
    booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2026}
}
```
