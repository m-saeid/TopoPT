#!/bin/bash
# =============================================================================
# run_insseg_lw_c_kd.sh
# =============================================================================
# Step 5: lw-c instance segmentation + SRFD Knowledge Distillation.
# Teacher (full LitePT insseg) is loaded INTERNALLY by DistillationPointGroupV2.
# Student starts from random weights (or optionally warm-started — see below).
#
# Usage:
#   bash scripts/run_insseg_lw_c_kd.sh                  # fresh student
#   bash scripts/run_insseg_lw_c_kd.sh -g 1             # explicit GPU count
#   bash scripts/run_insseg_lw_c_kd.sh -g 2             # multi-GPU
#   bash scripts/run_insseg_lw_c_kd.sh -w exp/scannet/insseg-litept-lw-c/model/model_best.pth
#                                                        # warm-start from step 4
#
# Output: exp/scannet/insseg-litept-lw-c-kd/
#
# PREREQUISITES:
#   1. configs/scannet/insseg-litept-lw-c-kd.py  is in place.
#   2. models/distillation_pg.py                  is in place.
#   3. models/__init__.py                         imports DistillationPointGroupV2.
#   4. Teacher checkpoint exists at the path set in _teacher_ckpt inside the
#      config (default: exp/scannet/insseg-litept-small-v1m2/model/model_best.pth).
# =============================================================================

set -euo pipefail
cd "$(dirname "$(dirname "$0")")" || exit 1

# ── Config ────────────────────────────────────────────────────────────────────
DATASET=scannet
CONFIG=insseg-litept-lw-c-kd
EXP_NAME=insseg-litept-lw-c-kd
TEACHER_CKPT=exp/scannet/insseg-litept-small-v1m2/model/model_best.pth
NUM_GPU=1
WARMSTART_WEIGHT="None"   # set via -w to warm-start student from step 4

# ── Parse optional overrides ──────────────────────────────────────────────────
while getopts "g:w:" opt; do
    case $opt in
        g) NUM_GPU=$OPTARG          ;;
        w) WARMSTART_WEIGHT=$OPTARG ;;
        \?) echo "Usage: bash scripts/run_insseg_lw_c_kd.sh [-g NUM_GPU] [-w WARMSTART_CKPT]"
            exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
CONFIG_PATH=configs/${DATASET}/${CONFIG}.py
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config not found: $CONFIG_PATH"
    echo "  Make sure you copied insseg-litept-lw-c-kd.py into configs/scannet/"
    exit 1
fi

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERROR: Teacher checkpoint not found: $TEACHER_CKPT"
    echo "  Download the pretrained insseg-litept-small-v1m2 weights from the"
    echo "  official LitePT repo and place at:"
    echo "    $TEACHER_CKPT"
    echo "  Or update _teacher_ckpt in configs/scannet/insseg-litept-lw-c-kd.py"
    exit 1
fi

if [ "$WARMSTART_WEIGHT" != "None" ] && [ ! -f "$WARMSTART_WEIGHT" ]; then
    echo "ERROR: Warm-start checkpoint not found: $WARMSTART_WEIGHT"
    exit 1
fi

# ── Info ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Step 5: insseg-litept-lw-c-kd  (SRFD distillation)"
echo "  Config   : $CONFIG_PATH"
echo "  Teacher  : $TEACHER_CKPT  (loaded internally)"
if [ "$WARMSTART_WEIGHT" != "None" ]; then
echo "  Student  : warm-started from $WARMSTART_WEIGHT"
else
echo "  Student  : random init"
fi
echo "  GPUs     : $NUM_GPU"
echo "  Output   : exp/${DATASET}/${EXP_NAME}/"
echo "  Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
# Teacher checkpoint is handled internally — no -w flag needed for that.
# -w here only warm-starts the STUDENT backbone (optional).
if [ "$WARMSTART_WEIGHT" = "None" ]; then
    bash scripts/train.sh \
        -d "$DATASET"   \
        -c "$CONFIG"    \
        -n "$EXP_NAME"  \
        -g "$NUM_GPU"
else
    bash scripts/train.sh \
        -d "$DATASET"   \
        -c "$CONFIG"    \
        -n "$EXP_NAME"  \
        -w "$WARMSTART_WEIGHT" \
        -g "$NUM_GPU"
fi

echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: exp/${DATASET}/${EXP_NAME}/model/model_best.pth"