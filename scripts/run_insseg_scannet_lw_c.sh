#!/bin/bash
# =============================================================================
# run_insseg_lw_c.sh
# =============================================================================
# Step 4: Plain lw-c instance segmentation (no KD).
# Backbone warm-started from the semseg-litept-lw-c-kd-2 checkpoint.
# Task heads (bias_head, seg_head) are randomly initialised — correct.
#
# Usage:
#   bash scripts/run_insseg_lw_c.sh
#   bash scripts/run_insseg_lw_c.sh -g 1          # explicit GPU count
#   bash scripts/run_insseg_lw_c.sh -g 2          # multi-GPU
#
# Output: exp/scannet/insseg-litept-lw-c/
# =============================================================================

set -euo pipefail
cd "$(dirname "$(dirname "$0")")" || exit 1

# ── Config ────────────────────────────────────────────────────────────────────
DATASET=scannet
CONFIG=insseg-litept-lw-c
EXP_NAME=insseg-litept-lw-c
SEMSEG_CKPT=exp/scannet/semseg-litept-lw-c-kd-2/model/model_best.pth
NUM_GPU=1

# ── Parse optional overrides ──────────────────────────────────────────────────
while getopts "g:" opt; do
    case $opt in
        g) NUM_GPU=$OPTARG ;;
        \?) echo "Usage: bash scripts/run_insseg_lw_c.sh [-g NUM_GPU]"; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
CONFIG_PATH=configs/${DATASET}/${CONFIG}.py
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config not found: $CONFIG_PATH"
    echo "  Make sure you copied insseg-litept-lw-c.py into configs/scannet/"
    exit 1
fi

if [ ! -f "$SEMSEG_CKPT" ]; then
    echo "ERROR: Semseg KD checkpoint not found: $SEMSEG_CKPT"
    echo "  Complete semseg-litept-lw-c-kd-2 training first, or update SEMSEG_CKPT."
    exit 1
fi

# ── Info ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Step 4: insseg-litept-lw-c  (plain, no KD)"
echo "  Config   : $CONFIG_PATH"
echo "  Backbone : $SEMSEG_CKPT"
echo "  GPUs     : $NUM_GPU"
echo "  Output   : exp/${DATASET}/${EXP_NAME}/"
echo "  Time     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
bash scripts/train.sh \
    -d "$DATASET"   \
    -c "$CONFIG"    \
    -n "$EXP_NAME"  \
    -w "$SEMSEG_CKPT" \
    -g "$NUM_GPU"

echo ""
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Checkpoint: exp/${DATASET}/${EXP_NAME}/model/model_best.pth"