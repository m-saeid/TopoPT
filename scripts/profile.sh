#!/usr/bin/env bash
# =============================================================================
# scripts/profile.sh
# =============================================================================
# Wrapper around tools/profile_model.py.
# Logs params, GFLOPs, GPU memory, inference latency to WandB project
# "LitePT-Profiling" (separate from the training project).
#
# Run BEFORE training (random/untrained weights — architecture cost):
#   bash scriptss/profile.sh -c semseg-litept-lw-e -n litept-lw-e-random
#
# Run AFTER training (loaded checkpoint — real-world cost):
#   bash scriptss/profile.sh \
#       -c semseg-litept-lw-e \
#       -w exp/scannet/semseg-litept-lw-e/model/model_best.pth \
#       -n litept-lw-e-trained
#
# Options:
#   -c  config name (under configs/DATASET/, WITHOUT .py)     [required]
#   -n  WandB run name                                        [required]
#   -w  weight file (.pth)              default: none (random)
#   -d  dataset subdirectory            default: scannet
#   -p  python interpreter              default: python
#   -j  WandB project name              default: LitePT-Profiling
#   -k  WandB API key                   default: current wandb login session
#   -W  n_warmup                        default: 5
#   -M  n_measure                       default: 30
#   -B  n_profile_batches               default: 50
#   -T  flag: skip training-memory step default: off (measure it)
# =============================================================================

set -euo pipefail
# Navigate to the LitePT repo root (parent of the scripts/ directory).
cd "$(dirname "$(dirname "$0")")" || exit 1

# ── Defaults ──────────────────────────────────────────────────────────────────
PYTHON=python
DATASET=scannet
CONFIG=""
RUN_NAME=""
WEIGHT=""
WANDB_PROJECT="LitePT-Profiling"
WANDB_KEY=""
N_WARMUP=5
N_MEASURE=30
N_BATCHES=50
SKIP_TRAIN=0   # 0 = measure training memory; 1 = skip it

# ── Argument parsing ──────────────────────────────────────────────────────────
# Note: -T is a boolean flag (no argument); all others take arguments (:).
while getopts "p:d:c:n:w:j:k:W:M:B:T" opt; do
    case $opt in
        p) PYTHON=$OPTARG         ;;
        d) DATASET=$OPTARG        ;;
        c) CONFIG=$OPTARG         ;;
        n) RUN_NAME=$OPTARG       ;;
        w) WEIGHT=$OPTARG         ;;
        j) WANDB_PROJECT=$OPTARG  ;;
        k) WANDB_KEY=$OPTARG      ;;
        W) N_WARMUP=$OPTARG       ;;
        M) N_MEASURE=$OPTARG      ;;
        B) N_BATCHES=$OPTARG      ;;
        T) SKIP_TRAIN=1           ;;
        \?) echo "ERROR: invalid option -$OPTARG"; exit 1 ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [ -z "$CONFIG" ]; then
    echo "ERROR: -c config_name is required."
    echo "Example: bash scriptss/profile.sh -c semseg-litept-lw-e -n lw-e-random"
    exit 1
fi

if [ -z "$RUN_NAME" ]; then
    # Auto-generate a run name from config + timestamp if not supplied.
    RUN_NAME="${CONFIG}-$(date +%Y%m%d-%H%M)"
    echo "  RUN_NAME not supplied; using auto-generated: $RUN_NAME"
fi

CONFIG_PATH="configs/${DATASET}/${CONFIG}.py"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ -n "$WEIGHT" ] && [ ! -f "$WEIGHT" ]; then
    echo "ERROR: Weight file not found: $WEIGHT"
    exit 1
fi

# ── Header ────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  LitePT Profiler"
printf "  Config   : %s\n" "$CONFIG"
printf "  Run name : %s\n" "$RUN_NAME"
printf "  Project  : %s\n" "$WANDB_PROJECT"
printf "  Weight   : %s\n" "${WEIGHT:-none (random/untrained weights)}"
printf "  Time     : %s\n" "$(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================"
echo ""

# ── Build Python command ──────────────────────────────────────────────────────
# Use an array to handle paths with spaces correctly.
CMD=( "$PYTHON" "tools/profile_model.py" )
CMD+=( --config-file     "$CONFIG_PATH"   )
CMD+=( --run-name        "$RUN_NAME"      )
CMD+=( --wandb-project   "$WANDB_PROJECT" )
CMD+=( --n-warmup        "$N_WARMUP"      )
CMD+=( --n-measure       "$N_MEASURE"     )
CMD+=( --n-profile-batches "$N_BATCHES"   )

[ -n "$WEIGHT"    ] && CMD+=( --weight    "$WEIGHT"    )
[ -n "$WANDB_KEY" ] && CMD+=( --wandb-key "$WANDB_KEY" )
[ "$SKIP_TRAIN" -eq 1 ] && CMD+=( --skip-train-memory )

echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "Finished at $(date '+%Y-%m-%d %H:%M:%S')"