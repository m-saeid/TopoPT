#!/usr/bin/env bash
# =============================================================================
# scripts/profile_all.sh  —  Batch profiler for all LitePT experiment configs
# =============================================================================
#

# All 32 configs across 4 datasets are registered with metadata: dataset,
# config stem, IS_KD flag, and WandB tags. Usage is simple:
# bash scripts/profile_all.sh                    # full run, both modes
# bash scripts/profile_all.sh -m random          # architecture survey only (fast)
# bash scripts/profile_all.sh -m trained -f lw-c # trained runs, lw-c variants only
# bash scripts/profile_all.sh -D                 # dry-run to preview all commands

# KD configs are handled correctly. Random profiling is skipped for *-kd-* configs (their
# inference architecture is identical to lw-c — running twice just duplicates the numbers).
# Trained profiling uses --no-strict-load so the teacher/projector keys in the KD
# checkpoint don't block loading the student.

# Profiles every listed config in random-weight mode (architecture / hardware
# cost, reproducible without any training) and — if a trained checkpoint is
# found — in trained-weight mode (real-world inference cost).
#
# Designed for NeurIPS Table 1 / Table 2 style reporting:
#   * Params (M) · GFLOPs · Latency (ms) · Memory (GB) · Throughput (sps)
#   * Consistent across all datasets and model variants
#
# KD (distillation) configs:
#   Random profiling is SKIPPED — their inference architecture is identical
#   to the base student config (e.g. lw-c), so the random numbers would be
#   duplicates.  Trained profiling IS run if a checkpoint exists, and uses
#   --no-strict-load so extra teacher/projector keys in the KD checkpoint
#   do not block loading into the student-only inference view.
#
# Usage:
#   bash scripts/profile_all.sh [OPTIONS]
#
# Options:
#   -p  Python interpreter              default: python
#   -j  WandB project name             default: LitePT-Profiling
#   -k  WandB API key                  default: current wandb login session
#   -m  Mode: random | trained | both  default: both
#   -W  n_warmup iterations            default: 10
#   -M  n_measure iterations           default: 50
#   -B  n_profile_batches              default: 50
#   -T  skip training-memory step      default: off
#   -D  dry-run (print, do not run)    default: off
#   -f  filter keyword (grep on name)  default: "" (all configs)
#   -s  stop on first error            default: off (continue)
#
# Examples:
#   # Full run, both modes, all configs:
#   bash scripts/profile_all.sh
#
#   # Only random-weight pass (fast architecture survey):
#   bash scripts/profile_all.sh -m random
#
#   # Only scannet semseg configs, trained mode:
#   bash scripts/profile_all.sh -m trained -f semseg
#
#   # Dry run to preview all commands:
#   bash scripts/profile_all.sh -D
#
#   # Custom WandB project + suppress training memory for speed:
#   bash scripts/profile_all.sh -j MyPaper-Profiling -T
# =============================================================================

set -uo pipefail
cd "$(dirname "$(dirname "$0")")" || exit 1

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; YELLOW='\033[1;33m'; GREEN='\033[0;32m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
log_info()  { echo -e "${CYAN}[INFO]${RESET}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${RESET}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
log_error() { echo -e "${RED}[ERROR]${RESET} $*"; }
log_head()  { echo -e "\n${BOLD}${CYAN}══ $* ══${RESET}"; }

# ── Defaults ───────────────────────────────────────────────────────────────────
PYTHON=python
WANDB_PROJECT="LitePT-Profiling"
WANDB_KEY=""
MODE="both"
N_WARMUP=10
N_MEASURE=50
N_BATCHES=50
SKIP_TRAIN=0
DRY_RUN=0
FILTER=""
STOP_ON_ERROR=0

# ── Argument parsing ───────────────────────────────────────────────────────────
while getopts "p:j:k:m:W:M:B:TDf:s" opt; do
    case $opt in
        p) PYTHON=$OPTARG         ;;
        j) WANDB_PROJECT=$OPTARG  ;;
        k) WANDB_KEY=$OPTARG      ;;
        m) MODE=$OPTARG           ;;
        W) N_WARMUP=$OPTARG       ;;
        M) N_MEASURE=$OPTARG      ;;
        B) N_BATCHES=$OPTARG      ;;
        T) SKIP_TRAIN=1           ;;
        D) DRY_RUN=1              ;;
        f) FILTER=$OPTARG         ;;
        s) STOP_ON_ERROR=1        ;;
        \?) log_error "Invalid option: -$OPTARG"; exit 1 ;;
    esac
done

# ── Validate mode ──────────────────────────────────────────────────────────────
if [[ "$MODE" != "random" && "$MODE" != "trained" && "$MODE" != "both" ]]; then
    log_error "-m must be one of: random | trained | both"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════════
#  Config registry
#
#  Format: "DATASET|CONFIG_STEM|IS_KD|TAGS"
#
#  DATASET    : subdirectory under configs/ and exp/
#  CONFIG_STEM: filename without .py
#  IS_KD      : 1 = skip random profiling; only run trained mode (student
#               arch is identical to base lw-c, random numbers are duplicate)
#  TAGS       : comma-separated WandB tags (used for filtering in the UI)
# ══════════════════════════════════════════════════════════════════════════════
CONFIGS=(
    # ── ScanNet · Semantic Segmentation ───────────────────────────────────────
    # S0 baselines
    "scannet|semseg-litept-small-v1m1|0|semseg,scannet,baseline,S0,no-decoder"
    "scannet|semseg-litept-small-v1m2|0|semseg,scannet,baseline,S0,full-decoder"
    "scannet|semseg-litept-rerun|0|semseg,scannet,baseline,S0,rerun"

    # Ablation series (Strategy-1)
    "scannet|semseg-litept-lw-a|0|semseg,scannet,lw-a,S1,depth-only"
    "scannet|semseg-litept-lw-b|0|semseg,scannet,lw-b,S2,channel-only"
    "scannet|semseg-litept-lw-d|0|semseg,scannet,lw-d,S3,patch-only"
    "scannet|semseg-lw-c-100epoch|0|semseg,scannet,lw-c,S4,depth+channel,100ep"
    "scannet|semseg-lw-c-1200epoch|0|semseg,scannet,lw-c,S4,depth+channel,1200ep"
    "scannet|semseg-litept-lw-e|0|semseg,scannet,lw-e,S5,all-reductions"

    # KD / SRFD (Strategy-2)
    "scannet|semseg-lw-c-kd-100epoch|1|semseg,scannet,lw-c,kd,srfd,100ep"
    "scannet|semseg-lw-c-kd-1200epoch|1|semseg,scannet,lw-c,kd,srfd,1200ep"

    # ── ScanNet · Instance Segmentation ───────────────────────────────────────
    "scannet|insseg-litept-small-v1m2|0|insseg,scannet,baseline,S0"
    "scannet|insseg-lw-c-100epoch|0|insseg,scannet,lw-c,100ep"
    "scannet|insseg-lw-c-800epoch|0|insseg,scannet,lw-c,800ep"
    "scannet|insseg-lw-c-kd-100epoch|1|insseg,scannet,lw-c,kd,srfd,100ep"
    "scannet|insseg-lw-c-kd-800epoch|1|insseg,scannet,lw-c,kd,srfd,800ep"

    # ── ScanNet200 · Instance Segmentation ────────────────────────────────────
    "scannet200|insseg-litept-small-v1m2|0|insseg,scannet200,baseline,S0"
    "scannet200|insseg-lw-c-100epoch|0|insseg,scannet200,lw-c,100ep"
    "scannet200|insseg-lw-c-800epoch|0|insseg,scannet200,lw-c,800ep"
    "scannet200|insseg-lw-c-kd-100epoch|1|insseg,scannet200,lw-c,kd,srfd,100ep"
    "scannet200|insseg-lw-c-kd-800epoch|1|insseg,scannet200,lw-c,kd,srfd,800ep"

    # ── nuScenes · Semantic Segmentation ──────────────────────────────────────
    "nuscenes|semseg-litept-small-v1m1|0|semseg,nuscenes,baseline,S0"
    "nuscenes|semseg-lw-c|0|semseg,nuscenes,lw-c"
    "nuscenes|semseg-lw-c-kd|1|semseg,nuscenes,lw-c,kd,srfd"

    # ── Structured3D · Semantic Segmentation ──────────────────────────────────
    "structured3d|semseg-litept-base-v1m1|0|semseg,structured3d,baseline,base"
    "structured3d|semseg-litept-large-v1m1|0|semseg,structured3d,baseline,large"
    "structured3d|semseg-litept-small-v1m1|0|semseg,structured3d,baseline,small"
    "structured3d|semseg-lw-c|0|semseg,structured3d,lw-c"
    "structured3d|semseg-lw-c-kd-tb|1|semseg,structured3d,lw-c,kd,srfd,teacher-base"
    "structured3d|semseg-lw-c-kd-tl|1|semseg,structured3d,lw-c,kd,srfd,teacher-large"
    "structured3d|semseg-lw-c-kd-ts|1|semseg,structured3d,lw-c,kd,srfd,teacher-small"
)

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: find best checkpoint in an experiment directory
#  Priority: model_best.pth > model_last.pth > (none)
# ══════════════════════════════════════════════════════════════════════════════
find_checkpoint() {
    local exp_dir="$1"
    local model_dir="${exp_dir}/model"
    if   [ -f "${model_dir}/model_best.pth" ]; then
        echo "${model_dir}/model_best.pth"
    elif [ -f "${model_dir}/model_last.pth" ]; then
        echo "${model_dir}/model_last.pth"
    else
        echo ""
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
#  Core profiler invocation
#  Args: $1=config_path $2=run_name $3=weight_path(or"") $4=extra_tags
#        $5=use_no_strict(0|1)  $6=student_config_path(for KD trained mode)
# ══════════════════════════════════════════════════════════════════════════════
run_one() {
    local config_path="$1"
    local run_name="$2"
    local weight="${3:-}"
    local extra_tags="${4:-}"
    local no_strict="${5:-0}"
    local override_config="${6:-}"

    # Allow config override (used for KD trained: student config + KD weights)
    local effective_config="${override_config:-$config_path}"

    local cmd=("$PYTHON" "tools/profile_model.py")
    cmd+=(--config-file     "$effective_config")
    cmd+=(--run-name        "$run_name")
    cmd+=(--wandb-project   "$WANDB_PROJECT")
    cmd+=(--n-warmup        "$N_WARMUP")
    cmd+=(--n-measure       "$N_MEASURE")
    cmd+=(--n-profile-batches "$N_BATCHES")

    [ -n "$weight"      ] && cmd+=(--weight      "$weight")
    [ -n "$WANDB_KEY"   ] && cmd+=(--wandb-key   "$WANDB_KEY")
    [ "$SKIP_TRAIN" -eq 1 ] && cmd+=(--skip-train-memory)
    [ "$no_strict"  -eq 1 ] && cmd+=(--no-strict-load)
    [ -n "$extra_tags"  ] && cmd+=(--extra-tags  "$extra_tags")

    if [ "$DRY_RUN" -eq 1 ]; then
        echo -e "  ${YELLOW}[DRY-RUN]${RESET} ${cmd[*]}"
        return 0
    fi

    echo ""
    log_info "Running: ${cmd[*]}"
    if "${cmd[@]}"; then
        log_ok "Finished: $run_name"
    else
        local rc=$?
        log_warn "FAILED (exit $rc): $run_name"
        if [ "$STOP_ON_ERROR" -eq 1 ]; then
            log_error "Aborting due to -s (stop-on-error)."
            exit $rc
        fi
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
#  Counters
# ══════════════════════════════════════════════════════════════════════════════
N_TOTAL=0; N_SKIPPED=0; N_RANDOM=0; N_TRAINED=0; N_FAILED=0

# ══════════════════════════════════════════════════════════════════════════════
#  Banner
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║         LitePT  Batch Profiler                   ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
echo ""
printf "  %-22s %s\n" "WandB project:"    "$WANDB_PROJECT"
printf "  %-22s %s\n" "Mode:"             "$MODE"
printf "  %-22s %s\n" "n_warmup:"         "$N_WARMUP"
printf "  %-22s %s\n" "n_measure:"        "$N_MEASURE"
printf "  %-22s %s\n" "n_profile_batches:" "$N_BATCHES"
printf "  %-22s %s\n" "Skip train mem:"   "$( [ $SKIP_TRAIN -eq 1 ] && echo yes || echo no )"
printf "  %-22s %s\n" "Filter:"           "${FILTER:-<none>}"
printf "  %-22s %s\n" "Dry-run:"          "$( [ $DRY_RUN -eq 1 ] && echo YES || echo no )"
printf "  %-22s %s\n" "Configs total:"    "${#CONFIGS[@]}"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════
for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r dataset config_stem is_kd tags <<< "$entry"

    # Apply filter
    if [ -n "$FILTER" ] && ! echo "${dataset}/${config_stem}" | grep -q "$FILTER"; then
        continue
    fi

    N_TOTAL=$((N_TOTAL + 1))
    config_path="configs/${dataset}/${config_stem}.py"
    exp_dir="exp/${dataset}/${config_stem}"

    log_head "${dataset} / ${config_stem}"

    # ── Config existence check ────────────────────────────────────────────────
    if [ ! -f "$config_path" ]; then
        log_warn "Config NOT FOUND: $config_path — skipping."
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi
    log_info "Config:     $config_path"

    # ── Locate trained checkpoint ─────────────────────────────────────────────
    trained_ckpt="$(find_checkpoint "$exp_dir")"
    if [ -n "$trained_ckpt" ]; then
        log_info "Checkpoint: $trained_ckpt"
    else
        log_info "Checkpoint: none found in $exp_dir/model/"
    fi

    # ════════════════════════════════════════════════════════════════════
    #  RANDOM profiling
    # ════════════════════════════════════════════════════════════════════
    if [[ "$MODE" == "random" || "$MODE" == "both" ]]; then
        if [ "$is_kd" -eq 1 ]; then
            log_info "[random] SKIP — KD config; inference arch = base student (already profiled)"
        else
            log_info "[random] Profiling random weights …"
            run_name="${dataset}/${config_stem}-random"
            run_one \
                "$config_path" \
                "$run_name" \
                "" \
                "${tags},random" \
                0
            N_RANDOM=$((N_RANDOM + 1))
        fi
    fi

    # ════════════════════════════════════════════════════════════════════
    #  TRAINED profiling
    # ════════════════════════════════════════════════════════════════════
    if [[ "$MODE" == "trained" || "$MODE" == "both" ]]; then
        if [ -z "$trained_ckpt" ]; then
            log_info "[trained] No checkpoint — skipping trained run."
        else
            log_info "[trained] Profiling trained checkpoint …"
            run_name="${dataset}/${config_stem}-trained"

            if [ "$is_kd" -eq 1 ]; then
                # ── KD trained: load KD checkpoint via --no-strict-load ──────
                # The KD checkpoint contains teacher_backbone.* and projectors.*
                # keys that don't exist in the student-only inference graph.
                # --no-strict-load (added to profile_model.py) filters them out
                # so only backbone.* keys are loaded. This correctly reflects
                # the STUDENT checkpoint after KD training.
                #
                # The WandB run name carries "-kd-trained" so it's distinct
                # from the base student trained run in the comparison table.
                log_info "[trained] KD mode: using --no-strict-load to extract student weights."
                run_one \
                    "$config_path" \
                    "$run_name" \
                    "$trained_ckpt" \
                    "${tags},trained,kd-checkpoint" \
                    1
            else
                run_one \
                    "$config_path" \
                    "$run_name" \
                    "$trained_ckpt" \
                    "${tags},trained" \
                    0
            fi
            N_TRAINED=$((N_TRAINED + 1))
        fi
    fi

done  # ── end main loop ──────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║  Profiling complete — Summary                    ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════╝${RESET}"
printf "  %-30s %d\n" "Configs processed:"      "$N_TOTAL"
printf "  %-30s %d\n" "Skipped (no config):"    "$N_SKIPPED"
printf "  %-30s %d\n" "Random runs launched:"   "$N_RANDOM"
printf "  %-30s %d\n" "Trained runs launched:"  "$N_TRAINED"
echo ""
log_info "Results → WandB project: $WANDB_PROJECT"
echo ""