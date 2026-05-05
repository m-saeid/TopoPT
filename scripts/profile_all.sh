#!/usr/bin/env bash
# =============================================================================
# scripts/profile_all.sh  —  Batch profiler for all LitePT / TopoPT experiments
# =============================================================================
#
# Runs tools/profile_model.py for every config in the registry, in two modes:
#   random  — random (untrained) weights; measures architecture cost only
#   trained — loads the best checkpoint from the matching exp/ directory
#
# Checkpoint search order (find_checkpoint):
#   1. exp/<dataset>-<stem>/model/model_best.pth       (student / ablation runs)
#   2. exp/<dataset>-<stem>/model/model_last.pth       (student fallback)
#   3. exp/litept/<dataset>-<stem>/model/model_best.pth (LitePT baseline teachers)
#   4. exp/litept/<dataset>-<stem>/model/model_last.pth (LitePT baseline fallback)
#
# KD trained runs:
#   The KD checkpoint embeds teacher backbone + projector weights alongside the
#   student.  For a correct inference-time profile (params / memory / latency
#   reflecting only the student), the profiler is invoked with the BASE STUDENT
#   config (not the KD config) plus --no-strict-load.  The get_student_config()
#   function maps every KD stem to its corresponding student stem.
#
# Usage:
#   bash scripts/profile_all.sh [options]
#
# Options:
#   -p PYTHON      Python interpreter (default: python)
#   -j PROJECT     WandB project name (default: LitePT-Profiling)
#   -k KEY         WandB API key (default: uses existing login or WANDB_API_KEY)
#   -m MODE        random | trained | both (default: both)
#   -W N           Latency warmup iterations (default: 10)
#   -M N           Latency measurement iterations (default: 50)
#   -B N           Val batches for memory/latency (default: 50)
#   -T             Skip training-mode memory measurement
#   -D             Dry-run: print commands without executing
#   -f PATTERN     Only run entries whose "dataset/stem" matches PATTERN (grep)
#   -s             Stop on first error (default: continue and count failures)
#
# Examples:
#   # Profile everything, both modes:
#   bash scripts/profile_all.sh
#
#   # Dry-run to verify commands:
#   bash scripts/profile_all.sh -D
#
#   # Only nuscenes entries:
#   bash scripts/profile_all.sh -f nuscenes
#
#   # Only trained runs, skip training memory (faster):
#   bash scripts/profile_all.sh -m trained -T
#
#   # Custom WandB project and Python env:
#   bash scripts/profile_all.sh -p /opt/conda/bin/python -j MyProject -k $WANDB_KEY
# =============================================================================

set -uo pipefail
# Note: -e (errexit) is intentionally omitted. Failures are tracked manually
# via N_FAILED so the batch continues even when individual runs fail.

# Move to repo root regardless of where the script is called from
cd "$(dirname "$(dirname "$(realpath "$0")")")" || {
    echo "[ERROR] Could not cd to repo root" >&2; exit 1
}

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
        \?) log_error "Unknown option: -$OPTARG"; exit 1 ;;
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
#  IS_KD=1 entries:
#    - are always skipped in random mode (the base student config already covers
#      the architecture; no need to duplicate the profile)
#    - in trained mode, the checkpoint is loaded with --no-strict-load and the
#      BASE STUDENT config is used (via get_student_config) so that params /
#      memory / latency reflect inference cost — not the full distillation graph.
#
#  Config stems must exactly match filenames in configs/<dataset>/.
#  Experiment stems are resolved as: exp/<dataset>-<stem>/ (flat hyphenated layout)
#  LitePT baseline stems are resolved as: exp/litept/<dataset>-<stem>/
# ══════════════════════════════════════════════════════════════════════════════
CONFIGS=(
    # ── ScanNet · Semantic Segmentation ───────────────────────────────────────
    "scannet|semseg-litept-small-v1m1|0|semseg,scannet,baseline,litept-S,no-decoder"
    "scannet|semseg-litept-small-v1m2|0|semseg,scannet,baseline,litept-S,full-decoder"
    "scannet|semseg-litept-rerun-100epoch|0|semseg,scannet,baseline,litept-S,rerun"

    # Ablation series
    "scannet|semseg-lw-a-100epoch|0|semseg,scannet,lw-a,depth-only"
    "scannet|semseg-lw-b-100epoch|0|semseg,scannet,lw-b,channel-only"
    "scannet|semseg-lw-c-100epoch|0|semseg,scannet,lw-c,depth+channel,100ep"
    "scannet|semseg-lw-c-1200epoch|0|semseg,scannet,lw-c,depth+channel,1200ep"
    "scannet|semseg-lw-d-100epoch|0|semseg,scannet,lw-d,patch-only"
    "scannet|semseg-lw-e-100epoch|0|semseg,scannet,lw-e,all-reductions"

    # KD / SRFD — TopoPT
    "scannet|semseg-lw-c-kd-100epoch|1|semseg,scannet,lw-c,kd,srfd,topopt,100ep"
    "scannet|semseg-lw-c-kd-1200epoch|1|semseg,scannet,lw-c,kd,srfd,topopt,1200ep"

    # ── ScanNet · Instance Segmentation ───────────────────────────────────────
    "scannet|insseg-litept-small-v1m2|0|insseg,scannet,baseline,litept-S"
    "scannet|insseg-lw-c-100epoch|0|insseg,scannet,lw-c,100ep"
    "scannet|insseg-lw-c-800epoch|0|insseg,scannet,lw-c,800ep"
    "scannet|insseg-lw-c-kd-100epoch|1|insseg,scannet,lw-c,kd,srfd,topopt,100ep"
    "scannet|insseg-lw-c-kd-800epoch|1|insseg,scannet,lw-c,kd,srfd,topopt,800ep"

    # ── ScanNet200 · Instance Segmentation ────────────────────────────────────
    "scannet200|insseg-litept-small-v1m2|0|insseg,scannet200,baseline,litept-S"
    "scannet200|insseg-lw-c-100epoch|0|insseg,scannet200,lw-c,100ep"
    "scannet200|insseg-lw-c-800epoch|0|insseg,scannet200,lw-c,800ep"
    "scannet200|insseg-lw-c-kd-100epoch|1|insseg,scannet200,lw-c,kd,srfd,topopt,100ep"
    "scannet200|insseg-lw-c-kd-800epoch|1|insseg,scannet200,lw-c,kd,srfd,topopt,800ep"

    # ── nuScenes · Semantic Segmentation ──────────────────────────────────────
    "nuscenes|semseg-litept-small-v1m1|0|semseg,nuscenes,baseline,litept-S"
    "nuscenes|semseg-lw-c-50epoch|0|semseg,nuscenes,lw-c,50ep"
    "nuscenes|semseg-lw-c-kd-50epoch|1|semseg,nuscenes,lw-c,kd,srfd,topopt,50ep"

    # ── Structured3D · Semantic Segmentation ──────────────────────────────────
    "structured3d|semseg-litept-small-v1m1|0|semseg,structured3d,baseline,litept-S,small"
    "structured3d|semseg-litept-base-v1m1|0|semseg,structured3d,baseline,litept-B,base"
    "structured3d|semseg-litept-large-v1m1|0|semseg,structured3d,baseline,litept-L,large"
    "structured3d|semseg-lw-c-50epoch|0|semseg,structured3d,lw-c,50ep"
    "structured3d|semseg-lw-c-200epoch|0|semseg,structured3d,lw-c,200ep"
    "structured3d|semseg-lw-c-kd-ts-50epoch|1|semseg,structured3d,lw-c,kd,srfd,topopt,teacher-small,50ep"
    "structured3d|semseg-lw-c-kd-tb-50epoch|1|semseg,structured3d,lw-c,kd,srfd,topopt,teacher-base,50ep"
    "structured3d|semseg-lw-c-kd-tl-50epoch|1|semseg,structured3d,lw-c,kd,srfd,topopt,teacher-large,50ep"
    "structured3d|semseg-lw-c-kd-ts-200epoch|1|semseg,structured3d,lw-c,kd,srfd,topopt,teacher-small,200ep"
    "structured3d|semseg-lw-c-kd-tb-200epoch|1|semseg,structured3d,lw-c,kd,srfd,topopt,teacher-base,200ep"
    "structured3d|semseg-lw-c-kd-tl-200epoch|1|semseg,structured3d,lw-c,kd,srfd,topopt,teacher-large,200ep"
)


# ══════════════════════════════════════════════════════════════════════════════
#  KD stem → base student config stem mapping
#
#  Used so trained KD runs profile with the student-only config, which ensures
#  params / memory / latency reflect only the inference-time student model.
#
#  Returns the full path to the student config file.
#  Falls back to the KD config itself (with a warning) if no mapping is found.
# ══════════════════════════════════════════════════════════════════════════════
get_student_config() {
    local dataset="$1"
    local kd_stem="$2"
    local student_stem=""

    case "${dataset}|${kd_stem}" in
        # ScanNet semseg
        "scannet|semseg-lw-c-kd-100epoch")           student_stem="semseg-lw-c-100epoch"   ;;
        "scannet|semseg-lw-c-kd-1200epoch")          student_stem="semseg-lw-c-1200epoch"  ;;

        # ScanNet insseg
        "scannet|insseg-lw-c-kd-100epoch")           student_stem="insseg-lw-c-100epoch"   ;;
        "scannet|insseg-lw-c-kd-800epoch")           student_stem="insseg-lw-c-800epoch"   ;;

        # ScanNet200 insseg
        "scannet200|insseg-lw-c-kd-100epoch")        student_stem="insseg-lw-c-100epoch"   ;;
        "scannet200|insseg-lw-c-kd-800epoch")        student_stem="insseg-lw-c-800epoch"   ;;

        # nuScenes semseg
        "nuscenes|semseg-lw-c-kd-50epoch")           student_stem="semseg-lw-c-50epoch"    ;;

        # Structured3D semseg — 50-epoch variants
        "structured3d|semseg-lw-c-kd-ts-50epoch")   student_stem="semseg-lw-c-50epoch"    ;;
        "structured3d|semseg-lw-c-kd-tb-50epoch")   student_stem="semseg-lw-c-50epoch"    ;;
        "structured3d|semseg-lw-c-kd-tl-50epoch")   student_stem="semseg-lw-c-50epoch"    ;;

        # Structured3D semseg — 200-epoch variants
        "structured3d|semseg-lw-c-kd-ts-200epoch")  student_stem="semseg-lw-c-200epoch"   ;;
        "structured3d|semseg-lw-c-kd-tb-200epoch")  student_stem="semseg-lw-c-200epoch"   ;;
        "structured3d|semseg-lw-c-kd-tl-200epoch")  student_stem="semseg-lw-c-200epoch"   ;;

        *)
            log_warn "No student config mapping for ${dataset}/${kd_stem} — falling back to KD config."
            echo "configs/${dataset}/${kd_stem}.py"
            return
            ;;
    esac

    echo "configs/${dataset}/${student_stem}.py"
}


# ══════════════════════════════════════════════════════════════════════════════
#  Checkpoint discovery
#
#  Arguments:
#    $1  exp_dir  — primary flat experiment directory, e.g. exp/nuscenes-semseg-lw-c-50epoch
#    $2  stem     — "<dataset>-<config_stem>", used to probe the litept/ subtree
#                   for LitePT baseline teacher checkpoints, e.g. nuscenes-semseg-litept-small-v1m1
#
#  Search order (first match wins):
#    1. <exp_dir>/model/model_best.pth
#    2. <exp_dir>/model/model_last.pth
#    3. exp/litept/<stem>/model/model_best.pth
#    4. exp/litept/<stem>/model/model_last.pth
#
#  Returns the checkpoint path on stdout, or empty string if none found.
# ══════════════════════════════════════════════════════════════════════════════
find_checkpoint() {
    local exp_dir="$1"
    local stem="$2"

    local candidates=(
        "${exp_dir}/model/model_best.pth"
        "${exp_dir}/model/model_last.pth"
        "exp/litept/${stem}/model/model_best.pth"
        "exp/litept/${stem}/model/model_last.pth"
    )

    for ckpt in "${candidates[@]}"; do
        if [ -f "$ckpt" ]; then
            echo "$ckpt"
            return 0
        fi
    done

    echo ""
    return 1
}


# ══════════════════════════════════════════════════════════════════════════════
#  Core profiler invocation
#
#  Arguments:
#    $1  config_path       — path to the .py config file
#    $2  run_name          — WandB run name, e.g. "nuscenes/semseg-lw-c-50epoch-random"
#    $3  weight            — checkpoint path, or "" for random weights
#    $4  extra_tags        — comma-separated WandB tags
#    $5  no_strict         — 0 (strict load) or 1 (--no-strict-load, for KD checkpoints)
#    $6  override_config   — if non-empty, use this config instead of $1 (KD student config)
# ══════════════════════════════════════════════════════════════════════════════
run_one() {
    local config_path="$1"
    local run_name="$2"
    local weight="${3:-}"
    local extra_tags="${4:-}"
    local no_strict="${5:-0}"
    local override_config="${6:-}"

    # For KD trained runs, override_config is the base student config.
    # For all other runs, override_config is empty and config_path is used directly.
    local effective_config="${override_config:-$config_path}"

    local cmd=("$PYTHON" "tools/profile_model.py")
    cmd+=(--config-file       "$effective_config")
    cmd+=(--run-name          "$run_name")
    cmd+=(--wandb-project     "$WANDB_PROJECT")
    cmd+=(--n-warmup          "$N_WARMUP")
    cmd+=(--n-measure         "$N_MEASURE")
    cmd+=(--n-profile-batches "$N_BATCHES")

    [ -n "$weight"         ] && cmd+=(--weight         "$weight")
    [ -n "$WANDB_KEY"      ] && cmd+=(--wandb-key      "$WANDB_KEY")
    [ "$SKIP_TRAIN" -eq 1  ] && cmd+=(--skip-train-memory)
    [ "$no_strict"  -eq 1  ] && cmd+=(--no-strict-load)
    [ -n "$extra_tags"     ] && cmd+=(--extra-tags     "$extra_tags")

    if [ "$DRY_RUN" -eq 1 ]; then
        echo -e "  ${YELLOW}[DRY-RUN]${RESET} ${cmd[*]}"
        return 0
    fi

    echo ""
    log_info "Running: ${cmd[*]}"

    if "${cmd[@]}"; then
        log_ok "Finished: $run_name"
        return 0
    else
        local rc=$?
        log_warn "FAILED (exit $rc): $run_name"
        return $rc
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
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║     TopoPT / LitePT  Batch Profiler                  ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""
printf "  %-24s %s\n" "WandB project:"      "$WANDB_PROJECT"
printf "  %-24s %s\n" "Mode:"               "$MODE"
printf "  %-24s %s\n" "n_warmup:"           "$N_WARMUP"
printf "  %-24s %s\n" "n_measure:"          "$N_MEASURE"
printf "  %-24s %s\n" "n_profile_batches:"  "$N_BATCHES"
printf "  %-24s %s\n" "Skip train mem:"     "$( [ $SKIP_TRAIN -eq 1 ] && echo yes || echo no )"
printf "  %-24s %s\n" "Filter:"             "${FILTER:-<none>}"
printf "  %-24s %s\n" "Stop on error:"      "$( [ $STOP_ON_ERROR -eq 1 ] && echo yes || echo no )"
printf "  %-24s %s\n" "Dry-run:"            "$( [ $DRY_RUN -eq 1 ] && echo YES || echo no )"
printf "  %-24s %s\n" "Configs registered:" "${#CONFIGS[@]}"
echo ""


# ══════════════════════════════════════════════════════════════════════════════
#  Main loop
# ══════════════════════════════════════════════════════════════════════════════
for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r dataset config_stem is_kd tags <<< "$entry"

    # ── Filter ────────────────────────────────────────────────────────────────
    if [ -n "$FILTER" ] && ! echo "${dataset}/${config_stem}" | grep -q "$FILTER"; then
        continue
    fi

    N_TOTAL=$((N_TOTAL + 1))

    config_path="configs/${dataset}/${config_stem}.py"

    # Flat hyphenated experiment directory (student / ablation runs)
    exp_dir="exp/${dataset}-${config_stem}"

    # Stem used to probe the litept/ subtree for baseline teacher checkpoints
    exp_stem="${dataset}-${config_stem}"

    log_head "${dataset} / ${config_stem}"

    # ── Config existence check ─────────────────────────────────────────────────
    if [ ! -f "$config_path" ]; then
        log_warn "Config NOT FOUND: $config_path — skipping."
        N_SKIPPED=$((N_SKIPPED + 1))
        [ "$STOP_ON_ERROR" -eq 1 ] && exit 1
        continue
    fi
    log_info "Config: $config_path"

    # ── Checkpoint discovery ───────────────────────────────────────────────────
    # Pass both the flat exp_dir AND the stem so find_checkpoint can also
    # probe exp/litept/<stem>/ for LitePT baseline teacher checkpoints.
    trained_ckpt="$(find_checkpoint "$exp_dir" "$exp_stem")"

    if [ -n "$trained_ckpt" ]; then
        log_info "Checkpoint: $trained_ckpt"
    else
        log_info "Checkpoint: none found (checked $exp_dir/model/ and exp/litept/${exp_stem}/model/)"
    fi

    # ── RANDOM profiling ───────────────────────────────────────────────────────
    # KD configs are skipped in random mode: the base student config (IS_KD=0)
    # already profiles the same inference-time architecture.
    if [[ "$MODE" == "random" || "$MODE" == "both" ]]; then
        if [ "$is_kd" -eq 1 ]; then
            log_info "[random] SKIP — KD config; inference arch = base student (already profiled)"
        else
            log_info "[random] Profiling random weights …"
            run_name="${dataset}/${config_stem}-random"
            run_one "$config_path" "$run_name" "" "${tags},random" 0 \
                || { N_FAILED=$((N_FAILED + 1)); [ "$STOP_ON_ERROR" -eq 1 ] && exit 1; }
            N_RANDOM=$((N_RANDOM + 1))
        fi
    fi

    # ── TRAINED profiling ──────────────────────────────────────────────────────
    if [[ "$MODE" == "trained" || "$MODE" == "both" ]]; then
        if [ -z "$trained_ckpt" ]; then
            log_info "[trained] No checkpoint found — skipping trained run."
        else
            log_info "[trained] Profiling trained checkpoint …"
            run_name="${dataset}/${config_stem}-trained"

            if [ "$is_kd" -eq 1 ]; then
                # KD trained run: use the base student config so build_model()
                # constructs only the student inference graph. The KD checkpoint
                # contains teacher / projector keys which are discarded via
                # --no-strict-load. The resulting params / memory / latency
                # correctly reflect inference cost with no distillation overhead.
                log_info "[trained][kd] Resolving student config …"
                student_cfg="$(get_student_config "$dataset" "$config_stem")"
                log_info "[trained][kd] Student config: $student_cfg"

                if [ ! -f "$student_cfg" ]; then
                    log_warn "Student config not found: $student_cfg — falling back to KD config (params will be inflated)."
                    student_cfg=""
                fi

                run_one \
                    "$config_path" \
                    "$run_name" \
                    "$trained_ckpt" \
                    "${tags},trained,kd-checkpoint" \
                    1 \
                    "$student_cfg" \
                    || { N_FAILED=$((N_FAILED + 1)); [ "$STOP_ON_ERROR" -eq 1 ] && exit 1; }
            else
                run_one \
                    "$config_path" \
                    "$run_name" \
                    "$trained_ckpt" \
                    "${tags},trained" \
                    0 \
                    || { N_FAILED=$((N_FAILED + 1)); [ "$STOP_ON_ERROR" -eq 1 ] && exit 1; }
            fi

            N_TRAINED=$((N_TRAINED + 1))
        fi
    fi

done  # ── end main loop ─────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║  Profiling complete — Summary                        ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${RESET}"
printf "  %-32s %d\n" "Configs processed:"    "$N_TOTAL"
printf "  %-32s %d\n" "Skipped (no config):"  "$N_SKIPPED"
printf "  %-32s %d\n" "Random runs launched:" "$N_RANDOM"
printf "  %-32s %d\n" "Trained runs launched:""$N_TRAINED"
printf "  %-32s %d\n" "Failed runs:"          "$N_FAILED"
echo ""
log_info "Results → WandB project: $WANDB_PROJECT"
echo ""

# Exit non-zero if any run failed, so CI/cluster job schedulers can detect failures
[ "$N_FAILED" -gt 0 ] && exit 1
exit 0