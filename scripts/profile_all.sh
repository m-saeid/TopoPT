#!/usr/bin/env bash
# =============================================================================
# scripts/profile_all.sh  —  Batch profiler for all LitePT experiment configs
# =============================================================================
# CHANGES vs original:
#   [FIX-1] exp_dir uses flat hyphenated naming: exp/${dataset}-${config_stem}
#           (was: exp/${dataset}/${config_stem} — never matched any real path)
#   [FIX-2] All 11 config stems updated to match actual filenames in configs/
#   [FIX-3] KD trained runs now pass the base student config via 6th arg to
#           run_one(), so build_model() sees only the student — not the full
#           distillation graph with teacher+projectors.  Params, memory, and
#           latency then correctly reflect inference-time cost.
#   [FIX-4] N_FAILED incremented and printed in summary.
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
#  [FIX-2] All config stems now match actual filenames in configs/<dataset>/.
#          Verified against configs.txt.
# ══════════════════════════════════════════════════════════════════════════════
CONFIGS=(
    # ── ScanNet · Semantic Segmentation ───────────────────────────────────────
    "scannet|semseg-litept-small-v1m1|0|semseg,scannet,baseline,S0,no-decoder"
    "scannet|semseg-litept-small-v1m2|0|semseg,scannet,baseline,S0,full-decoder"
    # FIX-2: was semseg-litept-rerun → semseg-litept-rerun-100epoch
    "scannet|semseg-litept-rerun-100epoch|0|semseg,scannet,baseline,S0,rerun"

    # Ablation series (Strategy-1)
    # FIX-2: semseg-litept-lw-{a,b,d,e} → semseg-lw-{a,b,d,e}-100epoch
    "scannet|semseg-lw-a-100epoch|0|semseg,scannet,lw-a,S1,depth-only"
    "scannet|semseg-lw-b-100epoch|0|semseg,scannet,lw-b,S2,channel-only"
    "scannet|semseg-lw-d-100epoch|0|semseg,scannet,lw-d,S3,patch-only"
    "scannet|semseg-lw-c-100epoch|0|semseg,scannet,lw-c,S4,depth+channel,100ep"
    "scannet|semseg-lw-c-1200epoch|0|semseg,scannet,lw-c,S4,depth+channel,1200ep"
    "scannet|semseg-lw-e-100epoch|0|semseg,scannet,lw-e,S5,all-reductions"

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
    # FIX-2: semseg-lw-c → semseg-lw-c-50epoch
    "nuscenes|semseg-lw-c-50epoch|0|semseg,nuscenes,lw-c"
    # FIX-2: semseg-lw-c-kd → semseg-lw-c-kd-50epoch
    "nuscenes|semseg-lw-c-kd-50epoch|1|semseg,nuscenes,lw-c,kd,srfd"

    # ── Structured3D · Semantic Segmentation ──────────────────────────────────
    "structured3d|semseg-litept-base-v1m1|0|semseg,structured3d,baseline,base"
    "structured3d|semseg-litept-large-v1m1|0|semseg,structured3d,baseline,large"
    "structured3d|semseg-litept-small-v1m1|0|semseg,structured3d,baseline,small"
    # FIX-2: semseg-lw-c → semseg-lw-c-50epoch
    "structured3d|semseg-lw-c-50epoch|0|semseg,structured3d,lw-c"
    # FIX-2: kd stems now carry -50epoch suffix
    "structured3d|semseg-lw-c-kd-tb-50epoch|1|semseg,structured3d,lw-c,kd,srfd,teacher-base"
    "structured3d|semseg-lw-c-kd-tl-50epoch|1|semseg,structured3d,lw-c,kd,srfd,teacher-large"
    "structured3d|semseg-lw-c-kd-ts-50epoch|1|semseg,structured3d,lw-c,kd,srfd,teacher-small"
)

# ══════════════════════════════════════════════════════════════════════════════
#  [FIX-3] Map each KD config stem → its base student config stem
#
#  This is used so KD trained runs profile with the student-only config,
#  ensuring params/memory/latency reflect inference cost — not the full
#  distillation graph that includes teacher backbone + projectors.
# ══════════════════════════════════════════════════════════════════════════════
get_student_config() {
    local dataset="$1"
    local kd_stem="$2"
    local student_stem=""

    case "${dataset}|${kd_stem}" in
        "scannet|semseg-lw-c-kd-100epoch")   student_stem="semseg-lw-c-100epoch"     ;;
        "scannet|semseg-lw-c-kd-1200epoch")  student_stem="semseg-lw-c-1200epoch"    ;;
        "scannet|insseg-lw-c-kd-100epoch")   student_stem="insseg-lw-c-100epoch"     ;;
        "scannet|insseg-lw-c-kd-800epoch")   student_stem="insseg-lw-c-800epoch"     ;;
        "scannet200|insseg-lw-c-kd-100epoch") student_stem="insseg-lw-c-100epoch"    ;;
        "scannet200|insseg-lw-c-kd-800epoch") student_stem="insseg-lw-c-800epoch"    ;;
        "nuscenes|semseg-lw-c-kd-50epoch")   student_stem="semseg-lw-c-50epoch"      ;;
        "structured3d|semseg-lw-c-kd-tb-50epoch") student_stem="semseg-lw-c-50epoch" ;;
        "structured3d|semseg-lw-c-kd-tl-50epoch") student_stem="semseg-lw-c-50epoch" ;;
        "structured3d|semseg-lw-c-kd-ts-50epoch") student_stem="semseg-lw-c-50epoch" ;;
        *)
            log_warn "No student config mapping for ${dataset}/${kd_stem} — using KD config as fallback."
            echo "configs/${dataset}/${kd_stem}.py"
            return
            ;;
    esac
    echo "configs/${dataset}/${student_stem}.py"
}

# ══════════════════════════════════════════════════════════════════════════════
#  Helper: find best checkpoint in an experiment directory
#  Priority: model_best.pth > model_last.pth > (none)
# ══════════════════════════════════════════════════════════════════════════════
find_checkpoint() {
    local exp_dir="$1"
    local stem="$2"        # pass dataset-config_stem
    # Check standard location first
    for dir in "$exp_dir" "exp/litept/${stem}"; do
        local model_dir="${dir}/model"
        if   [ -f "${model_dir}/model_best.pth" ]; then
            echo "${model_dir}/model_best.pth"; return
        elif [ -f "${model_dir}/model_last.pth" ]; then
            echo "${model_dir}/model_last.pth"; return
        fi
    done
    echo ""
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
        # FIX-4: increment failure counter (must be done at call site in main
        # loop because shell functions run in a subshell-like scope for exports)
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

    if [ -n "$FILTER" ] && ! echo "${dataset}/${config_stem}" | grep -q "$FILTER"; then
        continue
    fi

    N_TOTAL=$((N_TOTAL + 1))
    config_path="configs/${dataset}/${config_stem}.py"

    # [FIX-1] Flat hyphenated exp directory — matches actual layout from exp.txt
    exp_dir="exp/${dataset}-${config_stem}"

    log_head "${dataset} / ${config_stem}"

    if [ ! -f "$config_path" ]; then
        log_warn "Config NOT FOUND: $config_path — skipping."
        N_SKIPPED=$((N_SKIPPED + 1))
        continue
    fi
    log_info "Config:     $config_path"

    trained_ckpt="$(find_checkpoint "$exp_dir")"
    if [ -n "$trained_ckpt" ]; then
        log_info "Checkpoint: $trained_ckpt"
    else
        log_info "Checkpoint: none found in $exp_dir/model/"
    fi

    # ── RANDOM profiling ───────────────────────────────────────────────────────
    if [[ "$MODE" == "random" || "$MODE" == "both" ]]; then
        if [ "$is_kd" -eq 1 ]; then
            log_info "[random] SKIP — KD config; inference arch = base student (already profiled)"
        else
            log_info "[random] Profiling random weights …"
            run_name="${dataset}/${config_stem}-random"
            run_one "$config_path" "$run_name" "" "${tags},random" 0 || N_FAILED=$((N_FAILED + 1))
            N_RANDOM=$((N_RANDOM + 1))
        fi
    fi

    # ── TRAINED profiling ──────────────────────────────────────────────────────
    if [[ "$MODE" == "trained" || "$MODE" == "both" ]]; then
        if [ -z "$trained_ckpt" ]; then
            log_info "[trained] No checkpoint — skipping trained run."
        else
            log_info "[trained] Profiling trained checkpoint …"
            run_name="${dataset}/${config_stem}-trained"

            if [ "$is_kd" -eq 1 ]; then
                # [FIX-3] Use student config so build_model() creates only the
                # student at inference time — correct params/memory/latency.
                log_info "[trained] KD mode: using student config + --no-strict-load."
                student_cfg="$(get_student_config "$dataset" "$config_stem")"
                log_info "[trained] Student config: $student_cfg"

                if [ ! -f "$student_cfg" ]; then
                    log_warn "Student config not found: $student_cfg — falling back to KD config."
                    student_cfg=""
                fi

                run_one \
                    "$config_path" \
                    "$run_name" \
                    "$trained_ckpt" \
                    "${tags},trained,kd-checkpoint" \
                    1 \
                    "$student_cfg" || N_FAILED=$((N_FAILED + 1))
            else
                run_one \
                    "$config_path" \
                    "$run_name" \
                    "$trained_ckpt" \
                    "${tags},trained" \
                    0 || N_FAILED=$((N_FAILED + 1))
            fi
            N_TRAINED=$((N_TRAINED + 1))
        fi
    fi

done  # ── end main loop ───────────────────────────────────────────────────────

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
# FIX-4: failure count now visible
printf "  %-30s %d\n" "Failed runs:"            "$N_FAILED"
echo ""
log_info "Results → WandB project: $WANDB_PROJECT"
echo ""