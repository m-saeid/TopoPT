#!/usr/bin/env bash
# =============================================================================
# scripts/run_strategy1.sh  —  Strategy 1: Lightening the LitePT backbone
# =============================================================================
#
# ─── ABLATION DESIGN (6 configs, 3 independent axes) ─────────────────────────
#
# *** PointROPE CUDA constraint ***
# The PointROPE kernel requires head_dim % 6 == 0 (3D RoPE needs pairs for
# x,y,z axes).  The original uses head_dim=18 everywhere (18%6=0 ✓).
# All reduced-channel configs use channels = 18×k so head_dim=18 is preserved.
#   lw-b,c,e: (36,54,108,180,360) = 18×(2,3,6,10,20)  head_dim=18 ✓
#
#  Row  │ Config  │ Depth (D) │ Channels (C)               │ Patch (P)        │ GFLOPs│ back_out
#  ─────┼─────────┼───────────┼────────────────────────────┼──────────────────┼───────┼────────
#  S0★  │ rerun   │ 2,2,2,6,2 │ 36,72,144,252,504          │ 1024×5           │ 25.42 │ 72
#  S1   │ lw-a    │ 2,2,2,4↓,2│ 36,72,144,252,504          │ 1024×5           │ 20.63 │ 72
#  S2   │ lw-b    │ 2,2,2,6,2 │ 36,54↓,108↓,180↓,360↓     │ 1024×5           │ 15.78 │ 54
#  S3   │ lw-d    │ 2,2,2,6,2 │ 36,72,144,252,504          │ 1024×3,512↓×2    │ 22.04 │ 72
#  S4   │ lw-c    │ 2,2,2,4↓,2│ 36,54↓,108↓,180↓,360↓     │ 1024×5           │ 12.95 │ 54
#  S5★  │ lw-e    │ 2,2,2,4↓,2│ 36,54↓,108↓,180↓,360↓     │ 1024×3,512↓×2    │ 11.22 │ 54
#
#  GFLOPs measured by profiler at n_input=60 000 tokens (ScanNet typical scene).
#  ↓ = changed vs S0.  ★ = key starred results for the paper.
#
# ─── KEY COMPARISONS ─────────────────────────────────────────────────────────
#
#  S0 → S1:  Depth cut alone       (−4.8 GFLOPs, same params at attn stages)
#  S0 → S2:  Channel cut alone     (−9.6 GFLOPs, fewer attn params ∝C²)
#  S0 → S3:  Patch cut alone       (−3.4 GFLOPs, ZERO parameter change)
#  S1 vs S2:  Which axis costs more mIoU per FLOP saved?
#  S4 vs (S1+S2): Are D and C reductions additive?
#  S4 → S5:  Incremental patch cost on already-reduced model
#  S0 → S5:  Full combined efficiency: −56% GFLOPs
#
# ─── GPU ASSIGNMENT (2 machines × 3 runs = 6 total) ──────────────────────────
#
#  M1 (RTX 5090, 32 GB):    S0★ baseline → S3 patch-only → S5★ all-combined  (~22 h/run)
#  M2 (RTX 3090 #1, 24 GB): S1 LW-A      → S2 LW-B       → S4 LW-C          (~30 h/run)
#
#  NOTE: 3090 #2 removed. S2 and S3 redistributed:
#    S3 (22.04 GFLOPs, params=S0) → M1 (5090): full channel/depth budget fits 32 GB
#    S2 (~24M params, 15.78 GFLOPs) → M2 (3090): lightest config, fits 24 GB
#
# ─── USAGE ───────────────────────────────────────────────────────────────────
#  bash scripts/run_strategy1.sh machine1
#  bash scripts/run_strategy1.sh machine2
#  bash scripts/run_strategy1.sh profile           # random weights, all configs
#  bash scripts/run_strategy1.sh profile_trained   # trained checkpoints
#  bash scripts/run_strategy1.sh table             # print mIoU results
#  bash scripts/run_strategy1.sh check             # verify all files
# =============================================================================

set -euo pipefail
cd "$(dirname "$(dirname "$0")")" || exit 1

MACHINE=${1:-"machine1"}
GPU=1
DATASET="scannet"
PYTHON=python

RED='\033[0;31m'
GRN='\033[0;32m'
YEL='\033[1;33m'
BLU='\033[0;34m'
CYN='\033[0;36m'
NC='\033[0m'

banner() {
    local msg="$1"
    local w=72
    echo ""
    printf "╔%${w}s╗\n" " " | tr ' ' '='
    printf "║  %-$((w - 3))s║\n" "$msg"
    printf "║  %-$((w - 3))s║\n" "$(date '+%Y-%m-%d %H:%M')  machine=${MACHINE}"
    printf "╚%${w}s╝\n" " " | tr ' ' '='
    echo ""
}

run_exp() {
    local CFG="$1" NAME="$2" ROW="$3" EXPECTED="$4" CLAIM="$5"
    local cfg_path="configs/${DATASET}/${CFG}.py"

    echo ""
    echo -e "${BLU}========================================================${NC}"
    echo -e "${GRN}  RUN: ${YEL}${NAME}${NC}"
    echo -e "${BLU}  Config  : ${NC}${cfg_path}"
    echo -e "${BLU}  Row     : ${NC}${ROW}"
    echo -e "${BLU}  Expected: ${NC}${EXPECTED}"
    echo -e "${BLU}  Proves  : ${NC}${CLAIM}"
    echo -e "${BLU}  Started : ${NC}$(date '+%H:%M:%S')"
    echo -e "${BLU}========================================================${NC}"

    if [ ! -f "$cfg_path" ]; then
        echo -e "${RED}  ERROR: Config not found: ${cfg_path}${NC}"
        return 1
    fi

    bash scripts/train.sh -g "$GPU" -d "$DATASET" -c "$CFG" -n "$NAME"
    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo -e "${GRN}  DONE: ${NAME}  ($(date '+%H:%M:%S'))${NC}"
        local best_ckpt="exp/${DATASET}/${NAME}/model/model_best.pth"
        if [ -f "$best_ckpt" ]; then
            echo -e "${CYN}  Auto-profiling trained checkpoint ...${NC}"
            bash scripts/profile.sh -c "$CFG" -w "$best_ckpt" \
                -n "${NAME}-trained" -j "LitePT-Profiling" || true
        fi
    else
        echo -e "${RED}  FAILED: ${NAME}  (exit ${exit_code})${NC}"
        return "$exit_code"
    fi
}

run_profile_random() {
    local CFG="$1" NAME="$2"
    echo -e "${CYN}  Profile (random weights): ${NAME}${NC}"
    bash scripts/profile.sh -c "$CFG" -n "${NAME}-random" -j "LitePT-Profiling" || true
}

run_profile_trained() {
    local CFG="$1" NAME="$2"
    local best_ckpt="exp/${DATASET}/${NAME}/model/model_best.pth"
    if [ -f "$best_ckpt" ]; then
        echo -e "${CYN}  Profile (trained): ${NAME}${NC}"
        bash scripts/profile.sh -c "$CFG" -w "$best_ckpt" \
            -n "${NAME}-trained" -j "LitePT-Profiling" || true
    else
        echo -e "${YEL}  Checkpoint not found: ${best_ckpt} — skipping${NC}"
    fi
}


# =============================================================================
#  MACHINE 1 — RTX 5090  (32 GB, ~22 h/run)
#  S0★ (baseline) → S3 (patch-only, zero param change) → S5★ (all-combined)
#
#  S3 moved here from the removed 3090 #2.
#  Rationale: S3 keeps S0's full channel/depth budget (22.04 GFLOPs, params=S0)
#  so it benefits most from the 5090's memory headroom.
# =============================================================================
machine1() {
    banner "Machine 1 — RTX 5090 (32 GB, ~22 h/run): S0★ → S3 → S5★"

    run_profile_random "semseg-litept-rerun" "s1/litept-rerun"
    run_profile_random "semseg-litept-lw-d"  "s1/litept-lw-d"
    run_profile_random "semseg-litept-lw-e"  "s1/litept-lw-e"

    run_exp "semseg-litept-rerun" "s1/litept-rerun" \
        "S0★ baseline" \
        "~74.5–75.5% mIoU  (paper avg=75.4%; single-run typically 0.5–1% below)" \
        "Establishes the true single-run baseline for all ablation comparisons."

    run_exp "semseg-litept-lw-d" "s1/litept-lw-d" \
        "S3 patch-only" \
        "~73.5–75.0% mIoU at ~0.87× GFLOPs  (params IDENTICAL to S0!)" \
        "Zero-param change. Pure attention-window effect — unique ablation data point."

    run_exp "semseg-litept-lw-e" "s1/litept-lw-e" \
        "S5★ all-combined" \
        "~71–73% mIoU at ~0.44× baseline GFLOPs  (~0.65× params)" \
        "Maximum efficiency: D+C+P combined. Becomes base for all idea experiments."

    banner "Machine 1 complete"
}


# =============================================================================
#  MACHINE 2 — RTX 3090 #1  (24 GB, ~30 h/run)
#  S1 (depth only) → S2 (channel only) → S4 (depth+channel)
#
#  S2 moved here from the removed 3090 #2.
#  Rationale: S2 is the lightest config on this machine (~24M params,
#  15.78 GFLOPs), fits comfortably within 24 GB.
# =============================================================================
machine2() {
    banner "Machine 2 — RTX 3090 #1 (24 GB, ~30 h/run): S1 LW-A → S2 LW-B → S4 LW-C"

    run_profile_random "semseg-litept-lw-a" "s1/litept-lw-a"
    run_profile_random "semseg-litept-lw-b" "s1/litept-lw-b"
    run_profile_random "semseg-litept-lw-c" "s1/litept-lw-c"

    run_exp "semseg-litept-lw-a" "s1/litept-lw-a" \
        "S1 depth-only" \
        "~74.2–75.2% mIoU at ~0.81× GFLOPs" \
        "Isolates depth-cut cost. S0→S1 = pure effect of removing 2 stage-3 blocks."

    run_exp "semseg-litept-lw-b" "s1/litept-lw-b" \
        "S2 channel-only" \
        "~72–74% mIoU at ~0.62× GFLOPs" \
        "Channel cut alone. Compare with S1 (same GFLOPs~0.81): which axis costs mIoU more?"

    run_exp "semseg-litept-lw-c" "s1/litept-lw-c" \
        "S4 depth+channel" \
        "~71.5–73.5% mIoU at ~0.51× GFLOPs" \
        "D+C combined. Compare S4_drop vs (S1_drop + S2_drop) for additivity test."

    banner "Machine 2 complete"
}


# =============================================================================
#  PROFILE — all 6 configs with random weights (run on EACH machine)
# =============================================================================
profile() {
    banner "Pre-training profiling — random / untrained weights"
    echo "Run on EVERY machine before training starts."
    echo "WandB 'LitePT-Profiling' will have one entry per config per machine."
    echo ""

    for entry in \
        "semseg-litept-rerun:s1/litept-rerun" \
        "semseg-litept-lw-a:s1/litept-lw-a"  \
        "semseg-litept-lw-b:s1/litept-lw-b"  \
        "semseg-litept-lw-d:s1/litept-lw-d"  \
        "semseg-litept-lw-c:s1/litept-lw-c"  \
        "semseg-litept-lw-e:s1/litept-lw-e"
    do
        IFS=':' read -r cfg name <<< "$entry"
        run_profile_random "$cfg" "$name"
    done

    banner "Pre-training profiling complete"
}


# =============================================================================
#  PROFILE_TRAINED — all 6 configs with best checkpoints
# =============================================================================
profile_trained() {
    banner "Post-training profiling — best checkpoints"

    for entry in \
        "semseg-litept-rerun:s1/litept-rerun" \
        "semseg-litept-lw-a:s1/litept-lw-a"  \
        "semseg-litept-lw-b:s1/litept-lw-b"  \
        "semseg-litept-lw-d:s1/litept-lw-d"  \
        "semseg-litept-lw-c:s1/litept-lw-c"  \
        "semseg-litept-lw-e:s1/litept-lw-e"
    do
        IFS=':' read -r cfg name <<< "$entry"
        run_profile_trained "$cfg" "$name"
    done

    banner "Post-training profiling complete"
}


# =============================================================================
#  CHECK — verify all files exist
# =============================================================================
check() {
    banner "File verification"
    local all_ok=true

    echo "Strategy 1 configs (6 ablation points):"
    for cfg in \
        "semseg-litept-rerun" \
        "semseg-litept-lw-a"  \
        "semseg-litept-lw-b"  \
        "semseg-litept-lw-d"  \
        "semseg-litept-lw-c"  \
        "semseg-litept-lw-e"
    do
        local path="configs/${DATASET}/${cfg}.py"
        if [ -f "$path" ]; then
            echo -e "  ${GRN}OK${NC}  $path"
        else
            echo -e "  ${RED}MISSING${NC}  $path"
            all_ok=false
        fi
    done

    echo ""
    echo "Required scripts:"
    for f in "scripts/train.sh" "scripts/profile.sh" "tools/profile_model.py"; do
        if [ -f "$f" ]; then
            echo -e "  ${GRN}OK${NC}  $f"
        else
            echo -e "  ${RED}MISSING${NC}  $f"
            all_ok=false
        fi
    done

    echo ""
    if $all_ok; then
        echo -e "${GRN}All files present. Ready to run.${NC}"
        echo ""
        echo "WandB login tip: run 'wandb login' once in terminal before profiling."
        echo ""
        echo "Recommended first step (on EACH machine):"
        echo "  bash scripts/run_strategy1.sh profile"
        echo ""
        echo "Then training in parallel:"
        echo "  M1 (5090):    bash scripts/run_strategy1.sh machine1   # S0★, S3, S5★"
        echo "  M2 (3090 #1): bash scripts/run_strategy1.sh machine2   # S1, S2, S4"
    else
        echo -e "${RED}Missing files — run from LitePT repo root.${NC}"
        exit 1
    fi
}


# =============================================================================
#  TABLE — parse training logs and print current mIoU results
# =============================================================================
table() {
    banner "Strategy 1 — Results Table"
    local BASE="exp/${DATASET}"

    best_val() {
        local log="${BASE}/$1/train.log"
        [ -f "$log" ] && grep -oP 'mIoU/mAcc/allAcc \K[0-9.]+' "$log" 2>/dev/null \
            | sort -n | tail -1 || echo "-"
    }
    best_test() {
        local log="${BASE}/$1/test.log"
        [ -f "$log" ] && grep -oP 'mIoU/mAcc/allAcc \K[0-9.]+' "$log" 2>/dev/null \
            | tail -1 || echo "-"
    }
    epoch_count() {
        local log="${BASE}/$1/train.log"
        [ -f "$log" ] && grep -c "Train result" "$log" 2>/dev/null || echo "0"
    }

    print_row() {
        local row="$1" exp="$2" desc="$3" params="$4" gflops="$5" machine="$6"
        printf "%-5s  %-22s  %-17s  %-8s  %-8s  %-6s  %-9s  %-9s  ep=%s\n" \
            "$row" "$exp" "$desc" "$params" "$gflops" "$machine" \
            "$(best_val "$exp")" "$(best_test "$exp")" "$(epoch_count "$exp")"
    }

    printf "\n"
    printf "%-5s  %-22s  %-17s  %-8s  %-8s  %-6s  %-9s  %-9s\n" \
        "Row" "Exp" "Changes" "~Params" "~GFLOPs" "GPU" "val_mIoU" "test_mIoU"
    printf "%-5s  %-22s  %-17s  %-8s  %-8s  %-6s  %-9s  %-9s\n" \
        "---" "---" "-------" "-------" "-------" "---" "--------" "---------"

    # GFLOPs from profiler (n_input=60000, corrected for head_dim=18 channels)
    print_row "S0*"  "s1/litept-rerun"  "baseline"         "~37M"  "25.42"  "5090"
    print_row "S1"   "s1/litept-lw-a"   "D only (6→4)"     "~34M"  "20.63"  "3090"
    print_row "S2"   "s1/litept-lw-b"   "C only (×0.75)"   "~24M"  "15.78"  "3090"
    print_row "S3"   "s1/litept-lw-d"   "P only (→512)"    "=S0"   "22.04"  "5090"
    print_row "S4"   "s1/litept-lw-c"   "D+C"              "~21M"  "12.95"  "3090"
    print_row "S5*"  "s1/litept-lw-e"   "D+C+P (all)"      "~21M"  "11.22"  "5090"

    echo ""
    echo "Key comparisons:"
    echo "  S0→S1: cost of depth cut only               (−4.8 GFLOPs)"
    echo "  S0→S2: cost of channel cut only             (−9.6 GFLOPs)"
    echo "  S0→S3: cost of patch halving only           (−3.4 GFLOPs, params=S0)"
    echo "  S4 drop vs (S1+S2 drops): additivity of D and C?"
    echo "  S4→S5: incremental patch cost on reduced model"
    echo ""
    echo "Decision rule:"
    echo "  S5 within 3% of S0 → use S5 (lw-e) as base for all idea experiments."
    echo "  S5 drop 3–4%       → use S4 (lw-c) instead."
    echo ""
}


# =============================================================================
#  DISPATCH
# =============================================================================
case "$MACHINE" in
    machine1)        machine1        ;;
    machine2)        machine2        ;;
    profile)         profile         ;;
    profile_trained) profile_trained ;;
    table)           table           ;;
    check)           check           ;;
    *)
        echo ""
        echo "Usage: bash scripts/run_strategy1.sh [COMMAND]"
        echo ""
        echo "  machine1         RTX 5090  (32 GB): S0★ baseline → S3 patch → S5★ all"
        echo "  machine2         3090 #1   (24 GB): S1 LW-A (D) → S2 LW-B (C) → S4 LW-C (D+C)"
        echo "  profile          Profile all 6 configs with random weights"
        echo "  profile_trained  Profile all 6 configs with trained checkpoints"
        echo "  table            Print mIoU table from logs"
        echo "  check            Verify all files exist"
        echo ""
        exit 1
        ;;
esac

banner "Done — $(date)"