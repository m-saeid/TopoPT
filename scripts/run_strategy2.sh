#!/usr/bin/env bash
# =============================================================================
# scripts/run_strategy2.sh  —  Strategy 2: LaRA3D Group-Latent Attention
# =============================================================================
#
# ─── CONTRIBUTION ─────────────────────────────────────────────────────────────
#
# LaRA3D adapts LaRa's Group Attention Layer (ECCV 2024) to 3D point cloud
# semantic segmentation. LaRa unifies local+global reasoning by dividing a
# feature volume into spatial groups and applying attention within and across
# groups. We transfer this to serialized point cloud patches in LitePT:
#
#   LaRa (2D volumes):   divide → local attn within groups → global attn
#   LaRA3D (3D patches): divide → local attn (K tokens/group) → group latents
#                        → global self-attn (G groups) → broadcast back
#
# The key insight from Strategy-1: patch-size reduction (S3/lw-d) costs -1.10%
# mIoU, more than depth+channel combined (S4/lw-c, -0.32%). LaRA3D directly
# addresses this by restoring global context within small patches at O(G·K²+G²)
# cost vs O(P²) for flat attention.
#
# ─── STRATEGY-2 ABLATION TABLE ────────────────────────────────────────────────
#
# Builds on Strategy-1 results (all done):
#   S0★ rerun  : 75.27% test mIoU, 12.71M, 25.42G  (true baseline)
#   S1  lw-a   : 75.43%,           11.18M, 20.63G  (depth cut only)
#   S2  lw-b   : 74.89%,            6.63M, 15.78G  (channel only)
#   S3  lw-d   : 74.17%,           12.71M, 22.04G  (patch only) ← -1.10% gap!
#   S4  lw-c   : 74.95%,            5.84M, 12.95G  (depth+channel)
#   S5★ lw-e   : 74.10%,            5.84M, 11.22G  (all combined)
#
# Strategy-2 targets (new):
#   Row │ Config                   │ Changes        │~Params│~GFLOPs│ Target
#   ────┼──────────────────────────┼────────────────┼───────┼───────┼────────
#   S6★ │ lw-e + LaRA3D(G=32)     │ lw-e + grp-attn│ ~7.0M │ ~9.65 │ >74.95%
#   S7  │ lw-e + LaRA3D(G=64)     │ ablation G size │ ~7.0M │ ~9.5  │ ~74.6%
#   S8  │ lw-c + LaRA3D(G=32)     │ generality test │ ~7.0M │ ~10.5 │ >74.95%
#
# KEY CLAIM: S6★ should exceed S4 (lw-c, 74.95%) at fewer GFLOPs (9.65 vs 12.95)
# → LaRA3D with small patches (P=512) is better than flat attention with full
#   patches (P=1024), while being more efficient.
#
# ─── GPU ASSIGNMENT ──────────────────────────────────────────────────────────
#
#   M1 (RTX 5090, 32 GB):   S6★ lw-e+LaRA3D(G=32)    [KEY — 28h]
#   M2 (RTX 3090 #1, 24 GB): S8  lw-c+LaRA3D(G=32)   [GENERALITY — 18h]
#   M3 (RTX 3090 #2, 24 GB): S7  lw-e+LaRA3D(G=64)   [ABLATION — 28h]
#
#   After wave 1 (~1.25 days for M2, ~1.2 days for M1/M3):
#   M2 free → run additional ablation or ScanNet200 extension
#
# ─── USAGE ───────────────────────────────────────────────────────────────────
#   sh scripts/run_strategy2.sh machine1          # RTX 5090: S6★ (KEY)
#   sh scripts/run_strategy2.sh machine2          # 3090 #1: S8 (generality)
#   sh scripts/run_strategy2.sh machine3          # 3090 #2: S7 (G ablation)
#   sh scripts/run_strategy2.sh profile           # Profile all with random wts
#   sh scripts/run_strategy2.sh profile_trained   # Profile trained checkpoints
#   sh scripts/run_strategy2.sh table             # Print results table
#   sh scripts/run_strategy2.sh check             # Verify all files exist
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

    sh scripts/train.sh -g "$GPU" -d "$DATASET" -c "$CFG" -n "$NAME"
    local ec=$?

    if [ $ec -eq 0 ]; then
        echo -e "${GRN}  DONE: ${NAME}  ($(date '+%H:%M:%S'))${NC}"
        local ckpt="exp/${DATASET}/${NAME}/model/model_best.pth"
        if [ -f "$ckpt" ]; then
            echo -e "${CYN}  Auto-profiling...${NC}"
            sh scripts/profile.sh -c "$CFG" -w "$ckpt" \
                -n "${NAME}-trained" -j "LitePT-Profiling" || true
        fi
    else
        echo -e "${RED}  FAILED: ${NAME}  (exit ${ec})${NC}"
        return "$ec"
    fi
}

run_profile_random() {
    local CFG="$1" NAME="$2"
    echo -e "${CYN}  Profile (random): ${NAME}${NC}"
    sh scripts/profile.sh -c "$CFG" -n "${NAME}-random" -j "LitePT-Profiling" || true
}

run_profile_trained() {
    local CFG="$1" NAME="$2"
    local ckpt="exp/${DATASET}/${NAME}/model/model_best.pth"
    if [ -f "$ckpt" ]; then
        echo -e "${CYN}  Profile (trained): ${NAME}${NC}"
        sh scripts/profile.sh -c "$CFG" -w "$ckpt" \
            -n "${NAME}-trained" -j "LitePT-Profiling" || true
    else
        echo -e "${YEL}  No checkpoint: ${ckpt}${NC}"
    fi
}


# =============================================================================
#  MACHINE 1 — RTX 5090 (32 GB, ~28 h)
#  S6★: lw-e + LaRA3D(G=32) — the key result
# =============================================================================
machine1() {
    banner "Machine 1 — RTX 5090 (32 GB, ~28 h): S6★ lw-e+LaRA3D(G=32)"

    run_profile_random "semseg-litept-lw-e-lara3d-g32" "s2/litept-lw-e-lara3d-g32"

    run_exp \
        "semseg-litept-lw-e-lara3d-g32" \
        "s2/litept-lw-e-lara3d-g32" \
        "S6★  lw-e + LaRA3D(G=32)  [KEY PAPER RESULT]" \
        ">74.95% mIoU (exceeding lw-c) at ~9.65 GFLOPs (< lw-c's 12.95)" \
        "LaRA3D with G=32,K=16 groups recovers the -1.10% patch-reduction penalty. \
If mIoU > lw-c (74.95%) at lower GFLOPs, this is a strong NeurIPS result: \
LaRA3D with small patches dominates flat attention with large patches."

    banner "Machine 1 complete"
}


# =============================================================================
#  MACHINE 2 — RTX 3090 #1 (24 GB, ~18 h)
#  S8: lw-c + LaRA3D(G=32) — generality (helps even without patch reduction)
# =============================================================================
machine2() {
    banner "Machine 2 — RTX 3090 #1 (24 GB, ~18 h): S8 lw-c+LaRA3D(G=32)"

    run_profile_random "semseg-litept-lw-c-lara3d-g32" "s2/litept-lw-c-lara3d-g32"

    run_exp \
        "semseg-litept-lw-c-lara3d-g32" \
        "s2/litept-lw-c-lara3d-g32" \
        "S8  lw-c + LaRA3D(G=32)  [GENERALITY]" \
        "~75.0–75.3% mIoU with full patches P=1024, G=32, K=32" \
        "Tests LaRA3D on full patches (P=1024). If better than lw-c (74.95%), \
then LaRA3D's hierarchical reasoning is intrinsically better than flat global \
attention, independent of patch-size reduction."

    banner "Machine 2 complete"
}


# =============================================================================
#  MACHINE 3 — RTX 3090 #2 (24 GB, ~28 h)
#  S7: lw-e + LaRA3D(G=64) — ablation on group size
# =============================================================================
machine3() {
    banner "Machine 3 — RTX 3090 #2 (24 GB, ~28 h): S7 lw-e+LaRA3D(G=64)"

    run_profile_random "semseg-litept-lw-e-lara3d-g64" "s2/litept-lw-e-lara3d-g64"

    run_exp \
        "semseg-litept-lw-e-lara3d-g64" \
        "s2/litept-lw-e-lara3d-g64" \
        "S7  lw-e + LaRA3D(G=64)  [GROUP-SIZE ABLATION]" \
        "~74.5–74.8% mIoU at slightly lower GFLOPs than G=32" \
        "G=64 (K=8) vs G=32 (K=16): smaller groups, faster attention but less \
local context per group. S7 vs S6 isolates local-size sensitivity."

    banner "Machine 3 complete"
}


# =============================================================================
#  PROFILE — profile all 3 Strategy-2 configs (random weights)
# =============================================================================
profile() {
    banner "Pre-training profiling — random weights (Strategy 2)"
    for entry in \
        "semseg-litept-lw-e-lara3d-g32:s2/litept-lw-e-lara3d-g32" \
        "semseg-litept-lw-e-lara3d-g64:s2/litept-lw-e-lara3d-g64" \
        "semseg-litept-lw-c-lara3d-g32:s2/litept-lw-c-lara3d-g32"
    do
        IFS=':' read -r cfg name <<< "$entry"
        run_profile_random "$cfg" "$name"
    done
    banner "Pre-training profiling complete"
}

profile_trained() {
    banner "Post-training profiling — trained checkpoints (Strategy 2)"
    for entry in \
        "semseg-litept-lw-e-lara3d-g32:s2/litept-lw-e-lara3d-g32" \
        "semseg-litept-lw-e-lara3d-g64:s2/litept-lw-e-lara3d-g64" \
        "semseg-litept-lw-c-lara3d-g32:s2/litept-lw-c-lara3d-g32"
    do
        IFS=':' read -r cfg name <<< "$entry"
        run_profile_trained "$cfg" "$name"
    done
    banner "Post-training profiling complete"
}


# =============================================================================
#  CHECK — verify all files
# =============================================================================
check() {
    banner "Strategy 2 — File verification"
    local ok=true

    echo "New files required (Strategy 2):"
    for f in \
        "models/litept/group_latent_attn.py" \
        "configs/scannet/semseg-litept-lw-e-lara3d-g32.py" \
        "configs/scannet/semseg-litept-lw-e-lara3d-g64.py" \
        "configs/scannet/semseg-litept-lw-c-lara3d-g32.py"
    do
        if [ -f "$f" ]; then
            echo -e "  ${GRN}OK${NC}  $f"
        else
            echo -e "  ${RED}MISSING${NC}  $f"
            ok=false
        fi
    done

    echo ""
    echo "Modified file (must contain GroupLatentAttention + enc_attn_type param):"
    if grep -q "GroupLatentAttention" models/litept/litept.py 2>/dev/null; then
        echo -e "  ${GRN}OK${NC}  models/litept/litept.py  (GroupLatentAttention found)"
    else
        echo -e "  ${RED}MISSING${NC}  models/litept/litept.py does not contain GroupLatentAttention"
        echo "  → Apply the patch from litept_v2.py to litept.py"
        ok=false
    fi

    echo ""
    if $ok; then
        echo -e "${GRN}All files present. Ready to launch.${NC}"
    else
        echo -e "${RED}Missing files — see above.${NC}"
        exit 1
    fi
}


# =============================================================================
#  TABLE — full combined results (Strategy 1 + 2)
# =============================================================================
table() {
    banner "Combined Results — Strategy 1 + Strategy 2"
    BASE="exp/${DATASET}"

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

    prow() {
        printf "%-5s  %-30s  %-8s  %-8s  %-9s  %-9s\n" \
            "$1" "$2" "$3" "$4" "$(best_val "$5")" "$(best_test "$5")"
    }

    printf "\n"
    printf "%-5s  %-30s  %-8s  %-8s  %-9s  %-9s\n" \
        "Row" "Experiment" "~Params" "~GFLOPs" "val_mIoU" "test_mIoU"
    printf "%-5s  %-30s  %-8s  %-8s  %-9s  %-9s\n" \
        "---" "----------" "-------" "-------" "--------" "---------"

    echo "─── Strategy 1 (completed) ───────────────────────────────────────────────"
    prow "S0*"  "litept-rerun  (baseline)"  "12.71M"  "25.42"  "s1/litept-rerun"
    prow "S1"   "litept-lw-a   (D)"          "11.18M"  "20.63"  "s1/litept-lw-a"
    prow "S2"   "litept-lw-b   (C)"           "6.63M"  "15.78"  "s1/litept-lw-b"
    prow "S3"   "litept-lw-d   (P)"          "12.71M"  "22.04"  "s1/litept-lw-d"
    prow "S4"   "litept-lw-c   (D+C)"         "5.84M"  "12.95"  "s1/litept-lw-c"
    prow "S5*"  "litept-lw-e   (D+C+P)"       "5.84M"  "11.22"  "s1/litept-lw-e"

    echo "─── Strategy 2 (LaRA3D) ─────────────────────────────────────────────────"
    prow "S6*"  "lw-e+LaRA3D(G=32) [KEY]"     "~7.0M"  "~9.65"  "s2/litept-lw-e-lara3d-g32"
    prow "S7"   "lw-e+LaRA3D(G=64) [ablation]" "~7.0M"  "~9.5"   "s2/litept-lw-e-lara3d-g64"
    prow "S8"   "lw-c+LaRA3D(G=32) [general]"  "~7.0M"  "~10.5"  "s2/litept-lw-c-lara3d-g32"

    echo ""
    echo "Key comparisons for paper:"
    echo "  S5→S6: Does LaRA3D recover the -1.10% patch-reduction penalty?"
    echo "  S4 vs S6: Does LaRA3D with P=512 beat flat with P=1024 at lower GFLOPs?"
    echo "  S6 vs S7: Optimal group size (G=32 vs G=64, K=16 vs K=8)?"
    echo "  S4 vs S8: Does LaRA3D improve even without patch reduction?"
    echo ""
    echo "Success criterion for NeurIPS:"
    echo "  S6 mIoU > S4 (74.95%) at lower GFLOPs (9.65 vs 12.95) → strong result"
    echo "  S6 mIoU > S5 (74.10%) → at minimum, LaRA3D recovers patch-cut penalty"
    echo ""
}


# =============================================================================
#  DISPATCH
# =============================================================================
case "$MACHINE" in
    machine1)        machine1        ;;
    machine2)        machine2        ;;
    machine3)        machine3        ;;
    profile)         profile         ;;
    profile_trained) profile_trained ;;
    table)           table           ;;
    check)           check           ;;
    *)
        echo ""
        echo "Usage: sh scripts/run_strategy2.sh [COMMAND]"
        echo ""
        echo "  machine1         RTX 5090:  S6★ lw-e+LaRA3D(G=32)  [KEY]"
        echo "  machine2         3090 #1:   S8  lw-c+LaRA3D(G=32)  [GENERALITY]"
        echo "  machine3         3090 #2:   S7  lw-e+LaRA3D(G=64)  [GROUP ABLATION]"
        echo "  profile          Profile all 3 configs (random weights)"
        echo "  profile_trained  Profile with trained checkpoints"
        echo "  table            Print full combined results table (S1-S8)"
        echo "  check            Verify all required files exist"
        echo ""
        exit 1
        ;;
esac

banner "Done — $(date)"