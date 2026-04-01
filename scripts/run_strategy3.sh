#!/usr/bin/env bash
# =============================================================================
# scripts/run_strategy3.sh  —  Strategy 3: LaRA3D-V2 (Perceiver-style)
# =============================================================================
#
# ─── BACKGROUND ───────────────────────────────────────────────────────────────
#
# Strategy-3 (LaRA3D-V1, mean-pool) FAILED:
#   lw-c + LaRA3D-V1 (G=32):  73.99% test  (−0.96% vs lw-c 74.95%)
#   lw-e + LaRA3D-V1 (G=32):  CRASHED at epoch 99 (training diverged)
#
# Root cause: mean-pool compression in Step 2 destroys boundary token
# information critical for per-point cross-entropy loss in segmentation.
# See models/litept/group_latent_attn.py for full diagnosis.
#
# ─── STRATEGY-3 FIX ──────────────────────────────────────────────────────────
#
# GroupLatentAttentionV2 (Perceiver-style) replaces mean-pool with:
#   Step 2: Learned latents ATTEND TO tokens via cross-attention
#           (no compression, content-adaptive aggregation)
#   Step 4: Gated additive injection (gate starts at 0.0 → safe initialisation)
#
# Tokens are NEVER compressed. Per-token gradient flow is clean.
# Gate_init=0.0 ensures stable start (begins as flat attention baseline).
#
# ─── EXPERIMENT TABLE ──────────────────────────────────────────────────────────
#
#  Row   │ Config                          │ Changes         │~Params│~GFLOPs│ Target
#  ──────┼─────────────────────────────────┼─────────────────┼───────┼───────┼──────────────
#  S0★   │ rerun  (baseline)               │ —               │12.71M │ 25.42 │ 75.27%
#  S4    │ lw-c   (D+C)                    │ —               │ 5.84M │ 12.95 │ 74.95%
#  S5★   │ lw-e   (D+C+P)                  │ —               │ 5.84M │ 11.22 │ 74.10%
#  S8-V1 │ lw-c + LaRA3D-V1(G=32,mean-pool)│ FAILED (−0.96%) │ 7.02M │ 12.95 │ (DONE)
#  S6-V1 │ lw-e + LaRA3D-V1(G=32,mean-pool)│ CRASHED ep99    │ 7.02M │ 11.22 │ (DONE)
#  ──────────────────────────────────────────────────────────────────────────────
#  S9★   │ lw-c + LaRA3D-V2(G=32,n_lat=1) │ V2 key generality│~7.1M │ 12.95 │ ≥74.95%
#  S6★   │ lw-e + LaRA3D-V2(G=32,n_lat=1) │ V2 KEY RESULT    │~7.1M │ 11.22 │ >74.95%
#  S10   │ lw-e + LaRA3D-V2(G=32,n_lat=2) │ ablation n_lat   │~7.4M │ 11.22 │ ~S6★±0.2%
#
# KEY PAPER CLAIM (if S6★ mIoU > S4):
#   "LaRA3D-V2 with small patches (P=512) outperforms flat attention with
#    large patches (P=1024), achieving higher accuracy at 56% fewer GFLOPs."
#
# ─── GPU ASSIGNMENT ───────────────────────────────────────────────────────────
#
#   M1 (RTX 5090, 32 GB):   S6★ lw-e+LaRA3DV2(G=32)  [~28h, KEY RESULT]
#   M2 (RTX 3090, 24 GB):   S9  lw-c+LaRA3DV2(G=32)  [~20h, GENERALITY]
#   Then (after M2 free):   S10 lw-e+LaRA3DV2(lat2)  [~28h, ABLATION — optional]
#
# ─── USAGE ────────────────────────────────────────────────────────────────────
#   bash scripts/run_strategy3.sh machine1     # RTX 5090: S6★ (KEY)
#   bash scripts/run_strategy3.sh machine2     # RTX 3090: S9  (GENERALITY)
#   bash scripts/run_strategy3.sh machine1_lat2  # S10 ablation (optional)
#   bash scripts/run_strategy3.sh profile      # profile all V2 configs
#   bash scripts/run_strategy3.sh table        # print results (S1–S10)
#   bash scripts/run_strategy3.sh check        # verify files exist
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
    local w=76
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
    local ec=$?

    if [ $ec -eq 0 ]; then
        echo -e "${GRN}  DONE: ${NAME}  ($(date '+%H:%M:%S'))${NC}"
        local ckpt="exp/${DATASET}/${NAME}/model/model_best.pth"
        if [ -f "$ckpt" ]; then
            echo -e "${CYN}  Auto-profiling ...${NC}"
            bash scripts/profile.sh -c "$CFG" -w "$ckpt" \
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
    bash scripts/profile.sh -c "$CFG" -n "${NAME}-random" -j "LitePT-Profiling" || true
}


# =============================================================================
#  MACHINE 1 — RTX 5090 (32 GB, ~28 h)
#  S6★: lw-e + LaRA3D-V2(G=32, n_lat=1)  — THE KEY PAPER RESULT
# =============================================================================
machine1() {
    banner "Machine 1 — RTX 5090 (32 GB, ~28 h): S6★ lw-e+LaRA3DV2(G=32)"

    run_profile_random "semseg-litept-lw-e-lara3dv2-g32" "s2b/litept-lw-e-lara3dv2-g32"

    run_exp \
        "semseg-litept-lw-e-lara3dv2-g32" \
        "s2b/litept-lw-e-lara3dv2-g32" \
        "S6★  lw-e + LaRA3D-V2 (G=32, n_lat=1, K=16)  [KEY PAPER RESULT]" \
        "> 74.95% mIoU (exceeding lw-c) at 11.22 GFLOPs (vs lw-c at 12.95)" \
        "V2 Perceiver-style restores global context lost by P=512 patch reduction.
  If mIoU > 74.95%: LaRA3D-V2 with small patches beats flat with large patches.
  This is the primary NeurIPS claim.
  If 74.10% < mIoU < 74.95%: V2 partially recovers the −1.10% patch penalty.
  Compare with V1 (CRASHED): training stability is the key V2 improvement."

    banner "Machine 1 (S6★) complete"
}


# =============================================================================
#  MACHINE 1 LAT2 — RTX 5090 (run after machine1 completes, or in parallel)
#  S10: lw-e + LaRA3D-V2(G=32, n_lat=2) — num_latents ablation
# =============================================================================
machine1_lat2() {
    banner "Machine 1 — RTX 5090: S10 lw-e+LaRA3DV2(G=32, n_lat=2) [ablation]"
    echo -e "${YEL}  NOTE: Run this only if S6★ shows good results (mIoU > 74.5%).${NC}"
    echo -e "${YEL}  S10 tests whether 2 latents per group beats 1 latent.${NC}"
    echo ""

    run_profile_random "semseg-litept-lw-e-lara3dv2-g32-lat2" "s2b/litept-lw-e-lara3dv2-g32-lat2"

    run_exp \
        "semseg-litept-lw-e-lara3dv2-g32-lat2" \
        "s2b/litept-lw-e-lara3dv2-g32-lat2" \
        "S10  lw-e + LaRA3D-V2 (G=32, n_lat=2, K=16)  [ablation]" \
        "~S6★ mIoU ± 0.2%  (whether 2 latents helps at K=16)" \
        "Isolates effect of latent count. n_lat=2 can represent bimodal groups
  (boundary regions spanning two classes). If significantly better than n_lat=1,
  use n_lat=2 as the default. Otherwise keep n_lat=1 for simplicity."

    banner "Machine 1 (S10 ablation) complete"
}


# =============================================================================
#  MACHINE 2 — RTX 3090 (24 GB, ~20 h)
#  S9: lw-c + LaRA3D-V2(G=32, n_lat=1) — generality test
# =============================================================================
machine2() {
    banner "Machine 2 — RTX 3090 (24 GB, ~20 h): S9 lw-c+LaRA3DV2(G=32)"

    run_profile_random "semseg-litept-lw-c-lara3dv2-g32" "s2b/litept-lw-c-lara3dv2-g32"

    run_exp \
        "semseg-litept-lw-c-lara3dv2-g32" \
        "s2b/litept-lw-c-lara3dv2-g32" \
        "S9  lw-c + LaRA3D-V2 (G=32, n_lat=1, K=32)  [GENERALITY]" \
        "≥ 74.95% mIoU with full patches P=1024, G=32, K=32" \
        "Tests V2 on full patches (no patch reduction). Two outcomes:
  (a) mIoU > lw-c (74.95%): V2 is intrinsically better than flat attention.
      Hierarchical local+global > global-only, even at large patch sizes.
  (b) mIoU ≈ lw-c (within 0.1%): V2 benefit is patch-reduction-specific.
  Compare with V1: lw-c+LaRA3D-V1 was 73.99% (−0.96%); V2 should be ≥ baseline."

    banner "Machine 2 (S9) complete"
}


# =============================================================================
#  PROFILE — all V2 configs with random weights
# =============================================================================
profile() {
    banner "Pre-training profiling — Strategy 2b V2 configs (random weights)"
    for entry in \
        "semseg-litept-lw-e-lara3dv2-g32:s2b/litept-lw-e-lara3dv2-g32"       \
        "semseg-litept-lw-c-lara3dv2-g32:s2b/litept-lw-c-lara3dv2-g32"       \
        "semseg-litept-lw-e-lara3dv2-g32-lat2:s2b/litept-lw-e-lara3dv2-g32-lat2"
    do
        IFS=':' read -r cfg name <<< "$entry"
        run_profile_random "$cfg" "$name"
    done
    banner "Pre-training profiling complete"
}


# =============================================================================
#  CHECK — verify all required files
# =============================================================================
check() {
    banner "Strategy 2b — File verification"
    local ok=true

    echo "New model files (must exist):"
    for f in \
        "models/litept/group_latent_attn.py" \
        "models/litept/litept.py"
    do
        if [ -f "$f" ]; then
            echo -e "  ${GRN}OK${NC}  $f"
        else
            echo -e "  ${RED}MISSING${NC}  $f"
            ok=false
        fi
    done

    echo ""
    echo "Check that litept.py contains GroupLatentAttentionV2:"
    if grep -q "GroupLatentAttentionV2" models/litept/litept.py 2>/dev/null; then
        echo -e "  ${GRN}OK${NC}  models/litept/litept.py  (GroupLatentAttentionV2 found)"
    else
        echo -e "  ${RED}MISSING${NC}  GroupLatentAttentionV2 not found in litept.py"
        ok=false
    fi

    echo ""
    echo "Check that group_latent_attn.py contains GroupLatentAttentionV2:"
    if grep -q "class GroupLatentAttentionV2" models/litept/group_latent_attn.py 2>/dev/null; then
        echo -e "  ${GRN}OK${NC}  group_latent_attn.py  (class GroupLatentAttentionV2 found)"
    else
        echo -e "  ${RED}MISSING${NC}  GroupLatentAttentionV2 not found in group_latent_attn.py"
        ok=false
    fi

    echo ""
    echo "New config files:"
    for cfg in \
        "semseg-litept-lw-e-lara3dv2-g32"      \
        "semseg-litept-lw-c-lara3dv2-g32"      \
        "semseg-litept-lw-e-lara3dv2-g32-lat2"
    do
        local path="configs/${DATASET}/${cfg}.py"
        if [ -f "$path" ]; then
            echo -e "  ${GRN}OK${NC}  $path"
        else
            echo -e "  ${RED}MISSING${NC}  $path"
            ok=false
        fi
    done

    echo ""
    if $ok; then
        echo -e "${GRN}All files present. Ready to launch Strategy-2b.${NC}"
        echo ""
        echo "Recommended launch order:"
        echo "  Simultaneously on both machines:"
        echo "    M1 (5090): bash scripts/run_strategy3.sh machine1"
        echo "    M2 (3090): bash scripts/run_strategy3.sh machine2"
        echo ""
        echo "After both complete (~1 day), check table:"
        echo "    bash scripts/run_strategy3.sh table"
        echo ""
        echo "If S6★ mIoU > 74.5%, optionally run:"
        echo "    M1 (5090): bash scripts/run_strategy3.sh machine1_lat2"
    else
        echo -e "${RED}Missing files — see above.${NC}"
        exit 1
    fi
}


# =============================================================================
#  TABLE — full results (Strategy 1 + 2 V1 + 2b V2)
# =============================================================================
table() {
    banner "Full Results — Strategy 1 + 2 (V1) + 2b (V2)"
    BASE="exp/${DATASET}"

    best_val()  { local l="${BASE}/$1/train.log"; [ -f "$l" ] && grep -oP 'mIoU/mAcc/allAcc \K[0-9.]+' "$l" 2>/dev/null | sort -n | tail -1 || echo "-"; }
    best_test() { local l="${BASE}/$1/test.log";  [ -f "$l" ] && grep -oP 'mIoU/mAcc/allAcc \K[0-9.]+' "$l" 2>/dev/null | tail -1  || echo "-"; }
    prow() { printf "%-6s  %-35s  %-8s  %-8s  %-9s  %-9s\n" "$1" "$2" "$3" "$4" "$(best_val "$5")" "$(best_test "$5")"; }

    printf "\n"
    printf "%-6s  %-35s  %-8s  %-8s  %-9s  %-9s\n" "Row" "Experiment" "~Params" "~GFLOPs" "val_mIoU" "test_mIoU"
    printf "%s\n" "$(printf '%0.s─' {1..88})"

    echo "── Strategy 1 (completed) ──────────────────────────────────────────────────────────"
    prow "S0★"    "litept-rerun  (baseline)"               "12.71M" "25.42" "s1/litept-rerun"
    prow "S1"     "litept-lw-a   (D only)"                 "11.18M" "20.63" "s1/litept-lw-a"
    prow "S2"     "litept-lw-b   (C only)"                  "6.63M" "15.78" "s1/litept-lw-b"
    prow "S3"     "litept-lw-d   (P only)"                 "12.71M" "22.04" "s1/litept-lw-d"
    prow "S4"     "litept-lw-c   (D+C)"                     "5.84M" "12.95" "s1/litept-lw-c"
    prow "S5★"    "litept-lw-e   (D+C+P)"                   "5.84M" "11.22" "s1/litept-lw-e"

    echo "── Strategy 2 V1 (FAILED — mean-pool) ─────────────────────────────────────────────"
    prow "S8-V1"  "lw-c+LaRA3D-V1(G32,mean-pool)"           "7.02M" "12.95" "s2/litept-lw-c-lara3d-g32"
    prow "S6-V1"  "lw-e+LaRA3D-V1(G32,mean-pool) CRASHED"   "7.02M" "11.22" "s2/litept-lw-e-lara3d-g32"

    echo "── Strategy 2b V2 (Perceiver — RUNNING) ────────────────────────────────────────────"
    prow "S9"     "lw-c+LaRA3D-V2(G32,n_lat=1) [generality]" "~7.1M" "12.95" "s2b/litept-lw-c-lara3dv2-g32"
    prow "S6★"    "lw-e+LaRA3D-V2(G32,n_lat=1) [KEY RESULT]" "~7.1M" "11.22" "s2b/litept-lw-e-lara3dv2-g32"
    prow "S10"    "lw-e+LaRA3D-V2(G32,n_lat=2) [ablation]"   "~7.4M" "11.22" "s2b/litept-lw-e-lara3dv2-g32-lat2"

    echo ""
    echo "Key comparisons for paper:"
    echo "  V1→V2 improvement:  S8-V1 (73.99%) vs S9 (≥74.95%?):  mean-pool vs Perceiver"
    echo "  V1→V2 improvement:  S6-V1 (CRASH)  vs S6★ (≥74.10%?): training stability"
    echo "  Main claim:         S6★ vs S4 (74.95%): V2+small-patch vs flat+big-patch"
    echo "  Latent ablation:    S6★ (n_lat=1) vs S10 (n_lat=2)"
    echo ""
    echo "Decision matrix:"
    printf "  %-40s %-s\n" "S6★ mIoU > 74.95% (beats S4):" "→ STRONG paper: V2+P512 > flat+P1024"
    printf "  %-40s %-s\n" "S6★ mIoU 74.10–74.95% (between S5,S4):" "→ GOOD: V2 recovers patch cut penalty"
    printf "  %-40s %-s\n" "S6★ mIoU < 74.10% (below S5):" "→ FALLBACK: write ablation paper (Strategy 1 only)"
    echo ""
}


# =============================================================================
#  DISPATCH
# =============================================================================
case "$MACHINE" in
    machine1)      machine1      ;;
    machine1_lat2) machine1_lat2 ;;
    machine2)      machine2      ;;
    profile)       profile       ;;
    table)         table         ;;
    check)         check         ;;
    *)
        echo ""
        echo "Usage: bash scripts/run_strategy3.sh [COMMAND]"
        echo ""
        echo "  machine1       RTX 5090: S6★ lw-e+LaRA3D-V2(G=32)  [KEY]"
        echo "  machine2       RTX 3090: S9  lw-c+LaRA3D-V2(G=32)  [GENERALITY]"
        echo "  machine1_lat2  RTX 5090: S10 lw-e+LaRA3D-V2(lat=2) [ABLATION, optional]"
        echo "  profile        Profile all V2 configs (random weights)"
        echo "  table          Print full results table (S0★–S10)"
        echo "  check          Verify all required files exist"
        echo ""
        exit 1
        ;;
esac

banner "Done — $(date)"