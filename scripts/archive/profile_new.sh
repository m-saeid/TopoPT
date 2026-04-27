#!/bin/sh


# profile_model.sh — Flags & Examples
# FlagMeaningDefault-dDatasetscannet-cConfig namerequired-nExperiment namedebug-wWeight pathauto-detected-rWandB run nameauto-generated-jWandB projectLitePT-Profiling-kWandB API keyfrom wandb login-uWarmup iterations5-eMeasure iterations30-bProfile batches50-sSkip training memory (flag, no value)false
# bash# Random weights (architecture cost only — before training)
# sh scripts/profile_model.sh -d scannet -c semseg-litept-lw-e \
#   -n semseg-litept-lw-e

# Trained checkpoint (real-world cost)
# sh scripts/profile_model.sh -d scannet -n semseg-litept-lw-e

# Fast run — skip training memory measurement
# sh scripts/profile_model.sh -d scannet -n semseg-litept-lw-e -s

# Custom WandB project + explicit API key
# sh scripts/profile_model.sh -d scannet -n semseg-litept-lw-e \
#   -j MyProject -k YOUR_WANDB_KEY

# Tip: Don't forget to chmod +x scripts/*.sh


cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

PROFILE_CODE=profile_model.py

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RUN_NAME="None"
WANDB_PROJECT="LitePT-Profiling"
WANDB_KEY="None"
N_WARMUP=5
N_MEASURE=30
N_PROFILE_BATCHES=50
SKIP_TRAIN_MEM=false


while getopts "p:d:c:n:w:r:j:k:u:e:b:s" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RUN_NAME=$OPTARG
      ;;
    j)
      WANDB_PROJECT=$OPTARG
      ;;
    k)
      WANDB_KEY=$OPTARG
      ;;
    u)
      N_WARMUP=$OPTARG
      ;;
    e)
      N_MEASURE=$OPTARG
      ;;
    b)
      N_PROFILE_BATCHES=$OPTARG
      ;;
    s)
      SKIP_TRAIN_MEM=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

echo "Experiment name:  $EXP_NAME"
echo "Python:           $PYTHON"
echo "Dataset:          $DATASET"
echo "Config:           $CONFIG"
echo "WandB project:    $WANDB_PROJECT"
echo "WandB run name:   $RUN_NAME"
echo "N warmup:         $N_WARMUP"
echo "N measure:        $N_MEASURE"
echo "N profile batches:$N_PROFILE_BATCHES"
echo "Skip train mem:   $SKIP_TRAIN_MEM"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=./

# ── Resolve config: prefer saved config inside exp dir, fallback to configs/ ──
if [ "${CONFIG}" = "None" ] && [ -f "${EXP_DIR}/config.py" ]; then
  CONFIG_DIR=${EXP_DIR}/config.py
  echo "Auto-detected config from experiment dir: $CONFIG_DIR"
else
  CONFIG_DIR=configs/${DATASET}/${CONFIG}.py
fi

# ── Resolve weight: prefer model_best, fallback to model_last ─────────────────
if [ "${WEIGHT}" = "None" ]; then
  if [ -f "${MODEL_DIR}/model_best.pth" ]; then
    WEIGHT=${MODEL_DIR}/model_best.pth
    echo "Auto-detected weight: $WEIGHT (model_best)"
  elif [ -f "${MODEL_DIR}/model_last.pth" ]; then
    WEIGHT=${MODEL_DIR}/model_last.pth
    echo "Auto-detected weight: $WEIGHT (model_last)"
  else
    echo "No checkpoint found in $MODEL_DIR — profiling with random weights."
  fi
fi

# ── Auto-build WandB run name if not specified ─────────────────────────────────
if [ "${RUN_NAME}" = "None" ]; then
  if [ "${WEIGHT}" = "None" ]; then
    RUN_NAME="${DATASET}/${EXP_NAME}-random"
  else
    RUN_NAME="${DATASET}/${EXP_NAME}-trained"
  fi
  echo "Auto-generated WandB run name: $RUN_NAME"
fi

echo "Loading config in: $CONFIG_DIR"
echo "Loading weight from: $WEIGHT"
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"

echo " =========> RUN PROFILER <========="

# ── Build the base command ─────────────────────────────────────────────────────
CMD="$PYTHON ${CODE_DIR}tools/$PROFILE_CODE \
    --config-file $CONFIG_DIR \
    --run-name $RUN_NAME \
    --wandb-project $WANDB_PROJECT \
    --n-warmup $N_WARMUP \
    --n-measure $N_MEASURE \
    --n-profile-batches $N_PROFILE_BATCHES"

# ── Conditionally append optional flags ───────────────────────────────────────
if [ "${WEIGHT}" != "None" ]; then
  CMD="$CMD --weight $WEIGHT"
fi

if [ "${WANDB_KEY}" != "None" ]; then
  CMD="$CMD --wandb-key $WANDB_KEY"
fi

if [ "${SKIP_TRAIN_MEM}" = "true" ]; then
  CMD="$CMD --skip-train-memory"
fi

eval $CMD