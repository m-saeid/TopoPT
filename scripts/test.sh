#!/bin/sh

# scripts/test.sh

# Flags & Examples: 
# Flag  Meaning Default
# -d  Dataset scannet
# -c  Config name (without .py) required
# -n  Experiment name debug
# -w  Weight path (.pth)  auto-detected
# -g  Number of GPUsauto -detected-mNumber of machines1
# -p  Python interpreterpython

# Smart auto-detection built in:
# If -w is not passed, it automatically finds model_best.pth → falls back to model_last.pth inside exp/<dataset>/<exp_name>/model/
# If -c is None, it loads the saved config.py from the experiment dir

# bash# Minimal — auto-finds config & best checkpoint
# > sh scripts/test.sh -d scannet -n semseg-litept-lw-e

# Explicit weight
# > sh scripts/test.sh -d scannet -n semseg-litept-lw-e \
#     -w exp/scannet/semseg-litept-lw-e/model/model_best.pth

# Specific config + 4 GPUs
# sh scripts/test.sh -d scannet -c semseg-litept-lw-e \
#   -n my_run -g 4

# Tip: Don't forget to chmod +x scripts/*.sh

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TEST_CODE=test.py

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
NUM_GPU=None
NUM_MACHINE=1
DIST_URL="auto"


while getopts "p:d:c:n:w:g:m:" opt; do
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
    g)
      NUM_GPU=$OPTARG
      ;;
    m)
      NUM_MACHINE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

# ── Auto-detect GPU count if not specified ────────────────────────────────────
if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"

# ── SLURM multi-node support ──────────────────────────────────────────────────
if [ -n "$SLURM_NODELIST" ]; then
  MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | awk '{ print $1 }')
  MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
  DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
fi

echo "Master addr: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Dist URL: $DIST_URL"

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT

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
    echo "[warn] No weight specified and none found in $MODEL_DIR. Testing with random weights."
  fi
fi

echo "Loading config in:" $CONFIG_DIR
echo "Loading weight from:" $WEIGHT
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"

echo " =========> RUN TEST <========="

if [ "${WEIGHT}" = "None" ]
then
    $PYTHON "$CODE_DIR"/tools/$TEST_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --options save_path="$EXP_DIR"
else
    $PYTHON "$CODE_DIR"/tools/$TEST_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank ${SLURM_NODEID:-0} \
    --dist-url ${DIST_URL} \
    --options save_path="$EXP_DIR" weight="$WEIGHT"
fi