# structured3d semseg 200 epoch : LW-C
# bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c -n structured3d/semseg-lw-c

# structured3d semseg 200 epoch : Teacher: Small ts
# bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c-kd-ts -n structured3d/semseg-lw-c-kd-ts

# structured3d semseg 200 epoch : Teacher: base tb
# bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c-kd-tb -n structured3d/semseg-lw-c-kd-tb

# structured3d semseg 200 epoch : Teacher: Large tl
# bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c-kd-tl -n structured3d/semseg-lw-c-kd-tl

# TEST
# bash scripts/test.sh -d structured3d -n semseg-lw-c-kd-ts -w exp/structured3d/semseg-lw-c-kd-ts/model/model_best.pth

# The student backbone in semseg-lw-c-kd-ts.py is architecturally identical to semseg-lw-c.py. Just point -c at the lighter config and load the KD checkpoint:
bash scripts/test.sh \
  -d structured3d \
  -c semseg-lw-c \
  -n semseg-lw-c-kd-ts \
  -w exp/structured3d/semseg-lw-c-kd-ts/model/model_best.pth

# This loads DefaultSegmentorV2 (student only) with the same weights — the teacher keys simply won't exist in the checkpoint and will be ignored (assuming your CheckpointLoader handles missing/extra keys gracefully, which is standard).