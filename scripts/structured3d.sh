# structured3d semseg 200 epoch : LW-C
# bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c -n structured3d/semseg-lw-c

# structured3d semseg 200 epoch : Teacher: Small ts
# bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c-kd-ts -n structured3d/semseg-lw-c-kd-ts

# structured3d semseg 200 epoch : Teacher: base tb
# bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c-kd-tb -n structured3d/semseg-lw-c-kd-tb

# structured3d semseg 200 epoch : Teacher: Large tl
bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c-kd-tl -n structured3d/semseg-lw-c-kd-tl