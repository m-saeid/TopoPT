# NuScenes        lw
bash scripts/train.sh -g 1 -d nuscenes -c semseg-lw-c -n nuscenes/semseg-lw-c

# Structured3D    lw  100 epoch
bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c-100epoch -n structured3d/semseg-lw-c-100epoch