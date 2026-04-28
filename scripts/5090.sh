# ScanNet  insseg   kd  100
bash scripts/train.sh -g 1 -d scannet -c insseg-lw-c-kd-100epoch -n scannet/insseg-lw-c-kd-100epoch

# Structured3D  tl  kd  100 epoch
bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c-kd-tl-100epoch -n structured3d/semseg-lw-c-kd-tl-100epoch