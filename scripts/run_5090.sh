sleep 4h && \
bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c-kd-tl -n structured3d/semseg-lw-c-kd-tl && \
bash scripts/train.sh -g 1 -d scannet -c insseg-lw-c-kd-100epoch -n scannet/insseg-lw-c-kd-100epoch && \
bash scripts/train.sh -g 1 -d nuscenes -c semseg-lw-c -n nuscenes/semseg-lw-c




# # Structured3D    kd    teacher: LitePT-L
# bash scripts/train.sh -g 1 -d structured3d -c semseg-lw-c-kd-tl -n structured3d/semseg-lw-c-kd-tl

# # ScanNet  insseg   kd  100
# bash scripts/train.sh -g 1 -d scannet -c insseg-lw-c-kd-100epoch -n scannet/insseg-lw-c-kd-100epoch

# # NuScenes        lw
# bash scripts/train.sh -g 1 -d nuscenes -c semseg-lw-c -n nuscenes/semseg-lw-c