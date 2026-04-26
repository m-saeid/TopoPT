# ScanNet insseg 100 epoch kd-teacher: LitePT-S*
bash scripts/train.sh -g 1 -d scannet -c insseg-lw-c-kd-100epoch -n scannet/insseg-lw-c-kd-100epoch

# ScanNet200 insseg 100 epoch lw-c
bash scripts/train.sh -g 1 -d scannet200 -c insseg-lw-c-100epoch -n scannet200/insseg-lw-c-100epoch

# ScanNet200 insseg 100 epoch kd-teacher: LitePT-S*
bash scripts/train.sh -g 1 -d scannet200 -c insseg-lw-c-kd-100epoch -n scannet200/insseg-lw-c-kd-100epoch









