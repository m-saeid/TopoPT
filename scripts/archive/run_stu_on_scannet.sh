
# student lw-c:

# ScanNet semseg 1200 epoch
bash scripts/train.sh -g 1 -d scannet -c semseg-lw-c-1200epoch -n scannet/semseg-lw-c-1200epoch

# ScanNet insseg 100 epoch
bash scripts/train.sh -g 1 -d scannet -c insseg-lw-c-100epoch -n scannet/insseg-lw-c-100epoch

# ScanNet insseg 800 epoch
bash scripts/train.sh -g 1 -d scannet -c insseg-lw-c-800epoch -n scannet/insseg-lw-c-800epoch

# ScanNet200 insseg 100 epoch
bash scripts/train.sh -g 1 -d scannet200 -c insseg-lw-c-100epoch -n scannet200/insseg-lw-c-100epoch

# ScanNet200 insseg 800 epoch
bash scripts/train.sh -g 1 -d scannet200 -c insseg-lw-c-800epoch -n scannet200/insseg-lw-c-800epoch
