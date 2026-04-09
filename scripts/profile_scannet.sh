bash scriptss/profile.sh \
    -c semseg-litept-lw-a \
    -w exp/scannet/semseg-litept-lw-a/model/model_best.pth \
    -n litept-lw-a-trained

bash scriptss/profile.sh \
    -c semseg-litept-lw-b \
    -w exp/scannet/semseg-litept-lw-b/model/model_best.pth \
    -n litept-lw-b-trained

bash scriptss/profile.sh \
    -c semseg-litept-lw-c \
    -w exp/scannet/semseg-litept-lw-c/model/model_best.pth \
    -n litept-lw-c-trained

bash scriptss/profile.sh \
    -c semseg-litept-lw-d \
    -w exp/scannet/semseg-litept-lw-d/model/model_best.pth \
    -n litept-lw-d-trained

bash scriptss/profile.sh \
    -c semseg-litept-lw-e \
    -w exp/scannet/semseg-litept-lw-e/model/model_best.pth \
    -n litept-lw-e-trained

bash scriptss/profile.sh \
    -c semseg-litept \
    -w exp/scannet/semseg-litept/model/model_best.pth \
    -n litept-trained

bash scriptss/profile.sh \
    -c semseg-litept-rerun \
    -w exp/scannet/semseg-litept-rerun/model/model_best.pth \
    -n litept-rerun-trained

bash scriptss/profile.sh \
    -c semseg-litept-lw-c-kd \
    -w exp/scannet/semseg-litept-lw-c-kd/model/model_best.pth \
    -n litept-lw-c-kd-trained

bash scriptss/profile.sh \
    -c semseg-litept-lw-c-kd-2 \
    -w exp/scannet/semseg-litept-lw-c-kd-2/model/model_best.pth \
    -n litept-lw-c-kd-2-trained
