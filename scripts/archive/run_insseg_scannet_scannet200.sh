# # scannet insseg — step 4: from scratch, no -w (rerun your previous attempt)
# bash scripts/train.sh -g 1 -d scannet -c insseg-litept-lw-c -n insseg-litept-lw-c

# # scannet insseg — step 5: KD, warm-start student from step 4
# bash scripts/train.sh -g 1 -d scannet -c insseg-litept-lw-c-kd -n insseg-litept-lw-c-kd

# # scannet200 insseg — step 4: from scratch, no -w
# bash scripts/train.sh -g 1 -d scannet200 -c insseg-litept-lw-c -n insseg-litept-lw-c

# scannet200 insseg — step 5: KD, warm-start student from step 4
bash scripts/train.sh -g 1 -d scannet200 -c insseg-litept-lw-c-kd -n insseg-litept-lw-c-kd