# !/bin/bash
train() {
    # echo \
    python train.py \
        -s data/$1/frame1 \
        --model_path output/coffee_martini/frame1
    for ((i = 2; i <= $2; ++i)); do
        # echo \
        python train.py \
            -s data/$1/frame$i \
            --model_path output/$1/frame$i \
            --start_ply output/$1/frame$(expr $i - 1)/point_cloud/iteration_30000/point_cloud.ply \
            --densify_until_iter 0 \
            --iterations 7000 \
            --position_lr_init 0.0016
    done
}

train coffee_martini 300
