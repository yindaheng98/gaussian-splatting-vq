# !/bin/bash
train() {
    if [ ! -e "output/$1/frame1/point_cloud/iteration_$3/point_cloud.ply" ]; then
    # echo \
    python train.py \
        -s data/$1/frame1 \
        --model_path output/$1/frame1 \
        --iterations $3
    fi
    LAST_FRAME="output/$1/frame1/point_cloud/iteration_$3/point_cloud.ply"
    for ((i = 2; i <= $2; ++i)); do
        if [ ! -e "$LAST_FRAME" ]; then
            continue
        fi
        THIS_FRAME="output/$1/frame$i/point_cloud/iteration_$3/point_cloud.ply"
        if [ ! -e "$THIS_FRAME" ]; then
        # echo \
        python train.py \
            -s data/$1/frame$i \
            --model_path output/$1/frame$i \
            --iterations $4 \
            --start_ply output/$1/frame$(expr $i - 1)/point_cloud/iteration_$3/point_cloud.ply \
            --densify_until_iter 0 \
            --position_lr_init 0.0016 \
            --incremental
        fi
        LAST_FRAME="$THIS_FRAME"
    done
}

train coffee_martini 300 30000 7000
train cook_spinach 300 30000 7000
train cut_roasted_beef 300 30000 7000
train flame_salmon_1 1200 30000 7000
train flame_steak 300 30000 7000
train sear_steak 300 30000 7000

train discussion 300 30000 7000
train stepin 300 30000 7000
train trimming 300 30000 7000
train vrheadset 300 30000 7000
