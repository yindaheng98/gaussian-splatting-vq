# !/bin/bash
prune() {
    rm -rf output/prune-$1/frame$2/point_cloud/iteration_$3
    mkdir -p output/prune-$1/frame$2/point_cloud/iteration_$3
    rm output/prune-$1/frame$2/cfg_args
    cp output/$1/frame$2/cfg_args output/prune-$1/frame$2/cfg_args
    rm output/prune-$1/frame$2/point_cloud/iteration_$3/point_cloud.ply
    python prune.py \
        --read output/$1/frame$2/point_cloud/iteration_$3/point_cloud.ply \
        --sh-degree 2 \
        --save output/prune-$1/frame$2/point_cloud/iteration_$3/point_cloud.ply \
        --target $4
}
# prune coffee_martini 1 30000 100000
