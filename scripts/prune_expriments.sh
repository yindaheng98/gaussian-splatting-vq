# !/bin/bash
prune() {
    rm -rf output/prune-$1/frame$2-$4/point_cloud/iteration_$3
    mkdir -p output/prune-$1/frame$2-$4/point_cloud/iteration_$3
    rm output/prune-$1/frame$2-$4/cfg_args
    cp output/$1/frame$2/cfg_args output/prune-$1/frame$2-$4/cfg_args
    rm output/prune-$1/frame$2-$4/point_cloud/iteration_$3/point_cloud.ply
    python prune.py \
        --read output/$1/frame$2/point_cloud/iteration_$3/point_cloud.ply \
        --sh-degree 2 \
        --save output/prune-$1/frame$2-$4/point_cloud/iteration_$3/point_cloud.ply \
        --target $4
}
# prune coffee_martini 1 30000 100000
render() {
    rm -rf output/prune-$1/frame$2-$4/train_interp/ours_$3
    python render.py \
        -m output/prune-$1/frame$2-$4 \
        --iteration $3 \
        $5
}
# render coffee_martini 1 30000 100000 "--skip_train --render_train_interp"
render_gt() {
    python render.py \
        -m output/$1/frame$2 \
        --iteration $3 \
        $4
}
# render_gt coffee_martini 1 30000 "--skip_train --render_train_interp"
metrics() {
    python scripts/metrics.py \
        --images output/prune-$1/frame$2-$4/train_interp/ours_$3/renders \
        --gt output/$1/frame$2/train_interp/ours_$3/renders
}
# metrics coffee_martini 1 30000 100000
doall() {
    # render_gt $1 "--skip_train --render_train_interp"
    eval_prune() {
        prune $1 $2
        render $1 $2 "--skip_train --render_train_interp"
        metrics $1 $2
    }
    eval_prune "$1" 200000
    eval_prune "$1" 175000
    eval_prune "$1" 150000
    eval_prune "$1" 125000
    eval_prune "$1" 100000
    eval_prune "$1" 75000
    eval_prune "$1" 50000
}
doall "coffee_martini 1 30000"
doall "cook_spinach 1 30000"
doall "cut_roasted_beef 1 30000"
doall "flame_salmon_1 1 30000"
doall "flame_steak 1 30000"
doall "sear_steak 1 30000"

doall "discussion 1 30000"
doall "stepin 1 30000"
doall "trimming 1 30000"
doall "vrheadset 1 30000"

doall "taekwondo 1 30000"
doall "walking 1 30000"
