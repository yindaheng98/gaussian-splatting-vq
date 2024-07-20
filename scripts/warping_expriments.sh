# !/bin/bash
# should run after kmeans_build.sh
quantize() {
    rm output/vq-$1/frame$2/point_cloud/iteration_$3/point_cloud_vq.ply
    # quant by kmeans
    python kmeans.py \
        --src output/$1/frame$2 \
        --save output/vq-$1/frame$2 \
        --iteration $3 \
        --sh-degree 2 \
        quantize \
        --dst output/vq-$1/frame$2 \
        --log2-clusters 16 \
        --log2-clusters-scaling $4 \
        --log2-clusters-rotation $5 \
        --log2-clusters-features_dc $6 \
        --log2-clusters-features_rest $7 \
        --log2-clusters-opacity $8
    # dequant by kmeans
    python kmeans.py \
        --src output/$1/frame$2 \
        --save output/vq-$1/frame$2 \
        --iteration $3 \
        --sh-degree 2 \
        dequantize \
        --filename point_cloud_vq \
        --dst output/vq-$1/frame$2 \
        --log2-clusters 16 \
        --log2-clusters-scaling $4 \
        --log2-clusters-rotation $5 \
        --log2-clusters-features_dc $6 \
        --log2-clusters-features_rest $7 \
        --log2-clusters-opacity $8

}
# quantize coffee_martini 1 30000 12 10 6 6 6
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
render_gt() {
    rm -rf output/$1/frame$2/train_interp/ours_$3
    python render.py \
        -m output/$1/frame$2 \
        --iteration $3 \
        $4
}
# render_gt coffee_martini 1 30000 "--skip_train --render_train_interp"
render_ref() {
    rm -rf output/$1/frame$2/train_interp_ref/ours_$3
    fovx=2.5
    fovy=2.0
    height=$4
    width=$(python -c "import math; print(round($height/math.tan($fovy/2)*math.tan($fovx/2)))")
    # echo \
    python render.py \
        -m output/$1/frame$2 \
        --iteration $3 \
        $5 \
        --render_train_interp_to train_interp_ref \
        --forcefovy $fovy --forceheight $height \
        --forcefovx $fovx --forcewidth $width # height/tan(fovy/2)*tan(fovx/2)
}
# render_ref coffee_martini 1 30000 600 "--skip_train --render_train_interp"
render_vq() {
    rm -rf output/vq-$1/frame$2/train_interp/ours_$3
    cp output/$1/frame$2/cfg_args output/vq-$1/frame$2/cfg_args
    python render.py \
        -m output/vq-$1/frame$2 \
        --iteration $3 \
        $4
}
# render_vq coffee_martini 1 30000 "--skip_train --render_train_interp"
warping() {
    rm -rf output/vq-$1/frame$2/train_interp/ours_$3/nowarp
    rm -rf output/vq-$1/frame$2/train_interp/ours_$3/warped
    rm -rf output/vq-$1/frame$2/train_interp/ours_$3/warpednoee
    for f in $(ls output/vq-$1/frame$2/train_interp/ours_$3/renders/); do
        if ! [[ $f =~ $4 ]]; then
            continue
        fi
        i="${BASH_REMATCH[1]}"
        python warping.py \
            --local output/vq-$1/frame$2/train_interp/ours_$3/renders/$i \
            --reference output/$1/frame$2/train_interp_ref/ours_$3/renders/$5 \
            --warped output/vq-$1/frame$2/train_interp/ours_$3/warped/$i
        mkdir -p output/vq-$1/frame$2/train_interp/ours_$3/warpednoee
        mv output/vq-$1/frame$2/train_interp/ours_$3/warped/$i.no_error_erosion.png output/vq-$1/frame$2/train_interp/ours_$3/warpednoee/$i.png
        python scripts/resize_like.py \
            --read output/$1/frame$2/train_interp/ours_$3/renders/$5.png \
            --like output/vq-$1/frame$2/train_interp/ours_$3/renders/$i.png \
            --save output/vq-$1/frame$2/train_interp/ours_$3/nowarp/$i.png \
            --scale 0.25
    done
}
# warping coffee_martini 1 30000 "([0-9]+)[.]png" 00000
convert() {
    rm -rf RT4KSR/data/$1-kmeans-qp-none-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8-warped
    python RT4KSR/scripts/nerfout2dual.py \
        --dataroot RT4KSR/data/$1-kmeans-qp-none-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8-warped \
        --hrsrcroot output/$1/frame$2/train_interp/ours_$3/renders \
        --grsrcroot output/vq-$1/frame$2/train_interp/ours_$3/renders \
        --crsrcroot output/vq-$1/frame$2/train_interp/ours_$3/warped \
        --name nerfout --skipresize
    rm -rf RT4KSR/data/$1-kmeans-qp-none-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8-warpednoee
    python RT4KSR/scripts/nerfout2dual.py \
        --dataroot RT4KSR/data/$1-kmeans-qp-none-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8-warpednoee \
        --hrsrcroot output/$1/frame$2/train_interp/ours_$3/renders \
        --grsrcroot output/vq-$1/frame$2/train_interp/ours_$3/renders \
        --crsrcroot output/vq-$1/frame$2/train_interp/ours_$3/warpednoee \
        --name nerfout --skipresize
    rm -rf RT4KSR/data/$1-kmeans-qp-none-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8-nowarp
    python RT4KSR/scripts/nerfout2dual.py \
        --dataroot RT4KSR/data/$1-kmeans-qp-none-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8-nowarp \
        --hrsrcroot output/$1/frame$2/train_interp/ours_$3/renders \
        --grsrcroot output/vq-$1/frame$2/train_interp/ours_$3/renders \
        --crsrcroot output/vq-$1/frame$2/train_interp/ours_$3/nowarp \
        --name nerfout --skipresize
}
# convert coffee_martini 1 30000 12 10 6 6 6
data4sr() {
    render_gt $1 "--skip_train --render_train_interp"
    render_ref $1 $2 "--skip_train --render_train_interp"
    quant_convert() {
        quantize $1 $2
        render_vq $1 "--skip_train --render_train_interp"
        warping $1 "([0-9]+)[.]png" 00000
        convert $1 $2
    }
    quant_convert "$1" "8 4 4 4 4"     # worst
    quant_convert "$1" "16 16 16 16 4" # best
    quant_convert "$1" "12 10 10 10 4" # dual=color restore
    quant_convert "$1" "14 13 13 13 4"
    quant_convert "$1" "10 7 7 7 4"
}
data4sr "coffee_martini 1 30000" 900
# data4sr "cook_spinach 1 30000" 900
# data4sr "cut_roasted_beef 1 30000" 900
# data4sr "flame_salmon_1 1 30000" 900
# data4sr "flame_steak 1 30000" 900
# data4sr "sear_steak 1 30000" 900

data4sr "discussion 1 30000" 540
# data4sr "stepin 1 30000" 540
# data4sr "trimming 1 30000" 540
# data4sr "vrheadset 1 30000" 540

data4sr "taekwondo 1 30000" 810
# data4sr "walking 1 30000" 810
