# !/bin/bash
# should run after kmeans_build.sh
QP=16
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
    python render.py \
        -m output/$1/frame$2 \
        --iteration $3 \
        $4 \
        --render_train_interp_to train_interp_ref \
        --forcefovy 2.0 --forceheight 1200 \
        --forcefovx 2.5 --forcewidth 2319 # height/tan(fovy/2)*tan(fovx/2)
}
# render_ref coffee_martini 1 30000 "--skip_train --render_train_interp"
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
    rm -rf output/vq-$1/frame$2/train_interp/ours_$3/warped
    rm -rf output/vq-$1/frame$2/train_interp/ours_$3/warped_no_ee
    for ((i = 0; i <= $4; ++i)); do
        echo $i/$4
        python warping.py \
            --local output/vq-$1/frame$2/train_interp/ours_$3/renders/$(printf "%05d" $i) \
            --reference output/$1/frame$2/train_interp_ref/ours_$3/renders/$5 \
            --warped output/vq-$1/frame$2/train_interp/ours_$3/warped/$(printf "%05d" $i)
        mkdir -p output/vq-$1/frame$2/train_interp/ours_$3/warped_no_ee
        mv output/vq-$1/frame$2/train_interp/ours_$3/warped/$(printf "%05d" $i).no_error_erosion.png output/vq-$1/frame$2/train_interp/ours_$3/warped_no_ee/$(printf "%05d" $i).png
    done
}
# warping coffee_martini 1 30000 323 00000
convert() {
    python RT4KSR/scripts/nerfout2dual.py \
        --dataroot RT4KSR/data/$1-kmeans-qp-none-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8-warped \
        --hrsrcroot output/$1/frame$2/train_interp/ours_$3/renders \
        --grsrcroot output/vq-$1/frame$2/train_interp/ours_$3/renders \
        --crsrcroot output/vq-$1/frame$2/train_interp/ours_$3/warped \
        --name nerfout
    python RT4KSR/scripts/nerfout2dual.py \
        --dataroot RT4KSR/data/$1-kmeans-qp-none-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8-warpednoee \
        --hrsrcroot output/$1/frame$2/train_interp/ours_$3/renders \
        --grsrcroot output/vq-$1/frame$2/train_interp/ours_$3/renders \
        --crsrcroot output/vq-$1/frame$2/train_interp/ours_$3/warped_no_ee \
        --name nerfout
}
# convert coffee_martini 1 30000 12 10 6 6 6
