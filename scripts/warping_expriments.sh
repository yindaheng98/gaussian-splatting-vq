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
render() {
    rm -rf output/$1/frame$2/train_interp/ours_$3
    python render.py \
        -m output/$1/frame$2 \
        --iteration $3 \
        $4
}
# render coffee_martini 1 30000 "--skip_train --render_train_interp"
render_vq() {
    rm -rf output/vq-$1/frame$2/train_interp/ours_$3
    cp output/$1/frame$2/cfg_args output/vq-$1/frame$2/cfg_args
    python render.py \
        -m output/vq-$1/frame$2 \
        --iteration $3 \
        $4
}
# render_vq coffee_martini 1 30000 "--skip_train --render_train_interp"