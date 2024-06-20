# !/bin/bash
# should run after kmeans_build.sh

getsize() {
    dirpath=output/vq-$1/frame$2/point_cloud/iteration_$3
    fname=clusters-16-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8.txt
    mkdir -p $dirpath/size
    stat -c%s $dirpath/point_cloud_vq.drc >>$dirpath/size/$fname
}
quantize() {
    # quant by kmeans
    python kmeans.py \
        --src output/$1/frame$2 \
        --save output/vq-$1/frame$2 \
        --iteration $3 \
        quantize \
        --dst output/vq-$1/frame$2 \
        --log2-clusters 16 \
        --log2-clusters-scaling $4 \
        --log2-clusters-rotation $5 \
        --log2-clusters-features_dc $6 \
        --log2-clusters-features_rest $7 \
        --log2-clusters-opacity $8
    # compress by draco
    ./build/Release/draco_encoder \
        -i output/vq-$1/frame$2/point_cloud/iteration_$3/point_cloud_vq.ply \
        -o output/vq-$1/frame$2/point_cloud/iteration_$3/point_cloud_vq.drc \
        -cl 0 \
        -qp 16 \
        -point_cloud \
        -use_metadata
    # get size
    getsize $@
    # decompress by draco
    ./build/Release/draco_decoder \
        -i output/vq-$1/frame$2/point_cloud/iteration_$3/point_cloud_vq.drc \
        -o output/vq-$1/frame$2/point_cloud/iteration_$3/point_cloud_vq_ddrc.ply
    # dequant by kmeans
    python kmeans.py \
        --src output/$1/frame$2 \
        --save output/vq-$1/frame$2 \
        --iteration $3 \
        dequantize \
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
    python render.py \
        -m output/$1/frame$2 \
        --iteration $3 \
        $4
}
render coffee_martini 1 30000 "--skip_train --render_train_interp"
render_vq() {
    python render.py \
        -m output/vq-$1/frame$2 \
        --iteration $3 \
        $4
}
render_vq coffee_martini 1 30000 "--skip_train --render_train_interp"
