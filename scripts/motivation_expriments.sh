# !/bin/bash
# should run after kmeans_build.sh

getsize() {
    dirpath=output/vq-$1/frame$2/point_cloud/iteration_$3
    fname=clusters-16-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8.txt
    mkdir -p $dirpath/size
    stat -c%s $dirpath/point_cloud_vq.drc >$dirpath/size/$fname
}
quantize() {
    rm output/vq-$1/frame$2/point_cloud/iteration_$3/point_cloud_vq.ply
    rm output/vq-$1/frame$2/point_cloud/iteration_$3/point_cloud_vq.drc
    rm output/vq-$1/frame$2/point_cloud/iteration_$3/point_cloud_vq_ddrc.ply
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
        -qs $4 \
        -qr $5 \
        -qdc $6 \
        -qrest $7 -qg $7 \
        -qo $8 \
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
    rm -rf output/$1/frame$2/train_interp/ours_$3
    python render.py \
        -m output/$1/frame$2 \
        --iteration $3 \
        $4
}
# render coffee_martini 1 30000 "--skip_train --render_train_interp"
render_vq() {
    rm -rf output/vq-$1/frame$2/train_interp/ours_$3
    python render.py \
        -m output/vq-$1/frame$2 \
        --iteration $3 \
        $4
}
# render_vq coffee_martini 1 30000 "--skip_train --render_train_interp"
convert() {
    python RT4KSR/scripts/nerfout2dual.py \
        --dataroot RT4KSR/data/coffee_martini-kmeans-16-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8 \
        --hrsrcroot output/$1/frame$2/train_interp/ours_$3/renders \
        --grsrcroot output/vq-$1/frame$2/train_interp/ours_$3/renders \
        --crsrcroot output/vq-$1/frame$2/train_interp/ours_$3/renders \
        --name nerfout
}
# convert coffee_martini 1 30000 12 10 6 6 6
data4sr() {
    render $1 $2 $3 "--skip_train --render_train_interp"
    quantize $1 $2 $3 $4 $5 $6 $7 $8
    render_vq $1 $2 $3 "--skip_train --render_train_interp"
    convert $1 $2 $3 $4 $5 $6 $7 $8
}
# data4sr coffee_martini 1 30000 8 4 4 4 4 # worst
# data4sr coffee_martini 1 30000 8 4 4 4 4 # best
