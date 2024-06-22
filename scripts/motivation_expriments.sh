# !/bin/bash
# should run after kmeans_build.sh
QP=16
getsize() {
    dirpath=output/vq-$1/frame$2/point_cloud/iteration_$3
    fname=qp-$QP-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8.json
    mkdir -p $dirpath/size
    jdata='"frame":'$(stat -c%s $dirpath/point_cloud_vq.drc)
    jdata="$jdata"',"scale":'$(stat -c%s $dirpath/kmeans_${4}_scaling.npz)
    jdata="$jdata"',"rot":'$(stat -c%s $dirpath/kmeans_${5}_rotation.npz)
    jdata="$jdata"',"f_dc":'$(stat -c%s $dirpath/kmeans_${6}_features_dc.npz)
    jdata="$jdata"',"f_rest":'$(stat -c%s $dirpath/kmeans_${7}_features_rest_0.npz)
    jdata="$jdata"',"opacity":'$(stat -c%s $dirpath/kmeans_${8}_opacity.npz)
    echo "{$jdata}" >$dirpath/size/$fname
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
        -qp $QP \
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
    cp output/$1/frame$2/cfg_args output/vq-$1/frame$2/cfg_args
    python render.py \
        -m output/vq-$1/frame$2 \
        --iteration $3 \
        $4
}
# render_vq coffee_martini 1 30000 "--skip_train --render_train_interp"
convert() {
    python RT4KSR/scripts/nerfout2dual.py \
        --dataroot RT4KSR/data/$1-kmeans-qp-$QP-scale-$4-rot-$5-f_dc-$6-f_rest-$7-opacity-$8 \
        --hrsrcroot output/$1/frame$2/train_interp/ours_$3/renders \
        --grsrcroot output/vq-$1/frame$2/train_interp/ours_$3/renders \
        --crsrcroot output/vq-$1/frame$2/train_interp/ours_$3/renders \
        --name nerfout
}
# convert coffee_martini 1 30000 12 10 6 6 6
data4sr() {
    render $1 $2 $3 "--skip_train --render_train_interp"
    quant_convert() {
        QP=$9
        quantize $1 $2 $3 $4 $5 $6 $7 $8
        render_vq $1 $2 $3 "--skip_train --render_train_interp"
        convert $1 $2 $3 $4 $5 $6 $7 $8
    }
    quant_convert $1 $2 $3 8 4 4 4 4 8     # worst
    quant_convert $1 $2 $3 16 16 16 16 4 8 # best
    quant_convert $1 $2 $3 12 10 10 10 4 8 # dual=color restore
    quant_convert $1 $2 $3 14 13 13 13 4 8
    quant_convert $1 $2 $3 10 7 7 7 4 8
}
data4sr coffee_martini 1 30000
data4sr cook_spinach 1 30000
data4sr cut_roasted_beef 1 30000
data4sr flame_salmon_1 1 30000
data4sr flame_steak 1 30000
data4sr sear_steak 1 30000

data4sr discussion 1 30000
data4sr stepin 1 30000
data4sr trimming 1 30000
data4sr vrheadset 1 30000