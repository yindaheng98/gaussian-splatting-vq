# !/bin/bash
build() {
    PERFIX="output/vq-$1/frame$2/point_cloud/iteration_$3/kmeans_${4}_$5"
    if [ ! -e "$PERFIX.npz" ] && [ ! -e "${PERFIX}_$6.npz" ]; then
    # echo \
    python kmeans.py \
        --src output/$1/frame$2 \
        --save output/vq-$1/frame$2 \
        --iteration $3 \
        build \
        --log2-clusters $4 \
        --attribute $5
    fi
}
build_all_attr() {
    build $1 $2 $3 $4 scaling
    build $1 $2 $3 $4 rotation
    build $1 $2 $3 $4 features_dc
    build $1 $2 $3 $4 features_rest 0
    build $1 $2 $3 $4 opacity
}
build_all_precision() {
    for i in {4..16}; do
        build_all_attr $1 $2 $3 $i
    done
}

build_all_precision coffee_martini 1 30000
