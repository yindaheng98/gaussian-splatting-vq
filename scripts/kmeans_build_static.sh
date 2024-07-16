# !/bin/bash
build() {
    INDEX=$([ -n "$5" ] && echo $5 || echo 0)
    PERFIX="output/vq-$1/point_cloud/iteration_$2/kmeans_${3}_$4"
    if [ ! -e "$PERFIX.npz" ] && [ ! -e "${PERFIX}_$INDEX.npz" ]; then
    # echo \
    python kmeans.py \
        --src output/$1 \
        --save output/vq-$1 \
        --iteration $2 \
        build \
        --log2-clusters $3 \
        --attribute $4 \
        --index $INDEX
    fi
}
build_all_attr() {
    build $1 $2 $3 scaling
    build $1 $2 $3 rotation
    build $1 $2 $3 features_dc
    build $1 $2 $3 features_rest 0
    build $1 $2 $3 opacity
}
build_all_precision() {
    for i in {4..16}; do
        build_all_attr $1 $2 $i
    done
}

build_all_precision truck 30000
