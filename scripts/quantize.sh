# !/bin/bash
build() {
    # echo \
    python kmeans.py \
        --src output/$1/frame$2 \
        --save output/vq-$1/frame$2 \
        --iteration $3 \
        build \
        --log2-clusters $4 \
        --attribute $5
}
build_all() {
    build $1 $2 $3 16 scaling
    build $1 $2 $3 16 rotation
    build $1 $2 $3 16 features_dc
    build $1 $2 $3 16 features_rest
    build $1 $2 $3 16 opacity
}

build_all coffee_martini 1 30000
