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
build_all_attr() {
    build $1 $2 $3 $4 scaling
    build $1 $2 $3 $4 rotation
    build $1 $2 $3 $4 features_dc
    build $1 $2 $3 $4 features_rest
    build $1 $2 $3 $4 opacity
}
build_all_precision() {
    for i in {6..10}; do
        build_all_attr $1 $2 $3 $i
    done
}

build_all_precision coffee_martini 1 30000
