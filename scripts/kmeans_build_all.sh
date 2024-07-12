# !/bin/bash
build() {
    INDEX=$([ -n "$6" ] && echo $6 || echo 0)
    PERFIX="output/vq-$1/frame$2/point_cloud/iteration_$3/kmeans_${4}_$5"
    if [ ! -e "$PERFIX.npz" ] && [ ! -e "${PERFIX}_$INDEX.npz" ]; then
    # echo \
    python kmeans.py \
        --src output/$1/frame$2 \
        --save output/vq-$1/frame$2 \
        --iteration $3 \
        --sh-degree 2 \
        build \
        --log2-clusters $4 \
        --attribute $5 \
        --index $INDEX
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
build_all_precision cook_spinach 1 30000
build_all_precision cut_roasted_beef 1 30000
build_all_precision flame_salmon_1 1 30000
build_all_precision flame_steak 1 30000
build_all_precision sear_steak 1 30000

build_all_precision discussion 1 30000
build_all_precision stepin 1 30000
build_all_precision trimming 1 30000
build_all_precision vrheadset 1 30000
