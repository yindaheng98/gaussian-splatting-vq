# !/bin/bash
build() {
    INDEX=$([ -n "$7" ] && echo $7 || echo 0)
    PERFIX="output/lvq-$1/frame$2/point_cloud/iteration_$3/lkmeans_${4}to${5}_$6"
    if [ ! -e "$PERFIX.npz" ] && [ ! -e "${PERFIX}_$INDEX.npz" ]; then
    # echo \
    python scalablevq.py \
        --src output/$1/frame$2 \
        --save output/lvq-$1/frame$2 \
        --iteration $3 \
        --sh-degree 2 \
        build \
        --save-init output/vq-$1/frame$2 \
        --log2-clusters-init $4 \
        --log2-clusters-final $5 \
        --attribute $6 \
        --index $INDEX
    fi
}
build_all_attr() {
    build $1 $2 $3 16 6 scaling
    build $1 $2 $3 16 6 rotation
    build $1 $2 $3 16 6 features_dc
    build $1 $2 $3 16 6 features_rest 0
    # build $1 $2 $3 $4 $5 opacity
}

build_all_attr coffee_martini 1 30000
build_all_attr cook_spinach 1 30000
build_all_attr cut_roasted_beef 1 30000
build_all_attr flame_salmon_1 1 30000
build_all_attr flame_steak 1 30000
build_all_attr sear_steak 1 30000

build_all_attr discussion 1 30000
build_all_attr stepin 1 30000
build_all_attr trimming 1 30000
build_all_attr vrheadset 1 30000
