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
build coffee_martini 1 30000 12 8 features_dc # debug
