#!/bin/sh
train_prediction() {
    mkdir -p output/run-$1
    python prediction_train.py \
        --cameras saved_data/$1-$2.txt \
        --save output/run-$1/prediction-$2.pth
}
# train_prediction coffee_martini 0
test_prediction() {
    mkdir -p output/run-$1
    python prediction_test.py \
        --cameras saved_data/$1-$2.txt \
        --save output/run-$1/prediction-$2.pth
}
# test_prediction coffee_martini 0
gen_fov() {
    mkdir -p output/run-$1
    rm output/run-$1/fov-$2.txt
    python run.py \
        --video output/$1 \
        --sh-degree 2 \
        --codebooks output/vq-$1/frame1/point_cloud/iteration_30000 \
        --bandwidth-start 3457 --bandwidth-end 3500 \
        --cameras saved_data/$1-$2.txt \
        --max-frame $3 \
        --fov-save output/run-$1/fov-$2.txt \
        --prediction $4 \
        --prediction-conf "$5" \
        --restore-save RT4KSR/code/checkpoints/$1-kmeans-merge-warped/nerfsrresnet_x1_rep_model.pth
}
gen_fov coffee_martini 0 15 VAR '{"path":"saved_data/test.txt"}'
