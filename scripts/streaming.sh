#!/bin/sh
train_prediction() {
    mkdir -p output/run-$1
    python prediction_train.py \
        --cameras saved_data/$1-camera$2.txt \
        --save output/run-$1/prediction-camera$2.pth
}
# train_prediction coffee_martini 0
test_prediction() {
    mkdir -p output/run-$1
    python prediction_test.py \
        --cameras saved_data/$1-camera$2.txt \
        --save output/run-$1/prediction-camera$2.pth
}
# test_prediction coffee_martini 0
run() {
    mkdir -p output/run-$1
    python run.py \
        --video output/$1 \
        --sh-degree 2 \
        --codebooks output/vq-$1/frame1/point_cloud/iteration_30000 \
        --bandwidth-start 3457 --bandwidth-end 3500 \
        --cameras saved_data/$1-camera$2.txt \
        --prediction $3 \
        --prediction-conf "$4" \
        --fov-save output/run-$1/fov-camera$2-$3.txt \
        --restore-save RT4KSR/code/checkpoints/$1-kmeans-merge-warped/nerfsrresnet_x1_rep_model.pth \
        --trace-save output/run-$1/trace-camera$2-$3-$6.json \
        --image-save output/run-$1/image-camera$2-$3-$6 \
        --max-frame $5
}
run coffee_martini 0 VAR '{"path":"saved_data/test.txt"}' 5 gen_fov
