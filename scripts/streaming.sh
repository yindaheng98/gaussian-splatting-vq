#!/bin/sh
train_prediction() {
    mkdir -p output/run-$1/camera$2
    python prediction_train.py \
        --cameras saved_data/$1-camera$2.txt \
        --save output/run-$1/camera$2/prediction.pth
}
# train_prediction coffee_martini 0
test_prediction() {
    mkdir -p output/run-$1/camera$2
    python prediction_test.py \
        --cameras saved_data/$1-camera$2.txt \
        --save output/run-$1/camera$2/prediction.pth
}
# test_prediction coffee_martini 0
run() {
    mkdir -p output/run-$1/camera$2
    python run.py \
        --video output/$1 \
        --sh-degree 2 \
        --codebooks output/vq-$1/frame1/point_cloud/iteration_30000 \
        --bandwidth-start 3457 --bandwidth-end 3500 \
        --cameras saved_data/$1-camera$2.txt \
        --prediction $3 \
        --prediction-conf "$4" \
        --fov-save output/run-$1/camera$2/fov-$3.txt \
        --restore-save RT4KSR/code/checkpoints/$1-kmeans-merge-warped/nerfsrresnet_x1_rep_model.pth \
        --trace-save output/run-$1/camera$2/trace-$3-$6.json \
        --image-save output/run-$1/camera$2/image-$3-$6 \
        --max-frame $5 \
        $7
}
# run coffee_martini 0 VAR '{"path":"saved_data/test.txt"}' 3 gen_fov
run_var() {
    rm output/run-$1/camera$2/fov-VAR.txt
    run $1 $2 VAR '{"path":"saved_data/test.txt"}' $3 gen_fov
    run $1 $2 VAR '{"path":"saved_data/test.txt"}' $3 run_fov
}
# run_var coffee_martini 0 3
run_lstm() {
    rm output/run-$1/camera$2/fov-LSTM.txt
    run $1 $2 LSTM '{"path":"saved_data/test.txt"}' $3 gen_fov "--prediction-load output/run-$1/camera$2/prediction.pth"
    run $1 $2 LSTM '{"path":"saved_data/test.txt"}' $3 run_fov
}
# run_lstm coffee_martini 0 3
