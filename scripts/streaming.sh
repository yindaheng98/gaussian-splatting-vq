#!/bin/bash
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
    mkdir -p output/run-$1/camera$2/start$3
    python run.py \
        --video output/$1 \
        --sh-degree 2 \
        --codebooks output/vq-$1/frame1/point_cloud/iteration_30000 \
        --bandwidth-start 3457 --bandwidth-end 3500 \
        --cameras saved_data/$1-camera$2.txt \
        --cameras-start $3 \
        --prediction $4 \
        --prediction-conf '{"path":"saved_data/test.txt"}' \
        --fov-save output/run-$1/camera$2/start$3/fov-$4.txt \
        --restore-save RT4KSR/code/checkpoints/$1-kmeans-merge-warped/nerfsrresnet_x1_rep_model.pth \
        --trace-save output/run-$1/camera$2/start$3/trace-$4-$6.json \
        --image-save output/run-$1/camera$2/start$3/image-$4-$6 \
        --max-frame $5 \
        $7
}
# run coffee_martini 0 10 VAR 3 gen_fov
run_var() {
    rm output/run-$1/camera$2/start$3/fov-VAR.txt
    run $1 $2 $3 VAR $4 gen_fov
    if [ -e "output/run-$1/camera$2/start$3/fov-VAR.txt" ]; then
        run $1 $2 $3 VAR $4 run_fov "--use-enlarged-in-mark-visible"
    fi
}
# run_var coffee_martini 0 10 3
run_lstm() {
    rm output/run-$1/camera$2/start$3/fov-LSTM.txt
    run $1 $2 $3 LSTM $4 gen_fov "--prediction-load output/run-$1/camera$2/prediction.pth"
    if [ -e "output/run-$1/camera$2/start$3/fov-LSTM.txt" ]; then
        run $1 $2 $3 LSTM $4 run_fov "--use-enlarged-in-mark-visible"
    fi
}
# run_lstm coffee_martini 0 10 3
run_both() {
    run_var $1 $2 $3 $4 &
    run_lstm $1 $2 $3 $4 &
    wait
}
# run_both coffee_martini 0 10 3
run_all() {
    train_prediction $1 $2
    test_prediction $1 $2
    for i in {0..10}; do
        run_both $1 $2 $(($i * 100)) 100
        # run_both $1 $2 $(($i * 100)) 100 &
        # if [ $(($i % 3)) == 2 ]; then
        #     wait
        # fi
    done
}
run_all coffee_martini 0
