#!/bin/bash
train_prediction() {
    mkdir -p output/run-$1/camera$2
    # echo \
    python prediction_train.py \
        --cameras saved_data/$1-camera$2.txt \
        --save output/run-$1/camera$2/prediction.pth
}
# train_prediction coffee_martini 0
test_prediction() {
    mkdir -p output/run-$1/camera$2
    # echo \
    python prediction_test.py \
        --cameras saved_data/$1-camera$2.txt \
        --save output/run-$1/camera$2/prediction.pth
}
# test_prediction coffee_martini 0
run() {
    mkdir -p output/run-$1/camera$2/start$3
    # echo \
    python run.py \
        --video output/$1 \
        --sh-degree 2 \
        --codebooks output/vq-$1/frame1/point_cloud/iteration_30000 \
        --bandwidth-start 2901 --bandwidth-end 2978 \
        --cameras saved_data/$1-camera$2.txt \
        --cameras-start $3 \
        --prediction $4 \
        --prediction-conf '{"path":"saved_data/test.txt"}' \
        --fov-save output/run-$1/camera$2/start$3/fov-$4.txt \
        --restore-save RT4KSR/code/checkpoints/$1-kmeans-merge-warped/nerfsrresnet_x1_rep_model.pth \
        --trace-save output/run-$1/camera$2/start$3/trace-$4-$6.json \
        --image-save output/run-$1/camera$2/start$3/image-$4-$6 \
        --max-frame $5 \
        $7 &&
        rm -rf output/run-$1/camera$2/start$3/image-$4-$6
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
    run_var $1 $2 $3 $4
    # run_var $1 $2 $3 $4 &
    run_lstm $1 $2 $3 $4
    # run_lstm $1 $2 $3 $4 &
    # wait
}
# run_both coffee_martini 0 10 3
run_all() {
    train_prediction $1 $2
    test_prediction $1 $2
    for i in $(seq 0 $4); do
        run_both $1 $2 $(($i * $3)) $3
        # run_both $1 $2 $(($i * 100)) 100 &
        # if [ $(($i % 3)) == 2 ]; then
        #     wait
        # fi
    done
}
run_all coffee_martini 0 200 10
run_all cook_spinach 0 200 10
run_all cut_roasted_beef 0 200 10
run_all flame_salmon_1 0 200 10
run_all flame_steak 0 200 10
run_all sear_steak 0 100 10

run_all discussion 0 100 10
run_all stepin 0 100 10
run_all trimming 0 100 10
run_all vrheadset 0 100 10

run_all taekwondo 0 101 10
run_all walking 0 75 10
