#!/bin/sh
train_prediction() {
    mkdir -p output/run-$1
    python prediction_train.py \
        --cameras saved_data/$1-$2.txt \
        --save output/run-$1/prediction-$2.pth
}
train_prediction coffee_martini 0
test_prediction() {
    mkdir -p output/run-$1
    python prediction_test.py \
        --cameras saved_data/$1-$2.txt \
        --save output/run-$1/prediction-$2.pth
}
test_prediction coffee_martini 0

