#!/bin/sh
train_prediction() {
    mkdir -p output/prediction
    python prediction_train.py \
        --cameras saved_data/$1.txt \
        --save output/prediction/$1.pth
}
# train_prediction coffee_martini
test_prediction() {
    mkdir -p output/prediction
    python prediction_test.py \
        --cameras saved_data/$1.txt \
        --save output/prediction/$1.pth
}
# test_prediction coffee_martini
