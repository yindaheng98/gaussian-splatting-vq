#!/bin/sh
OPT="--multi-thread-cutoff 16M --multi-thread-streams 16"
# 下载首帧训练结果

download() {
    rclone -v --config rclone.conf $OPT copy server:gaussian-splatting-vq/data/$1/frame1 ./data/$1/frame1
    rclone -v --config rclone.conf $OPT copy server:gaussian-splatting-vq/output/$1/frame1 ./output/$1/frame1
    rclone -v --config rclone.conf $OPT copy server:gaussian-splatting-vq/output/vq-$1/frame1 ./output/vq-$1/frame1
}

# 下载全部训练结果

download_all() {
    for i in $(seq 1 $2); do
        rclone -v --config rclone.conf $OPT copy server:gaussian-splatting-vq/output/$1/frame$i ./output/$1/frame$i
    done
}

TASK=download_all
$TASK coffee_martini 100
$TASK cook_spinach 100
$TASK cut_roasted_beef 100
$TASK flame_salmon_1 100
$TASK flame_steak 100
$TASK sear_steak 100

$TASK discussion 100
$TASK stepin 100
$TASK trimming 100
$TASK vrheadset 100

$TASK taekwondo 100
$TASK walking 100
