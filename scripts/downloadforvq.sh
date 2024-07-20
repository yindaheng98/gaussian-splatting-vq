#!/bin/sh
# 下载首帧训练结果

download() {
    rclone -v --config rclone.conf copy server:gaussian-splatting-vq/output/$1/frame1 ./output/$1/frame1
    rclone -v --config rclone.conf copy server:gaussian-splatting-vq/output/vq-$1/frame1 ./output/vq-$1/frame1
}

download coffee_martini
download cook_spinach
download cut_roasted_beef
download flame_salmon_1
download flame_steak
download sear_steak

download discussion
download stepin
download trimming
download vrheadset

download taekwondo
download walking
