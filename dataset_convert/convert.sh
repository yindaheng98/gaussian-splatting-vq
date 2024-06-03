# !/bin/bash
convert_n3dv() {
    # echo \
    python dataset_convert/n3dv2imgs.py \
        --path data/$1 \
        --exec ./data/ffmpeg \
        --n_frames $2 >./temp.sh && ./temp.sh && rm ./temp.sh
    # echo \
    python convert.py \
        --colmap_executable ".\data\colmap\COLMAP.bat" \
        -s "data/$1/frame1" \
        --single_camera_per_image
    for ((i = 2; i <= $2; ++i)); do
        # echo \
        python convert.py \
            --colmap_executable ".\data\colmap\COLMAP.bat" \
            -s "data/$1/frame$i" \
            --single_camera_per_image \
            --load_cameras_path data/$1/frame1
    done
}
convert_n3dv coffee_martini 300
# convert_n3dv flame_salmon_1 1200
convert_n3dv flame_steak 300
convert_n3dv sear_steak 300
