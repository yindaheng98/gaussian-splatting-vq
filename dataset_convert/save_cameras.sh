# !/bin/bash

save_cameras() {
    rm -rf data/cameras/$1
    mkdir -p data/cameras/$1
    cp -r data/$1/frame1 data/cameras/$1/frame1
}

rm -rf data/cameras
mkdir -p data/cameras

save_cameras coffee_martini
save_cameras cook_spinach
save_cameras cut_roasted_beef
save_cameras flame_salmon_1
save_cameras flame_steak
save_cameras sear_steak

save_cameras discussion
save_cameras stepin
save_cameras trimming
save_cameras vrheadset

cd data/cameras
rm cameras.zip
zip -r cameras.zip ./
