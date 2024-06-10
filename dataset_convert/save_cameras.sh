# !/bin/bash

save_cameras() {
    rm -rf data/cameras/$1
    mkdir -p data/cameras/$1
    cp -r data/$1/frame1 data/cameras/$1/frame1
}

save_cameras coffee_martini
save_cameras cook_spinach
