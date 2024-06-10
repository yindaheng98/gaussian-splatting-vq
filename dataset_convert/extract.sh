# !/bin/bash
# Just an example to get ffmpeg
wget -O data/ffmpeg-6.0.1-amd64-static.tar.xz https://www.johnvansickle.com/ffmpeg/old-releases/ffmpeg-6.0.1-amd64-static.tar.xz
tar -xvf data/ffmpeg-6.0.1-amd64-static.tar.xz ffmpeg-6.0.1-amd64-static/ffmpeg
mv ffmpeg-6.0.1-amd64-static/ffmpeg data/ffmpeg
rm -r ffmpeg-6.0.1-amd64-static

# Just an example to build colmap
sudo ./dataset_convert/colmap.sh

# Just an example for extracting datasets
unzip -o ~/dataset/video/Neural_3D_Video/coffee_martini.zip -d data
unzip -o ~/dataset/video/Neural_3D_Video/cook_spinach.zip -d data
unzip -o ~/dataset/video/Neural_3D_Video/cut_roasted_beef.zip -d data
unzip -o ~/dataset/video/Neural_3D_Video/flame_steak.zip -d data
unzip -o ~/dataset/video/Neural_3D_Video/sear_steak.zip -d data
unzip -o ~/dataset/video/dynamic_nerf_meeting_room_dataset/discussion.zip -d data
unzip -o ~/dataset/video/dynamic_nerf_meeting_room_dataset/stepin.zip -d data
unzip -o ~/dataset/video/dynamic_nerf_meeting_room_dataset/trimming.zip -d data
unzip -o ~/dataset/video/dynamic_nerf_meeting_room_dataset/vrheadset.zip -d data

zip -F ~/dataset/video/Neural_3D_Video/flame_salmon_1_split.zip --out flame_salmon_1.zip
unzip -o flame_salmon_1.zip -d data
rm flame_salmon_1.zip

mkdir -p data/cameras
wget -O data/cameras/cameras.zip https://github.com/yindaheng98/gaussian-splatting-vq/releases/download/v0.0-camera/cameras.zip
cd data/cameras
unzip cameras.zip
cd ../../
