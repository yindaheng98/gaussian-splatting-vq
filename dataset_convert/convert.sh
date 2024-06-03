# for coffee_martini
python dataset_convert/n3dv2imgs.py --path data/coffee_martini --exec ./data/ffmpeg --n_frames 300 > ./temp.sh && ./temp.sh && rm ./temp.sh
python convert.py --colmap_executable ".\data\colmap\COLMAP.bat" -s "data/coffee_martini/frame1"
