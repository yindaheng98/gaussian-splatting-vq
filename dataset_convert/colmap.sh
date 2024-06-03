export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC
apt-get update && \
apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev && \
git clone https://github.com/colmap/colmap.git -b 3.9.1 && \
cd colmap && \
mkdir -p build && \
cd build && \
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=native && \
ninja && \
rm /data/colmap && \
cp src/colmap/exe/colmap /data
