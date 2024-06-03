# !/bin/bash
apt-get update && \
DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y \
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
cd data &&
git clone https://github.com/colmap/colmap.git -b 3.9.1 && \
cd colmap && mkdir -p build && cd build && \
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=native && \
ninja && ninja install