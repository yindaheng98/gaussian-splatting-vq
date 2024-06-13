FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

COPY ./submodules /gaussian-splatting/submodules
COPY ./environment.yml /gaussian-splatting/environment.yml

WORKDIR /gaussian-splatting

ARG DEBIAN_FRONTEND=noninteractive
RUN echo "Installing apt packages..." \
	&& apt -y update --no-install-recommends \
	&& apt -y install software-properties-common \
	&& add-apt-repository ppa:deadsnakes/ppa \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends wget git cmake python3.11 \
	&& apt autoremove -y \
	&& apt clean -y \
	&& ln -s $(which cuda-gdb) /usr/bin/gdb
RUN wget -O Miniconda3-install.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
	&& chmod +x ./Miniconda3-install.sh && bash ./Miniconda3-install.sh -b \
    && rm -f ./Miniconda3-install.sh \
    && echo 'PATH="/root/miniconda3/bin":$PATH' >> /root/.bashrc \
	&& /root/miniconda3/bin/conda env create --file environment.yml \
	&& /root/miniconda3/bin/conda init bash \
	&& echo "conda activate gaussian_splatting" >> /root/.bashrc

SHELL [ "/bin/bash" ]