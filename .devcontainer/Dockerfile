FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

COPY ./submodules /gaussian-splatting/submodules
COPY ./environment.yml /gaussian-splatting/environment.yml

WORKDIR /gaussian-splatting

RUN echo "Installing apt packages..." \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends git cmake \
	&& apt autoremove -y \
	&& apt clean -y \
	&& ln -s $(which cuda-gdb) /usr/bin/gdb
RUN conda env create --file environment.yml \
	&& conda init bash \
	&& echo "conda activate gaussian_splatting" >> /root/.bashrc

SHELL [ "/bin/bash" ]