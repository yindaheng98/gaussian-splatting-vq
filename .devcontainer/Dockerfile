FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

COPY ./ /gaussian-splatting

WORKDIR /gaussian-splatting

RUN echo "Installing apt packages..." \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends git \
	&& apt autoremove -y \
	&& apt clean -y \
	&& ln -s $(which cuda-gdb) /usr/bin/gdb
RUN conda env create --file environment.yml \
	&& conda init bash

SHELL [ "/bin/bash" ]

ENTRYPOINT [ "/gaussian-splatting/.devcontainer/entrypoint.sh" ]