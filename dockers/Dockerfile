FROM continuumio/miniconda3
ARG PYTHON_VERSION=3.7
ARG WITH_TORCHVISION=1

# Update environment
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libjpeg-dev \
    libpng-dev && \
    rm -rf /var/lib/apt/lists/*

RUN conda create -n env python=${PYTHON_VERSION}
RUN echo "source activate env" > ~/.bashrc
ENV PATH /opt/conda/envs/env/bin:$PATH

RUN conda install -n env -y numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && conda clean -ya

WORKDIR /source

# Install dependencies.
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# Copy remaining code.
COPY . .


