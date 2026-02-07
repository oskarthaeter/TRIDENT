FROM nvcr.io/nvidia/cuda-dl-base:25.06-cuda12.9-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME
ARG UID
ARG GID

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo openslide-tools python3.12-dev python3-pip python3-venv \
 && rm -rf /var/lib/apt/lists/*

# Create user first
RUN groupadd --gid ${GID} ${USERNAME} \
 && useradd --no-log-init --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Prepare venv location and hand over ownership
ENV VENV_PATH=/opt/venv
RUN mkdir -p ${VENV_PATH} && chown -R ${UID}:${GID} ${VENV_PATH}

# Switch to user before creating the venv
USER ${USERNAME}

# Create and use the venv (no 'activate' needed in Dockerfile)
RUN python3 -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Install packages into the venv
RUN python -m pip install --upgrade pip \
 && python -m pip install --no-cache-dir \
    torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu129 \
#  && python -m pip install -U --no-cache-dir xformers --extra-index-url https://download.pytorch.org/whl/cu129 \
 && python -m pip install --no-cache-dir \
    openslide-python scikit-image numpy ipykernel

ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH
WORKDIR /workspaces

# Optional CUDA alloc config
ENV PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:False,pinned_use_cuda_host_register:True,pinned_num_register_threads:4