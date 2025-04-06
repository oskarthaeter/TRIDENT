FROM nvcr.io/nvidia/pytorch:24.12-py3

ENV DEBIAN_FRONTEND=noninteractive

# Build-time arguments for user id mapping
ARG USERNAME
ARG UID
ARG GID

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    openslide-tools \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a user and group with the specified UID and GID
RUN groupadd --gid $GID $USERNAME && \
    useradd --no-log-init --uid $UID --gid $GID --create-home --shell /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

RUN pip3 install --no-cache-dir \
    openslide-python \
    scikit-image \
    && rm -rf /root/.cache/pip

# Switch to the new user
USER $USERNAME

# Set the working directory
WORKDIR /workspaces

ENV PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync,expandable_segments:False,pinned_use_cuda_host_register:True,pinned_num_register_threads:4
