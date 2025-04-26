FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y git build-essential \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    wget curl gcc ocl-icd-opencl-dev opencl-headers clinfo pkg-config \
    libclblast-dev libopenblas-dev ffmpeg libsm6 libxext6 libssl-dev \
    && mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Install rust, needed for pip installation later on
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain nightly -y
RUN . "$HOME/.cargo/env"

# Set up Python and make Python 3.11 the default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip

# Install uv to speed up pip installations
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"

# Create workspace directory
WORKDIR /app

# Install llama-cpp-python with CUDA
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1
RUN CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python fastapi uvicorn sse-starlette pydantic-settings starlette-context --system

# install stable-diffusion-webui
WORKDIR /app/stable-diffusion-webui
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --system
RUN uv pip install transformers diffusers invisible-watermark --system
RUN uv pip install git+https://github.com/crowsonkb/k-diffusion.git --system
RUN uv pip install git+https://github.com/TencentARC/GFPGAN.git --system
COPY ./stable-diffusion-webui/repositories/CodeFormer/requirements.txt /app/stable-diffusion-webui/requirements_CodeFormer.txt
COPY ./stable-diffusion-webui/requirements.txt /app/stable-diffusion-webui/requirements.txt
RUN uv pip install -r requirements_CodeFormer.txt --system
RUN uv pip install -r requirements.txt --system
RUN uv pip install -U numpy --system

# Copy run script
#WORKDIR /app
#COPY start.sh /app/start.sh
#RUN chmod +x /app/start.sh

# Expose ports for both services
#EXPOSE 7860 8000

#CMD ["/app/start.sh"]
#CMD ["/bin/bash"]