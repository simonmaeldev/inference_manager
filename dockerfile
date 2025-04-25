FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y git build-essential \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    wget gcc ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev ffmpeg libsm6 libxext6 \
    && mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

# Set up Python and make Python 3.11 the default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip

# Create workspace directory
WORKDIR /app

# Install llama-cpp-python with CUDA
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Clone and install stable-diffusion-webui
RUN git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
WORKDIR /app/stable-diffusion-webui
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy run script
WORKDIR /app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose ports for both services
EXPOSE 7860 8000

CMD ["/app/start.sh"]
