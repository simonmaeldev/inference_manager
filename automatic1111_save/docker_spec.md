# Docker image for local inference

The goal of this docker image is to run on my computer for local inference.
We will do txt-to-txt, txt-to-image, image-to-txt and image-to-image.
For that we will have a python inference maganer, that expose an API / MCP and have intern rules for the priority of the inferences.

# Installation

Here are some example and instructions from the web to install the different parts I need. You'll find my requirements and requests in the last section of this file.


## llama-cpp-python

for txt-to-txt and image-to-txt

<docker-example>
ARG CUDA_IMAGE="12.5.0-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

# We need to set the host to 0.0.0.0 to allow outside access
ENV HOST 0.0.0.0

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

COPY . .

# setting build related env vars
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1

# Install depencencies
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Install llama-cpp-python (build with cuda)
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

# Run the server
CMD python3 -m llama_cpp.server
</docker-example>

<docker-run-cli>
docker run -it -p 7860:7860 --platform=linux/amd64 --gpus all \
	-e HF_HOME="/data/.huggingface" \
	-e REPO_ID="TheBloke/Llama-2-7B-Chat-GGML" \
	-e MODEL_FILE="llama-2-7b-chat.ggmlv3.q5_0.bin" \
	registry.hf.space/spacesexamples-llama-cpp-python-cuda-gradio:latest
</docker-run-cli>

<official-installation-doc>
To install with CUDA support, set the `GGML_CUDA=on` environment variable before installing:

CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

**Pre-built Wheel (New)** It is also possible to install a pre-built wheel with CUDA support. As long as your system meets some requirements: - CUDA Version is 12.1, 12.2, 12.3, 12.4 or 12.5 - Python Version is 3.10, 3.11 or 3.12

pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version>

Where `` is one of the following: - `cu121`: CUDA 12.1 - `cu122`: CUDA 12.2 - `cu123`: CUDA 12.3 - `cu124`: CUDA 12.4 - `cu125`: CUDA 12.5 For example, to install the CUDA 12.1 wheel:

pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
</official-installation-doc>
note : python 3.13 and cu128 works, I've tested it on my machine.

## automatic1111
api for txt-to-image and image-to-image inference

<official-documentation>
Automatic Installation on Linux

    Install the dependencies:

# Debian-based:
sudo apt install wget git python3 python3-venv libgl1 libglib2.0-0
# Red Hat-based:
sudo dnf install wget git python3 gperftools-libs libglvnd-glx
# openSUSE-based:
sudo zypper install wget git python3 libtcmalloc4 libglvnd
# Arch-based:
sudo pacman -S wget git python3

If your system is very new, you need to install python3.11 or python3.10:

# Ubuntu 24.04
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11

# Manjaro/Arch
sudo pacman -S yay
yay -S python311 # do not confuse with python3.11 package

# Only for 3.11
# Then set up env variable in launch script
export python_cmd="python3.11"
# or in webui-user.sh
python_cmd="python3.11"

    Navigate to the directory you would like the webui to be installed and execute the following command:

wget -q https://raw.githubusercontent.com/AUTOMATIC1111/stable-diffusion-webui/master/webui.sh

Or just clone the repo wherever you want:

git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui

    Run webui.sh.
    Check webui-user.sh for options.
</official-documentation>

## directory links
How I want to map the directories from my computer to the docker image.
The docker image will run using Docker Desktop (windows) with the wsl2 support.

models -to-txt are located in "C:/Users/ApprenTyr/.lmstudio/models/"
For example: "C:/Users/ApprenTyr/.lmstudio/models/lmstudio-community/Qwen2.5-32B-Instruct-GGUF/Qwen2.5-32B-Instruct-Q4_K_M.gguf"

Models -to-image are located in "C:\Users\ApprenTyr\Documents\StabilityMatrix-win-x64\Data\Models\StableDiffusion" and should be mapped to "stable-diffusion-webui/models/Stable-diffusion"
Loras for images are located in "C:\Users\ApprenTyr\Documents\StabilityMatrix-win-x64\Data\Models\Lora" and should be mapped to "stable-diffusion-webui/models"

As models are heavy (30G), do not copy them, just mount them.

## requirements and requests
use python 3.13 and nvidia cuda 12.8

run the automatic1111 to check that it works. It should expose an endpoint for me to connect with my browser.
import the C:\Users\ApprenTyr\Documents\projects\youtube-package-generator\code_examples\llama_cpp_examples.py and run it. I want to see the output of what it does.