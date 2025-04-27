Quickstart

LocalAI is a free, open-source alternative to OpenAI (Anthropic, etc.), functioning as a drop-in replacement REST API for local inferencing. It allows you to run LLMs, generate images, and produce audio, all locally or on-premises with consumer-grade hardware, supporting multiple model families and architectures.

💡

Security considerations

If you are exposing LocalAI remotely, make sure you protect the API endpoints adequately with a mechanism which allows to protect from the incoming traffic or alternatively, run LocalAI with API_KEY to gate the access with an API key. The API key guarantees a total access to the features (there is no role separation), and it is to be considered as likely as an admin role.
Quickstart
Using the Bash Installer

# Basic installation
curl https://localai.io/install.sh | sh

See Installer for all the supported options
Run with docker:

# CPU only image:
docker run -ti --name local-ai -p 8080:8080 localai/localai:latest-cpu

# Nvidia GPU:
docker run -ti --name local-ai -p 8080:8080 --gpus all localai/localai:latest-gpu-nvidia-cuda-12

# CPU and GPU image (bigger size):
docker run -ti --name local-ai -p 8080:8080 localai/localai:latest

# AIO images (it will pre-download a set of models ready for use, see https://localai.io/basics/container/)
docker run -ti --name local-ai -p 8080:8080 localai/localai:latest-aio-cpu

Load models:

# From the model gallery (see available models with `local-ai models list`, in the WebUI from the model tab, or visiting https://models.localai.io)
local-ai run llama-3.2-1b-instruct:q4_k_m
# Start LocalAI with the phi-2 model directly from huggingface
local-ai run huggingface://TheBloke/phi-2-GGUF/phi-2.Q8_0.gguf
# Install and run a model from the Ollama OCI registry
local-ai run ollama://gemma:2b
# Run a model from a configuration file
local-ai run https://gist.githubusercontent.com/.../phi-2.yaml
# Install and run a model from a standard OCI registry (e.g., Docker Hub)
local-ai run oci://localai/phi-2:latest

For a full list of options, refer to the Installer Options documentation.

Binaries can also be manually downloaded.
Using Homebrew on MacOS

⚠️

The Homebrew formula currently doesn’t have the same options than the bash script

You can install Homebrew’s LocalAI

with the following command:

brew install localai

Using Container Images or Kubernetes

LocalAI is available as a container image compatible with various container engines such as Docker, Podman, and Kubernetes. Container images are published on quay.io
and Docker Hub

.

For detailed instructions, see Using container images. For Kubernetes deployment, see Run with Kubernetes.

Run with container images

LocalAI provides a variety of images to support different environments. These images are available on quay.io
and Docker Hub

.

All-in-One images comes with a pre-configured set of models and backends, standard images instead do not have any model pre-configured and installed.

For GPU Acceleration support for Nvidia video graphic cards, use the Nvidia/CUDA images, if you don’t have a GPU, use the CPU images. If you have AMD or Mac Silicon, see the build section.

💡

Available Images Types:

    Images ending with -core are smaller images without predownload python dependencies. Use these images if you plan to use llama.cpp, stablediffusion-ncn or rwkv backends - if you are not sure which one to use, do not use these images.

    Images containing the aio tag are all-in-one images with all the features enabled, and come with an opinionated set of configuration.

    FFMpeg is not included in the default images due to its licensing

    . If you need FFMpeg, use the images ending with -ffmpeg. Note that ffmpeg is needed in case of using audio-to-text LocalAI’s features.

    If using old and outdated CPUs and no GPUs you might need to set REBUILD to true as environment variable along with options to disable the flags which your CPU does not support, however note that inference will perform poorly and slow. See also flagset compatibility.

Prerequisites

Before you begin, ensure you have a container engine installed if you are not using the binaries. Suitable options include Docker or Podman. For installation instructions, refer to the following guides:

    Install Docker Desktop (Mac, Windows, Linux)

Install Podman (Linux)
Install Docker engine (Servers)

💡

Hardware Requirements: The hardware requirements for LocalAI vary based on the model size and quantization method used. For performance benchmarks with different backends, such as llama.cpp, visit this link

. The rwkv backend is noted for its lower resource consumption.
All-in-one images

All-In-One images are images that come pre-configured with a set of models and backends to fully leverage almost all the LocalAI featureset. These images are available for both CPU and GPU environments. The AIO images are designed to be easy to use and requires no configuration. Models configuration can be found here

separated by size.

In the AIO images there are models configured with the names of OpenAI models, however, they are really backed by Open Source models. You can find the table below
Category	Model name	Real model (CPU)	Real model (GPU)
Text Generation	gpt-4	phi-2	hermes-2-pro-mistral
Multimodal Vision	gpt-4-vision-preview	bakllava	llava-1.6-mistral
Image Generation	stablediffusion	stablediffusion	dreamshaper-8
Speech to Text	whisper-1	whisper with whisper-base model	<= same
Text to Speech	tts-1	en-us-amy-low.onnx from rhasspy/piper	<= same
Embeddings	text-embedding-ada-002	all-MiniLM-L6-v2 in Q4	all-MiniLM-L6-v2
Usage

Select the image (CPU or GPU) and start the container with Docker:

# CPU example
docker run -p 8080:8080 --name local-ai -ti localai/localai:latest-aio-cpu
# For Nvidia GPUs:
# docker run -p 8080:8080 --gpus all --name local-ai -ti localai/localai:latest-aio-gpu-nvidia-cuda-11
# docker run -p 8080:8080 --gpus all --name local-ai -ti localai/localai:latest-aio-gpu-nvidia-cuda-12

LocalAI will automatically download all the required models, and the API will be available at localhost:8080

.

Or with a docker-compose file:

version: "3.9"
services:
  api:
    image: localai/localai:latest-aio-cpu
    # For a specific version:
    # image: localai/localai:v2.28.0-aio-cpu
    # For Nvidia GPUs decomment one of the following (cuda11 or cuda12):
    # image: localai/localai:v2.28.0-aio-gpu-nvidia-cuda-11
    # image: localai/localai:v2.28.0-aio-gpu-nvidia-cuda-12
    # image: localai/localai:latest-aio-gpu-nvidia-cuda-11
    # image: localai/localai:latest-aio-gpu-nvidia-cuda-12
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/readyz"]
      interval: 1m
      timeout: 20m
      retries: 5
    ports:
      - 8080:8080
    environment:
      - DEBUG=true
      # ...
    volumes:
      - ./models:/build/models:cached
    # decomment the following piece if running with Nvidia GPUs
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - driver: nvidia
    #           count: 1
    #           capabilities: [gpu]

💡

Models caching: The AIO image will download the needed models on the first run if not already present and store those in /build/models inside the container. The AIO models will be automatically updated with new versions of AIO images.

You can change the directory inside the container by specifying a MODELS_PATH environment variable (or --models-path).

If you want to use a named model or a local directory, you can mount it as a volume to /build/models:

docker run -p 8080:8080 --name local-ai -ti -v $PWD/models:/build/models localai/localai:latest-aio-cpu

or associate a volume:

docker volume create localai-models
docker run -p 8080:8080 --name local-ai -ti -v localai-models:/build/models localai/localai:latest-aio-cpu

Available AIO images
Description	Quay	Docker Hub
Latest images for CPU	quay.io/go-skynet/local-ai:latest-aio-cpu	localai/localai:latest-aio-cpu
Versioned image (e.g. for CPU)	quay.io/go-skynet/local-ai:v2.28.0-aio-cpu	localai/localai:v2.28.0-aio-cpu
Latest images for Nvidia GPU (CUDA11)	quay.io/go-skynet/local-ai:latest-aio-gpu-nvidia-cuda-11	localai/localai:latest-aio-gpu-nvidia-cuda-11
Latest images for Nvidia GPU (CUDA12)	quay.io/go-skynet/local-ai:latest-aio-gpu-nvidia-cuda-12	localai/localai:latest-aio-gpu-nvidia-cuda-12
Latest images for AMD GPU	quay.io/go-skynet/local-ai:latest-aio-gpu-hipblas	localai/localai:latest-aio-gpu-hipblas
Latest images for Intel GPU (sycl f16)	quay.io/go-skynet/local-ai:latest-aio-gpu-intel-f16	localai/localai:latest-aio-gpu-intel-f16
Latest images for Intel GPU (sycl f32)	quay.io/go-skynet/local-ai:latest-aio-gpu-intel-f32	localai/localai:latest-aio-gpu-intel-f32
Available environment variables

The AIO Images are inheriting the same environment variables as the base images and the environment of LocalAI (that you can inspect by calling --help). However, it supports additional environment variables available only from the container image
Variable	Default	Description
PROFILE	Auto-detected	The size of the model to use. Available: cpu, gpu-8g
MODELS	Auto-detected	A list of models YAML Configuration file URI/URL (see also running models)
Standard container images

Standard container images do not have pre-installed models.

Images are available with and without python dependencies. Note that images with python dependencies are bigger (in order of 17GB).

Images with core in the tag are smaller and do not contain any python dependencies.
Vanilla / CPU Images
GPU Images CUDA 11
GPU Images CUDA 12
Intel GPU (sycl f16)
Intel GPU (sycl f32)
AMD GPU
Vulkan Images
Nvidia Linux for tegra
Description	Quay	Docker Hub
Latest images from the branch (development)	quay.io/go-skynet/local-ai:master	localai/localai:master
Latest tag	quay.io/go-skynet/local-ai:latest-cpu	localai/localai:latest-cpu
Versioned image	quay.io/go-skynet/local-ai:v2.28.0	localai/localai:v2.28.0
Versioned image including FFMpeg	quay.io/go-skynet/local-ai:v2.28.0-ffmpeg	localai/localai:v2.28.0-ffmpeg
Versioned image including FFMpeg, no python	quay.io/go-skynet/local-ai:v2.28.0-ffmpeg-core	localai/localai:v2.28.0-ffmpeg-core
CUDA(NVIDIA) acceleration
Requirements

Requirement: nvidia-container-toolkit (installation instructions 1
2

)

To check what CUDA version do you need, you can either run nvidia-smi or nvcc --version.

Alternatively, you can also check nvidia-smi with docker:

docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi

To use CUDA, use the images with the cublas tag, for example.

The image list is on quay

:

    CUDA 11 tags: master-cublas-cuda11, v1.40.0-cublas-cuda11, …
    CUDA 12 tags: master-cublas-cuda12, v1.40.0-cublas-cuda12, …
    CUDA 11 + FFmpeg tags: master-cublas-cuda11-ffmpeg, v1.40.0-cublas-cuda11-ffmpeg, …
    CUDA 12 + FFmpeg tags: master-cublas-cuda12-ffmpeg, v1.40.0-cublas-cuda12-ffmpeg, …

In addition to the commands to run LocalAI normally, you need to specify --gpus all to docker, for example:

docker run --rm -ti --gpus all -p 8080:8080 -e DEBUG=true -e MODELS_PATH=/models -e THREADS=1 -v $PWD/models:/models quay.io/go-skynet/local-ai:v1.40.0-cublas-cuda12

If the GPU inferencing is working, you should be able to see something like:

5:22PM DBG Loading model in memory from file: /models/open-llama-7b-q4_0.bin
ggml_init_cublas: found 1 CUDA devices:
  Device 0: Tesla T4
llama.cpp: loading model from /models/open-llama-7b-q4_0.bin
llama_model_load_internal: format     = ggjt v3 (latest)
llama_model_load_internal: n_vocab    = 32000
llama_model_load_internal: n_ctx      = 1024
llama_model_load_internal: n_embd     = 4096
llama_model_load_internal: n_mult     = 256
llama_model_load_internal: n_head     = 32
llama_model_load_internal: n_layer    = 32
llama_model_load_internal: n_rot      = 128
llama_model_load_internal: ftype      = 2 (mostly Q4_0)
llama_model_load_internal: n_ff       = 11008
llama_model_load_internal: n_parts    = 1
llama_model_load_internal: model size = 7B
llama_model_load_internal: ggml ctx size =    0.07 MB
llama_model_load_internal: using CUDA for GPU acceleration
llama_model_load_internal: mem required  = 4321.77 MB (+ 1026.00 MB per state)
llama_model_load_internal: allocating batch_size x 1 MB = 512 MB VRAM for the scratch buffer
llama_model_load_internal: offloading 10 repeating layers to GPU
llama_model_load_internal: offloaded 10/35 layers to GPU
llama_model_load_internal: total VRAM used: 1598 MB
...................................................................................................
llama_init_from_file: kv self size  =  512.00 MB

Text generation (GPT)

LocalAI supports generating text with GPT with llama.cpp and other backends (such as rwkv.cpp as ) see also the Model compatibility for an up-to-date list of the supported model families.

Note:

    You can also specify the model name as part of the OpenAI token.
    If only one model is available, the API will use it for all the requests.

API Reference
Chat completions

https://platform.openai.com/docs/api-reference/chat

For example, to generate a chat completion, you can send a POST request to the /v1/chat/completions endpoint with the instruction as the request body:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "ggml-koala-7b-model-q4_0-r2.bin",
  "messages": [{"role": "user", "content": "Say this is a test!"}],
  "temperature": 0.7
}'

Available additional parameters: top_p, top_k, max_tokens
Edit completions

https://platform.openai.com/docs/api-reference/edits

To generate an edit completion you can send a POST request to the /v1/edits endpoint with the instruction as the request body:

curl http://localhost:8080/v1/edits -H "Content-Type: application/json" -d '{
  "model": "ggml-koala-7b-model-q4_0-r2.bin",
  "instruction": "rephrase",
  "input": "Black cat jumped out of the window",
  "temperature": 0.7
}'

Available additional parameters: top_p, top_k, max_tokens.
Completions

https://platform.openai.com/docs/api-reference/completions

To generate a completion, you can send a POST request to the /v1/completions endpoint with the instruction as per the request body:

curl http://localhost:8080/v1/completions -H "Content-Type: application/json" -d '{
  "model": "ggml-koala-7b-model-q4_0-r2.bin",
  "prompt": "A long time ago in a galaxy far, far away",
  "temperature": 0.7
}'

Available additional parameters: top_p, top_k, max_tokens
List models

You can list all the models available with:

curl http://localhost:8080/v1/models

Backends
RWKV

RWKV support is available through llama.cpp (see below)
llama.cpp

llama.cpp

is a popular port of Facebook’s LLaMA model in C/C++.
notifications

The ggml file format has been deprecated. If you are using ggml models and you are configuring your model with a YAML file, specify, use a LocalAI version older than v2.25.0. For gguf models, use the llama backend. The go backend is deprecated as well but still available as go-llama.
Features

The llama.cpp model supports the following features:

    📖 Text generation (GPT)
    🧠 Embeddings
    🔥 OpenAI functions
    ✍️ Constrained grammars

Setup

LocalAI supports llama.cpp models out of the box. You can use the llama.cpp model in the same way as any other model.
Manual setup

It is sufficient to copy the ggml or gguf model files in the models folder. You can refer to the model in the model parameter in the API calls.

You can optionally create an associated YAML model config file to tune the model’s parameters or apply a template to the prompt.

Prompt templates are useful for models that are fine-tuned towards a specific prompt.
Automatic setup

LocalAI supports model galleries which are indexes of models. For instance, the huggingface gallery contains a large curated index of models from the huggingface model hub for ggml or gguf models.

For instance, if you have the galleries enabled and LocalAI already running, you can just start chatting with models in huggingface by running:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
   "model": "TheBloke/WizardLM-13B-V1.2-GGML/wizardlm-13b-v1.2.ggmlv3.q2_K.bin",
   "messages": [{"role": "user", "content": "Say this is a test!"}],
   "temperature": 0.1
 }'

LocalAI will automatically download and configure the model in the model directory.

Models can be also preloaded or downloaded on demand. To learn about model galleries, check out the model gallery documentation.
YAML configuration

To use the llama.cpp backend, specify llama as the backend in the YAML file:

name: llama
backend: llama
parameters:
  # Relative to the models path
  model: file.gguf

Reference

    llama

exllama/2

Exllama

is a “A more memory-efficient rewrite of the HF transformers implementation of Llama for use with quantized weights”. Both exllama and exllama2 are supported.
Model setup

Download the model as a folder inside the model directory and create a YAML file specifying the exllama backend. For instance with the TheBloke/WizardLM-7B-uncensored-GPTQ model:

$ git lfs install
$ cd models && git clone https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GPTQ
$ ls models/
.keep                        WizardLM-7B-uncensored-GPTQ/ exllama.yaml
$ cat models/exllama.yaml
name: exllama
parameters:
  model: WizardLM-7B-uncensored-GPTQ
backend: exllama
# Note: you can also specify "exllama2" if it's an exllama2 model here
# ...

Test with:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "exllama",
  "messages": [{"role": "user", "content": "How are you?"}],
  "temperature": 0.1
}'

vLLM

vLLM

is a fast and easy-to-use library for LLM inference.

LocalAI has a built-in integration with vLLM, and it can be used to run models. You can check out vllm performance here

.
Setup

Create a YAML file for the model you want to use with vllm.

To setup a model, you need to just specify the model name in the YAML config file:

name: vllm
backend: vllm
parameters:
    model: "facebook/opt-125m"

# Uncomment to specify a quantization method (optional)
# quantization: "awq"
# Uncomment to limit the GPU memory utilization (vLLM default is 0.9 for 90%)
# gpu_memory_utilization: 0.5
# Uncomment to trust remote code from huggingface
# trust_remote_code: true
# Uncomment to enable eager execution
# enforce_eager: true
# Uncomment to specify the size of the CPU swap space per GPU (in GiB)
# swap_space: 2
# Uncomment to specify the maximum length of a sequence (including prompt and output)
# max_model_len: 32768
# Uncomment and specify the number of Tensor divisions.
# Allows you to partition and run large models. Performance gains are limited.
# https://github.com/vllm-project/vllm/issues/1435
# tensor_parallel_size: 2

The backend will automatically download the required files in order to run the model.
Usage

Use the completions endpoint by specifying the vllm backend:

curl http://localhost:8080/v1/completions -H "Content-Type: application/json" -d '{
  "model": "vllm",
  "prompt": "Hello, my name is",
  "temperature": 0.1, "top_p": 0.1
}'

Transformers

Transformers

is a State-of-the-art Machine Learning library for PyTorch, TensorFlow, and JAX.

LocalAI has a built-in integration with Transformers, and it can be used to run models.

This is an extra backend - in the container images (the extra images already contains python dependencies for Transformers) is already available and there is nothing to do for the setup.
Setup

Create a YAML file for the model you want to use with transformers.

To setup a model, you need to just specify the model name in the YAML config file:

name: transformers
backend: transformers
parameters:
    model: "facebook/opt-125m"
type: AutoModelForCausalLM
quantization: bnb_4bit # One of: bnb_8bit, bnb_4bit, xpu_4bit, xpu_8bit (optional)

The backend will automatically download the required files in order to run the model.
Parameters
Type
Type	Description
AutoModelForCausalLM	AutoModelForCausalLM is a model that can be used to generate sequences. Use it for NVIDIA CUDA and Intel GPU with Intel Extensions for Pytorch acceleration
OVModelForCausalLM	for Intel CPU/GPU/NPU OpenVINO Text Generation models
OVModelForFeatureExtraction	for Intel CPU/GPU/NPU OpenVINO Embedding acceleration
N/A	Defaults to AutoModel

    OVModelForCausalLM requires OpenVINO IR Text Generation

models from Hugging face
OVModelForFeatureExtraction works with any Safetensors Transformer Feature Extraction

    model from Huggingface (Embedding Model)

Please note that streaming is currently not implemente in AutoModelForCausalLM for Intel GPU. AMD GPU support is not implemented. Although AMD CPU is not officially supported by OpenVINO there are reports that it works: YMMV.
Embeddings

Use embeddings: true if the model is an embedding model
Inference device selection

Transformer backend tries to automatically select the best device for inference, anyway you can override the decision manually overriding with the main_gpu parameter.
Inference Engine	Applicable Values
CUDA	cuda, cuda.X where X is the GPU device like in nvidia-smi -L output
OpenVINO	Any applicable value from Inference Modes
like AUTO,CPU,GPU,NPU,MULTI,HETERO

Example for CUDA: main_gpu: cuda.0

Example for OpenVINO: main_gpu: AUTO:-CPU

This parameter applies to both Text Generation and Feature Extraction (i.e. Embeddings) models.
Inference Precision

Transformer backend automatically select the fastest applicable inference precision according to the device support. CUDA backend can manually enable bfloat16 if your hardware support it with the following parameter:

f16: true
Quantization
Quantization	Description
bnb_8bit	8-bit quantization
bnb_4bit	4-bit quantization
xpu_8bit	8-bit quantization for Intel XPUs
xpu_4bit	4-bit quantization for Intel XPUs
Trust Remote Code

Some models like Microsoft Phi-3 requires external code than what is provided by the transformer library. By default it is disabled for security. It can be manually enabled with: trust_remote_code: true
Maximum Context Size

Maximum context size in bytes can be specified with the parameter: context_size. Do not use values higher than what your model support.

Usage example: context_size: 8192
Auto Prompt Template

Usually chat template is defined by the model author in the tokenizer_config.json file. To enable it use the use_tokenizer_template: true parameter in the template section.

Usage example:

template:
use_tokenizer_template: true

Custom Stop Words

Stopwords are usually defined in tokenizer_config.json file. They can be overridden with the stopwords parameter in case of need like in llama3-Instruct model.

Usage example:

stopwords:
- "<|eot_id|>"
- "<|end_of_text|>"

Usage

Use the completions endpoint by specifying the transformers model:

curl http://localhost:8080/v1/completions -H "Content-Type: application/json" -d '{
  "model": "transformers",
  "prompt": "Hello, my name is",
  "temperature": 0.1, "top_p": 0.1
}'

Examples
OpenVINO

A model configuration file for openvion and starling model:

name: starling-openvino
backend: transformers
parameters:
  model: fakezeta/Starling-LM-7B-beta-openvino-int8
context_size: 8192
threads: 6
f16: true
type: OVModelForCausalLM
stopwords:
- <|end_of_turn|>
- <|endoftext|>
prompt_cache_path: "cache"
prompt_cache_all: true
template:
  chat_message: |
    {{if eq .RoleName "system"}}{{.Content}}<|end_of_turn|>{{end}}{{if eq .RoleName "assistant"}}<|end_of_turn|>GPT4 Correct Assistant: {{.Content}}<|end_of_turn|>{{end}}{{if eq .RoleName "user"}}GPT4 Correct User: {{.Content}}{{end}}

  chat: |
    {{.Input}}<|end_of_turn|>GPT4 Correct Assistant:

  completion: |
    {{.Input}}

Image generation

anime_girl (Generated with AnimagineXL

)

LocalAI supports generating images with Stable diffusion, running on CPU using C++ and Python implementations.
Usage

OpenAI docs: https://platform.openai.com/docs/api-reference/images/create

To generate an image you can send a POST request to the /v1/images/generations endpoint with the instruction as the request body:

# 512x512 is supported too
curl http://localhost:8080/v1/images/generations -H "Content-Type: application/json" -d '{
  "prompt": "A cute baby sea otter",
  "size": "256x256"
}'

Available additional parameters: mode, step.

Note: To set a negative prompt, you can split the prompt with |, for instance: a cute baby sea otter|malformed.

curl http://localhost:8080/v1/images/generations -H "Content-Type: application/json" -d '{
  "prompt": "floating hair, portrait, ((loli)), ((one girl)), cute face, hidden hands, asymmetrical bangs, beautiful detailed eyes, eye shadow, hair ornament, ribbons, bowties, buttons, pleated skirt, (((masterpiece))), ((best quality)), colorful|((part of the head)), ((((mutated hands and fingers)))), deformed, blurry, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus, long neck, long body, Octane renderer, lowres, bad anatomy, bad hands, text",
  "size": "256x256"
}'

Backends
stablediffusion-ggml

This backend is based on stable-diffusion.cpp

. Every model supported by that backend is suppoerted indeed with LocalAI.
Setup

There are already several models in the gallery that are available to install and get up and running with this backend, you can for example run flux by searching it in the Model gallery (flux.1-dev-ggml) or start LocalAI with run:

local-ai run flux.1-dev-ggml

To use a custom model, you can follow these steps:

    Create a model file stablediffusion.yaml in the models folder:

name: stablediffusion
backend: stablediffusion-ggml
parameters:
  model: gguf_model.gguf
step: 25
cfg_scale: 4.5
options:
- "clip_l_path:clip_l.safetensors"
- "clip_g_path:clip_g.safetensors"
- "t5xxl_path:t5xxl-Q5_0.gguf"
- "sampler:euler"

    Download the required assets to the models repository
    Start LocalAI

Diffusers

Diffusers

is the go-to library for state-of-the-art pretrained diffusion models for generating images, audio, and even 3D structures of molecules. LocalAI has a diffusers backend which allows image generation using the diffusers library.

anime_girl (Generated with AnimagineXL

)
Model setup

The models will be downloaded the first time you use the backend from huggingface automatically.

Create a model configuration file in the models directory, for instance to use Linaqruf/animagine-xl with CPU:

name: animagine-xl
parameters:
  model: Linaqruf/animagine-xl
backend: diffusers

# Force CPU usage - set to true for GPU
f16: false
diffusers:
  cuda: false # Enable for GPU usage (CUDA)
  scheduler_type: euler_a

Dependencies

This is an extra backend - in the container is already available and there is nothing to do for the setup. Do not use core images (ending with -core). If you are building manually, see the build instructions.
Model setup

The models will be downloaded the first time you use the backend from huggingface automatically.

Create a model configuration file in the models directory, for instance to use Linaqruf/animagine-xl with CPU:

name: animagine-xl
parameters:
  model: Linaqruf/animagine-xl
backend: diffusers
cuda: true
f16: true
diffusers:
  scheduler_type: euler_a

Local models

You can also use local models, or modify some parameters like clip_skip, scheduler_type, for instance:

name: stablediffusion
parameters:
  model: toonyou_beta6.safetensors
backend: diffusers
step: 30
f16: true
cuda: true
diffusers:
  pipeline_type: StableDiffusionPipeline
  enable_parameters: "negative_prompt,num_inference_steps,clip_skip"
  scheduler_type: "k_dpmpp_sde"
  clip_skip: 11

cfg_scale: 8

Configuration parameters

The following parameters are available in the configuration file:
Parameter	Description	Default
f16	Force the usage of float16 instead of float32	false
step	Number of steps to run the model for	30
cuda	Enable CUDA acceleration	false
enable_parameters	Parameters to enable for the model	negative_prompt,num_inference_steps,clip_skip
scheduler_type	Scheduler type	k_dpp_sde
cfg_scale	Configuration scale	8
clip_skip	Clip skip	None
pipeline_type	Pipeline type	AutoPipelineForText2Image
lora_adapters	A list of lora adapters (file names relative to model directory) to apply	None
lora_scales	A list of lora scales (floats) to apply	None

There are available several types of schedulers:
Scheduler	Description
ddim	DDIM
pndm	PNDM
heun	Heun
unipc	UniPC
euler	Euler
euler_a	Euler a
lms	LMS
k_lms	LMS Karras
dpm_2	DPM2
k_dpm_2	DPM2 Karras
dpm_2_a	DPM2 a
k_dpm_2_a	DPM2 a Karras
dpmpp_2m	DPM++ 2M
k_dpmpp_2m	DPM++ 2M Karras
dpmpp_sde	DPM++ SDE
k_dpmpp_sde	DPM++ SDE Karras
dpmpp_2m_sde	DPM++ 2M SDE
k_dpmpp_2m_sde	DPM++ 2M SDE Karras

Pipelines types available:
Pipeline type	Description
StableDiffusionPipeline	Stable diffusion pipeline
StableDiffusionImg2ImgPipeline	Stable diffusion image to image pipeline
StableDiffusionDepth2ImgPipeline	Stable diffusion depth to image pipeline
DiffusionPipeline	Diffusion pipeline
StableDiffusionXLPipeline	Stable diffusion XL pipeline
StableVideoDiffusionPipeline	Stable video diffusion pipeline
AutoPipelineForText2Image	Automatic detection pipeline for text to image
VideoDiffusionPipeline	Video diffusion pipeline
StableDiffusion3Pipeline	Stable diffusion 3 pipeline
FluxPipeline	Flux pipeline
FluxTransformer2DModel	Flux transformer 2D model
SanaPipeline	Sana pipeline
Advanced: Additional parameters

Additional arbitrarly parameters can be specified in the option field in key/value separated by ::

name: animagine-xl
# ...
options:
- "cfg_scale:6"

Note: There is no complete parameter list. Any parameter can be passed arbitrarly and is passed to the model directly as argument to the pipeline. Different pipelines/implementations support different parameters.

The example above, will result in the following python code when generating images:

pipe(
    prompt="A cute baby sea otter", # Options passed via API
    size="256x256", # Options passed via API
    cfg_scale=6 # Additional parameter passed via configuration file
)

Usage
Text to Image

Use the image generation endpoint with the model name from the configuration file:

curl http://localhost:8080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "<positive prompt>|<negative prompt>",
    "model": "animagine-xl",
    "step": 51,
    "size": "1024x1024"
  }'

Image to Image

https://huggingface.co/docs/diffusers/using-diffusers/img2img

An example model (GPU):

name: stablediffusion-edit
parameters:
  model: nitrosocke/Ghibli-Diffusion
backend: diffusers
step: 25
cuda: true
f16: true
diffusers:
  pipeline_type: StableDiffusionImg2ImgPipeline
  enable_parameters: "negative_prompt,num_inference_steps,image"

IMAGE_PATH=/path/to/your/image
(echo -n '{"file": "'; base64 $IMAGE_PATH; echo '", "prompt": "a sky background","size": "512x512","model":"stablediffusion-edit"}') |
curl -H "Content-Type: application/json" -d @-  http://localhost:8080/v1/images/generations

Depth to Image

https://huggingface.co/docs/diffusers/using-diffusers/depth2img

name: stablediffusion-depth
parameters:
  model: stabilityai/stable-diffusion-2-depth
backend: diffusers
step: 50
# Force CPU usage
f16: true
cuda: true
diffusers:
  pipeline_type: StableDiffusionDepth2ImgPipeline
  enable_parameters: "negative_prompt,num_inference_steps,image"

cfg_scale: 6

(echo -n '{"file": "'; base64 ~/path/to/image.jpeg; echo '", "prompt": "a sky background","size": "512x512","model":"stablediffusion-depth"}') |
curl -H "Content-Type: application/json" -d @-  http://localhost:8080/v1/images/generations

img2vid

name: img2vid
parameters:
  model: stabilityai/stable-video-diffusion-img2vid
backend: diffusers
step: 25
# Force CPU usage
f16: true
cuda: true
diffusers:
  pipeline_type: StableVideoDiffusionPipeline

(echo -n '{"file": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true","size": "512x512","model":"img2vid"}') |
curl -H "Content-Type: application/json" -X POST -d @- http://localhost:8080/v1/images/generations

txt2vid

name: txt2vid
parameters:
  model: damo-vilab/text-to-video-ms-1.7b
backend: diffusers
step: 25
# Force CPU usage
f16: true
cuda: true
diffusers:
  pipeline_type: VideoDiffusionPipeline
  cuda: true

(echo -n '{"prompt": "spiderman surfing","size": "512x512","model":"txt2vid"}') |
curl -H "Content-Type: application/json" -X POST -d @- http://localhost:8080/v1/images/generations


GPT Vision

LocalAI supports understanding images by using LLaVA
, and implements the GPT Vision API

from OpenAI.

llava
Usage

OpenAI docs: https://platform.openai.com/docs/guides/vision

To let LocalAI understand and reply with what sees in the image, use the /v1/chat/completions endpoint, for example with curl:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
   "model": "llava",
   "messages": [{"role": "user", "content": [{"type":"text", "text": "What is in the image?"}, {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg" }}], "temperature": 0.9}]}'

Grammars and function tools can be used as well in conjunction with vision APIs:

curl http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "llava", "grammar": "root ::= (\"yes\" | \"no\")",
  "messages": [{"role": "user", "content": [{"type":"text", "text": "Is there some grass in the image?"}, {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg" }}], "temperature": 0.9}]}'

Setup

All-in-One images have already shipped the llava model as gpt-4-vision-preview, so no setup is needed in this case.

To setup the LLaVa models, follow the full example in the configuration examples