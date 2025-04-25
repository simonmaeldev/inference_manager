#!/bin/bash

# Start llama.cpp server in the background
python -m llama_cpp.server --host 0.0.0.0 --port 8000 &
LLAMA_PID=$!

# Start stable-diffusion-webui
cd /app/stable-diffusion-webui
python launch.py --listen --port 7860 --api

# If stable-diffusion-webui terminates, kill llama.cpp server
kill $LLAMA_PID
