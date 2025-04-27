# Inference Manager for LocalAI

## Goal
Create a queue-based inference server that:
- Handles txt2txt, txt2img, img2img, img2txt requests
- Processes each queue type completely before switching
- Integrates with LocalAI via Docker
- Exposes FastAPI and MCP server interfaces

## Key Features
- Request queuing by type (4 queues)
- Priority processing (configurable order)
- Basic monitoring/logging
- Dockerized with LocalAI base

## Usage
1. Build the Docker image:
```bash
docker build -t inference-manager .
```

2. Run with LocalAI:
```bash
docker-compose up
```

3. Send requests to:
- `POST /v1/chat/completions` - Text generation
- `POST /v1/images/generations` - Image generation
- `POST /v1/vision/completions` - Image understanding