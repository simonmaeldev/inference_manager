# Inference Manager for LocalAI [![CI](https://github.com/yourusername/inference_manager/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/inference_manager/actions/workflows/ci.yml)

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

### Quick Start
1. Build the Docker image:
```bash
docker build -t inference-manager .
```

2. Run with LocalAI:
```bash
docker-compose up
```

### API Documentation

The API provides the following endpoints through FastAPI and MCP:

#### Text-to-Text Generation
- **Endpoint**: `/txt2txt`
- **Method**: POST
- **Parameters**:
  - `model`: Supported model ID (e.g. "Qwen2.5-32B-Instruct")
  - `messages`: List of message dicts with "role" and "content"
  - `temperature`: Sampling temperature (0.0-2.0)
- **Example Request**:
  ```json
  {
    "model": "Qwen2.5-32B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7
  }
  ```

#### Text-to-Image Generation
- **Endpoint**: `/txt2img`
- **Method**: POST
- **Parameters**:
  - `model`: Supported model ID (e.g. "Flux-Dev")
  - `prompt`: Text description of desired image
  - `height/width`: Output dimensions in pixels
- **Example Response**: Base64 encoded image

#### Image-to-Image Generation
- **Endpoint**: `/img2img`
- **Method**: POST
- **Parameters**:
  - `model`: Supported model ID
  - `file`: Base64 encoded input image
  - `prompt`: Modification instructions
  - `height/width`: Output dimensions

#### Image-to-Text Generation
- **Endpoint**: `/img2txt`
- **Method**: POST
- **Parameters**:
  - `model`: Supported model ID
  - `messages`: Must include image in base64
  - `temperature`: Sampling temperature

#### Monitoring Endpoints
- `/get_error_metrics`: Get error statistics
- `/get_request_metrics`: Get performance metrics

### Interactive Documentation
The API includes an interactive OpenAPI/Swagger UI at `/docs` when running locally.

### Authentication
All endpoints require an API key in the `X-API-Key` header.

## Continuous Integration

Our CI pipeline runs on every push to main and pull requests, performing:

- Python 3.9, 3.10, and 3.11 matrix testing
- Dependency caching for faster builds
- Unit test execution with coverage reporting
- Integration test verification
- Automatic coverage upload to Codecov

To run tests locally:
```bash
pip install -r requirements.txt
pytest tests/ --cov=src --cov-report=term-missing

## Load Testing

We use [Locust](https://locust.io/) for load testing. The test scenarios include:

- **Concurrent request simulation**: Multiple users hitting endpoints simultaneously
- **Mixed workloads**: Different request types with realistic distributions
- **Performance metrics**: Throughput, latency, error rates collection

### Running Load Tests

1. Install locust:
```bash
pip install locust
```

2. Start the load test (from project root):
```bash
locust -f tests/load_test.py
```

3. Open web interface at http://localhost:8089 and configure:
   - Number of users
   - Spawn rate (users/second)
   - Host (http://localhost:8000 for local dev)

### Test Scenarios

- **Normal traffic**: 1-5 second wait between requests
- **Peak traffic**: 0.1-1 second wait between requests
- **Sustained load**: 5-10 second wait between requests

### Interpreting Results

The web interface shows:
- Requests per second (RPS)
- Response times (ms)
- Failure rates
- Charts of performance over time

For CI/CD integration, run in headless mode:
```bash
locust -f tests/load_test.py --headless --users 100 --spawn-rate 10 --run-time 1m
```