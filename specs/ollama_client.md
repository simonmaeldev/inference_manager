1. Update docker-compose.yml
```yaml
services:
  ollama:
    image: ollama/ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    volumes:
      - ollama_models:/root/.ollama
      - C:/Users/ApprenTyr/.ollama/models:/models
    environment:
      - OLLAMA_MODELS=/models
    ports:
      - "11434:11434"
    restart: unless-stopped

volumes:
  ollama_models:
```


2. Create src/utils/ollama_client.py
```python
import os
import httpx
from typing import Any, Dict, List

# Base URL for Ollama API - configured via OLLAMA_URL environment variable
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Timeout settings matching stability_matrix_client
API_TIMEOUT = httpx.Timeout(600.0, connect=120.0)

async def post_chat_completion(
    model: str, 
    messages: List[Dict[str, str]], 
    **kwargs
) -> Dict[str, Any]:
    """Send chat completion request to Ollama API
    
    Args:
        model: The model name to use for completion
        messages: List of message dicts with 'role' and 'content'
        **kwargs: Additional parameters like temperature
        
    Returns:
        Dict with completion result
        
    Raises:
        httpx.HTTPStatusError: If request fails
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        **kwargs
    }
    
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload
        )
        resp.raise_for_status()
        return resp.json()
```



3. Update src/models/schemas.py
```python
# Change import from:
from src.utils.localai_client import post_chat_completion
# To:
from src.utils.ollama_client import post_chat_completion
```


4. Update src/routes/tools.py
```python
# Update generate_text to use queues:
@staticmethod
async def generate_text(
    messages: list, 
    queues: InferenceQueues,
    model: str = "qwen3:32b", 
    temperature: float = 0.7
) -> Dict[str, Any]:
    """Generate text using the queue system"""
    req = Txt2TxtRequest(
        model=model,
        messages=messages,
        temperature=temperature
    )
    await queues.add_request(model, req)
    return await req.future
```


5. Update main.py
```
# Change endpoint from:
@app.post("/generate-text")
async def api_generate_text(messages: list, model: str = "gpt-4", temperature: float = 0.7):
    return await Tools.generate_text(messages, model, temperature)

# To:
@app.post("/api/chat")
async def api_generate_text(
    messages: list, 
    model: str = "qwen3:32b", 
    temperature: float = 0.7
):
    return await Tools.generate_text(messages, queues, model, temperature)

# Remove MCP tool:
@mcp.tool()
async def generate_text(...):  # Delete this entire function
```


6. Create tests/test_ollama_client.py
```python
import pytest
from unittest.mock import AsyncMock
from src.utils.ollama_client import post_chat_completion

@pytest.mark.asyncio
async def test_post_chat_completion(mocker):
    mock_resp = AsyncMock()
    mock_resp.json.return_value = {"message": "test"}
    mock_client = mocker.patch("httpx.AsyncClient")
    mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp
    
    result = await post_chat_completion(
        model="test",
        messages=[{"role": "user", "content": "test"}]
    )
    
    assert result == {"message": "test"}
```