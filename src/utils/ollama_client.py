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

async def get_ollama_tags() -> Dict[str, Any]:
    """Get available models/tags from Ollama API
    
    Returns:
        Dict with models list from Ollama
        
    Raises:
        httpx.HTTPStatusError: If request fails
    """
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        resp.raise_for_status()
        return resp.json()