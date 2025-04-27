import os
import httpx

# Base URL for LocalAI API - can be configured via LOCALAI_URL environment variable
# Defaults to "http://localhost:8080" if not set
LOCALAI_BASE_URL = os.getenv("LOCALAI_URL", "http://localhost:8080")
from typing import Any, Dict, List

async def post_chat_completion(model: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
    """Send chat completion request to LocalAI API
    
    Args:
        model: The model name to use for completion
        messages: List of message dicts with 'role' and 'content'
        **kwargs: Additional parameters like temperature
        
    Returns:
        Dict with completion result
        
    Raises:
        httpx.HTTPStatusError: If request fails
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{LOCALAI_BASE_URL}/v1/chat/completions",
            json={"model": model, "messages": messages, **kwargs}
        )
        resp.raise_for_status()
        return resp.json()

async def post_image_generation(model: str, prompt: str, **kwargs) -> Dict[str, Any]:
    """Send image generation request to LocalAI API
    
    Args:
        model: The model name to use for generation
        prompt: Text prompt for image generation
        **kwargs: Additional parameters like height, width
        
    Returns:
        Dict with image generation result
        
    Raises:
        httpx.HTTPStatusError: If request fails
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{LOCALAI_BASE_URL}/v1/images/generations",
            json={"model": model, "prompt": prompt, **kwargs}
        )
        resp.raise_for_status()
        return resp.json()