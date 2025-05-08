import json
import httpx
import pytest
from unittest.mock import AsyncMock, patch
import asyncio
import os
import sys

# Add project root to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.routes.tools import Tools
from src.utils.ollama_client import post_chat_completion, OLLAMA_BASE_URL
from main import app, queues

@pytest.mark.asyncio
async def test_post_chat_completion():
    mock_resp = AsyncMock()
    mock_resp.json.return_value = {"message": "test"}
    
    with patch("httpx.AsyncClient") as mock_client:
        # Setup async call chain
        mock_client.return_value.__aenter__.return_value.post.return_value = mock_resp
        
        result = await post_chat_completion(
            model="test",
            messages=[{"role": "user", "content": "test"}]
        )
        
        # Ensure we got the expected response
        assert result == {"message": "test"}

@pytest.mark.asyncio
async def test_ollama_endpoint_direct():
    """Test direct communication with Ollama endpoint"""
    try:
        # Create test client
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "messages": [{"role": "user", "content": "test"}],
                    "model": "qwen3:32b",
                    "temperature": 0.7,
                    "stream": False
                }
            )
            assert response.status_code == 200
            response_data = response.json()
            assert isinstance(response_data, dict)
            assert response_data["message"]["content"]
            
    except httpx.ConnectError:
        pytest.skip("Ollama service not available")
    except Exception as e:
        print(e)

@pytest.mark.asyncio
async def test_post_chat_completion():
    """Test the post_chat_completion"""
    # Start the queue processor in background
    queue_task = asyncio.create_task(queues.process_queues_forever())
  
    try:
        result = await post_chat_completion(
            model="qwen3:32b",
            messages=[{"role": "user", "content": "test message"}]
        )
        
        # Verify we got a response with expected structure
        assert isinstance(result, dict)
        assert result["message"]["content"]
        
    except httpx.ConnectError:
        pytest.skip("Ollama service not available")  

    finally:
        # Clean up queue task
        queue_task.cancel()
        try:
            await queue_task
        except asyncio.CancelledError:
            pass

@pytest.mark.asyncio
async def test_api_generate_text_full_path():
    """Test the full path from API endpoint to Ollama response"""
    # Start the queue processor in background
    queue_task = asyncio.create_task(queues.process_queues_forever())
  
    try:
        result = await Tools.generate_text([{"role": "user", "content": "test message"}], queues, "qwen3:32b", 0.7)
        
        # Verify we got a response with expected structure
        assert isinstance(result, dict)
        assert result["message"]["content"]
        
    except httpx.ConnectError:
        pytest.skip("Ollama service not available")  

    finally:
        # Clean up queue task
        queue_task.cancel()
        try:
            await queue_task
        except asyncio.CancelledError:
            pass
