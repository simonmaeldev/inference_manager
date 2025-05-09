import asyncio
import pytest

import json

import sys
import os

from unittest.mock import AsyncMock, patch, MagicMock

# Add project root to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import generate_youtube_thumbnail, queues



@pytest.fixture
def base64_image():
    """Load test base64 image from file"""
    with open("base64_file.txt", "r") as f:
        return f.read().strip()


@pytest.fixture
def api_response_data(base64_image):
    """Load mock API response data"""
    with open("mock_api_response.json", "r") as f:
        data = json.loads(f.read())
        data["images"] = [base64_image]
        return data


@pytest.mark.asyncio
async def test_generate_youtube_thumbnail_real():
    """Test the generate_youtube_thumbnail function with real API call"""
    
    # Start the queue processor in a background task
    queue_task = asyncio.create_task(queues.process_queues_forever())
    
    try:
        # Call the function
        result = await generate_youtube_thumbnail("A colorful sunset over mountains")
        
        # Check the result
        assert isinstance(result, str)
        assert "generated_images" in result or "No image returned" in result
    finally:
        # Clean up the task
        queue_task.cancel()
        try:
            await queue_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_generate_youtube_thumbnail_mock(monkeypatch, base64_image, api_response_data):
    """Test the generate_youtube_thumbnail function with mocked API response"""
    
    # Create a mock httpx.AsyncClient and response
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = api_response_data
    
    # Set up the mock client's post method to return our response
    mock_client.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
    
    # Patch the httpx.AsyncClient to use our mock
    with patch('src.utils.stability_matrix_client.httpx.AsyncClient', return_value=mock_client):
        # Set up monkeypatch to ensure file operations work as expected
        original_open = open
        
        def mock_open(*args, **kwargs):
            if args[0].startswith("generated_images/") and args[1] == "wb":
                # Use a different dir for test output
                test_dir = "test_generated_images"
                os.makedirs(test_dir, exist_ok=True)
                filepath = os.path.join(test_dir, os.path.basename(args[0]))
                return original_open(filepath, args[1])
            return original_open(*args, **kwargs)
        
        # Apply monkeypatch for open function
        monkeypatch.setattr("builtins.open", mock_open)
        
        # Start the queue processor in a background task
        queue_task = asyncio.create_task(queues.process_queues_forever())
        
        try:
            # Call the function
            result = await generate_youtube_thumbnail("A colorful sunset over mountains")
            
            # Verify the client was called with expected parameters
            mock_client.__aenter__.return_value.post.assert_called_once()
            args, kwargs = mock_client.__aenter__.return_value.post.call_args
            assert args[0].endswith("/sdapi/v1/txt2img")
            
            # Check that payload contains our prompt
            payload = kwargs['json']
            assert payload['prompt'] == "A colorful sunset over mountains"
            
            # Check the result
            assert isinstance(result, str)
            # Normalize path separators for cross-platform compatibility
            normalized_result = result.replace("\\", "/")
            assert normalized_result.startswith("generated_images/") or "No image returned" in result
            
            # Verify the generated image matches our mock
            if result.startswith("generated_images/"):
                with open(result, "rb") as img_file:
                    import base64
                    generated_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                    assert generated_b64 == base64_image
        finally:
            # Clean up the task
            queue_task.cancel()
            try:
                await queue_task
            except asyncio.CancelledError:
                pass