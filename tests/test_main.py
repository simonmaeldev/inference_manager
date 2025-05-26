import asyncio
import pytest
import time
from typing import List, Dict, Any
import sys
import os
import httpx

# Add project root to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app, queues, generate_youtube_thumbnail
from src.routes.tools import Tools

@pytest.mark.asyncio
async def test_concurrent_processing_order():
    """Test that text inference queue is processed before image generation queue under load."""
    # Start the queue processor in background
    queue_task = asyncio.create_task(queues.process_queues_forever())
    
    try:
        # Track processing order and results
        processing_order = []
        results = []
        
        # Create async tasks for both types of requests
        async def make_text_request(i: int):
            start = time.time()
            result = await Tools.generate_text(
                [{"role": "user", "content": f"Test message {i}"}],
                queues,
                "qwen3:32b",
                0.7
            )
            end = time.time()
            processing_order.append(("text", i, start, end))
            results.append(result)
            return result
            
        async def make_image_request(i: int):
            start = time.time()
            result = await generate_youtube_thumbnail(f"Test image prompt {i}")
            end = time.time()
            processing_order.append(("image", i, start, end))
            results.append(result)
            return result
            
        # Create 10 of each request type, alternating starting with image
        tasks = []
        for i in range(5):
            tasks.append(make_image_request(i))
            tasks.append(make_text_request(i))
            
        # Run all requests concurrently
        await asyncio.gather(*tasks)
        
        # Verify all requests completed successfully
        assert len(results) == 20
        for result in results:
            assert result is not None
            
        # Analyze processing order - text requests should complete before image requests
        # Group by type and get average completion time
        text_times = [end for (typ, i, start, end) in processing_order if typ == "text"]
        image_times = [end for (typ, i, start, end) in processing_order if typ == "image"]
        
        # Assert text requests completed before image requests on average
        # This verifies the queue priority behavior
        assert sum(text_times)/len(text_times) < sum(image_times)/len(image_times), \
            "Text requests should complete before image requests"
            
        # Print detailed timing info for debugging
        print("\nProcessing order analysis:")
        for entry in sorted(processing_order, key=lambda x: x[2]):  # Sort by start time
            typ, i, start, end = entry
            print(f"{typ} request {i}: started at {start:.3f}, completed at {end:.3f} (duration: {end-start:.3f}s)")
            
    except (httpx.ConnectError, ConnectionError):
        pytest.skip("Required services not available")
        
    finally:
        # Clean up queue task
        queue_task.cancel()
        try:
            await queue_task
        except asyncio.CancelledError:
            pass