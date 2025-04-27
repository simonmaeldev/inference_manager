from mcp.server.fastmcp import FastMCP
from models.schemas import *
from models.queues import InferenceQueues
from fastapi import HTTPException
import logging
from time import perf_counter

queues = InferenceQueues()

def setup_mcp_server():
    mcp = FastMCP("inference-manager")
    
    async def process_queue():
        """Process all requests in priority order"""
        for model_type in queues.priority_order:
            for model in queues.model_sets[model_type]:
                while queues.queues[model]:
                    request = queues.queues[model].popleft()
                    start_time = perf_counter()
                    queue_wait_time = start_time - request.queue_time
                    try:
                        await request.process()
                        queues.record_metrics(request, start_time, queue_wait_time, True)
                    except Exception as e:
                        logging.error(f"Error processing {request.__class__.__name__} request: {str(e)}")
                        queues.record_error(request, e)
                        queues.record_metrics(request, start_time, queue_wait_time, False)
                        request.future.set_exception(e)
    
    @mcp.tool()
    async def txt2txt(model: str, messages: List[dict], temperature: float = 0.7) -> str:
        """
        Process text-to-text generation request.
        
        Args:
            model: The model ID to use for generation (e.g. "Qwen2.5-32B-Instruct")
            messages: List of message dicts with "role" and "content" keys
            temperature: Sampling temperature (0.0-2.0, default: 0.7)
            
        Returns:
            Generated text response
            
        Raises:
            HTTPException 400: Invalid model or parameters
            HTTPException 500: Internal server error
            
        Example Request:
            {
                "model": "Qwen2.5-32B-Instruct",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "temperature": 0.7
            }
            
        Example Response:
            "Hello! How can I help you today?"
        """
        if not validate_model("txt2txt", model):
            raise HTTPException(status_code=400, detail=f"Model {model} not supported for text-to-text")
            
        req = Txt2TxtRequest(model=model, messages=messages, temperature=temperature)
        req.queue_time = perf_counter()
        queues.queues[req.model].append(req)
        return await req.future
    
    @mcp.tool()
    async def txt2img(model: str, prompt: str, height: int = 512, width: int = 512) -> Any:
        """
        Generate image from text prompt.
        
        Args:
            model: The model ID to use (e.g. "Flux-Dev")
            prompt: Text description of desired image
            height: Image height in pixels (default: 512)
            width: Image width in pixels (default: 512)
            
        Returns:
            Base64 encoded image or image URL
            
        Raises:
            HTTPException 400: Invalid model or parameters
            HTTPException 500: Internal server error
            
        Example Request:
            {
                "model": "Flux-Dev",
                "prompt": "A sunset over mountains",
                "height": 512,
                "width": 512
            }
            
        Example Response:
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAYAAAD0eNT6AA..."
        """
        if not validate_model("txt2img", model):
            raise HTTPException(status_code=400, detail=f"Model {model} not supported for text-to-image")
            
        req = Txt2ImgRequest(model=model, prompt=prompt, height=height, width=width)
        req.queue_time = perf_counter()
        queues.queues[req.model].append(req)
        return await req.future
    
    @mcp.tool()
    async def img2img(model: str, file: str, prompt: str, height: int = 512, width: int = 512) -> Any:
        """
        Generate modified image from input image and prompt.
        
        Args:
            model: The model ID to use (e.g. "Flux-Dev")
            file: Base64 encoded input image
            prompt: Text description of desired modifications
            height: Output image height in pixels (default: 512)
            width: Output image width in pixels (default: 512)
            
        Returns:
            Base64 encoded modified image or image URL
            
        Raises:
            HTTPException 400: Invalid model, image or parameters
            HTTPException 500: Internal server error
            
        Example Request:
            {
                "model": "Flux-Dev",
                "file": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
                "prompt": "Make it look like winter",
                "height": 512,
                "width": 512
            }
            
        Example Response:
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAgAAAAIACAYAAAD0eNT6AA..."
        """
        if not validate_model("img2img", model):
            raise HTTPException(status_code=400, detail=f"Model {model} not supported for image-to-image")
            
        req = Img2ImgRequest(model=model, file=file, prompt=prompt, height=height, width=width)
        req.queue_time = perf_counter()
        queues.queues[req.model].append(req)
        return await req.future
    
    @mcp.tool()
    async def img2txt(model: str, messages: List[dict], temperature: float = 0.9) -> str:
        """
        Generate text description from image.
        
        Args:
            model: The model ID to use (e.g. "Qwen2.5-VL-32B-Instruct")
            messages: List of message dicts with "role" and "content" keys
                     Must include one message with {"role": "user", "content": "image:<base64>"}
            temperature: Sampling temperature (0.0-2.0, default: 0.9)
            
        Returns:
            Generated text description
            
        Raises:
            HTTPException 400: Invalid model, image or parameters
            HTTPException 500: Internal server error
            
        Example Request:
            {
                "model": "Qwen2.5-VL-32B-Instruct",
                "messages": [
                    {"role": "user", "content": "image:data:image/png;base64,iVBORw0KGgo..."},
                    {"role": "user", "content": "Describe this image"}
                ],
                "temperature": 0.9
            }
            
        Example Response:
            "This is a photo of a sunset over mountains with vibrant colors."
        """
        if not validate_model("img2txt", model):
            raise HTTPException(status_code=400, detail=f"Model {model} not supported for image-to-text")
            
        req = Img2TxtRequest(model=model, messages=messages, temperature=temperature)
        req.queue_time = perf_counter()
        queues.queues[req.model].append(req)
        return await req.future

    @mcp.tool()
    async def get_error_metrics() -> Dict[str, Any]:
        """
        Get error metrics and samples.
        
        Returns:
            Dictionary containing:
            - counts: Error counts by type
            - recent_samples: List of recent error samples
            - timestamp: Last update time
            
        Example Response:
            {
                "counts": {
                    "TimeoutError": 5,
                    "ModelNotFound": 2
                },
                "recent_samples": [
                    {
                        "error": "TimeoutError",
                        "timestamp": "2025-04-27T12:00:00Z",
                        "model": "Qwen2.5-32B-Instruct"
                    }
                ],
                "timestamp": "2025-04-27T12:38:56Z"
            }
        """
        return queues.get_error_metrics()
    
    @mcp.tool()
    async def get_request_metrics() -> Dict[str, Any]:
        """
        Get request processing metrics.
        
        Returns:
            Dictionary containing:
            - throughput: Requests per minute
            - avg_queue_time: Average queue wait time in seconds
            - avg_process_time: Average processing time in seconds
            - success_rate: Percentage of successful requests
            - timestamp: Last update time
            
        Example Response:
            {
                "throughput": 42.5,
                "avg_queue_time": 0.25,
                "avg_process_time": 1.8,
                "success_rate": 0.98,
                "timestamp": "2025-04-27T12:38:56Z"
            }
        """
        return queues.get_request_metrics()
    
    return mcp