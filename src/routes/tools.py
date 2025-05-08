import httpx
from typing import Dict, Any, List

from src.models.queues import InferenceQueues
from src.models.schemas import Txt2ImgRequest, Txt2TxtRequest

class Tools:
    @staticmethod
    def add(a: int, b: int) -> int:
        """Add two numbers together"""
        return a + b
    
    @staticmethod
    async def text_to_image(prompt: str, queues: InferenceQueues, model: str, width: int = 640, height: int = 360) -> List[str]:
        """Generate an image using the queue system"""
        
        req = Txt2ImgRequest(
            model=model,
            prompt=prompt,
            width=width,
            height=height
        )
        await queues.add_request(model, req)
        result = await req.future
        return result if result else []
    
    @staticmethod
    async def generate_image(prompt: str, queues: InferenceQueues, model: str, step: int = 50, size: str = "640x360") -> Dict[str, Any]:
        """Generate an image with specific parameters using the queue system"""
        # Parse size string to get width and height
        width, height = map(int, size.split('x'))
        
        req = Txt2ImgRequest(
            model=model,
            prompt=prompt,
            width=width,
            height=height,
            steps=step
        )
        await queues.add_request(model, req)
        result = await req.future
        return result if result else []
    
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