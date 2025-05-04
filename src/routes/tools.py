import httpx
from typing import Dict, Any

class Tools:
    @staticmethod
    def add(a: int, b: int) -> int:
        """Add two numbers together"""
        return a + b
    
    @staticmethod
    async def text_to_image(prompt: str, model: str, width: int = 640, height: int = 360) -> List[str]:
        """Generate an image using the queue system"""
        from src.models.schemas import Txt2ImgRequest
        from main import queues
        
        req = Txt2ImgRequest(
            model=model,
            prompt=prompt,
            width=width,
            height=height
        )
        queues.queues[model].append(req)
        result = await req.future
        return result if result else []
    
    @staticmethod
    async def generate_text(messages: list, model: str = "gpt-4", temperature: float = 0.7) -> Dict[str, Any]:
        """Generate text using the local API service"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localai:8080/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": temperature
                }
            )
            return response.json()