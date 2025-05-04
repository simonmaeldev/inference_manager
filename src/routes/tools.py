import httpx
from typing import Dict, Any

class Tools:
    @staticmethod
    def add(a: int, b: int) -> int:
        """Add two numbers together"""
        return a + b
    
    @staticmethod
    async def generate_image(prompt: str, model: str = "flux.1-dev", step: int = 50, size: str = "1024x1024") -> Dict[str, Any]:
        """Generate an image using the local API service"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localai:8080/v1/images/generations",
                json={
                    "prompt": prompt,
                    "model": model,
                    "step": step,
                    "size": size
                }
            )
            return response.json()
    
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