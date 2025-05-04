from pydantic import BaseModel
from typing import List, Optional, Callable, Any, Dict
import asyncio
from pathlib import Path
from abc import abstractmethod

from src.utils.localai_client import post_chat_completion, post_image_generation

class InferenceRequest(BaseModel):
    """Base request model with future support and processing"""
    model: str
    future: Optional[asyncio.Future] = None
    fulfill: Optional[Callable[[Any], None]] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        if not Path(f"models/{self.model}").exists():
            raise ValueError(f"Model {self.model} not found in models directory")
        self.future = asyncio.Future()
        self.fulfill = lambda result: self.future.set_result(result)
    
    @abstractmethod
    async def process(self):
        """Process this request and call fulfill() with result"""
        pass

class Txt2TxtRequest(InferenceRequest):
    """Text-to-text generation request"""
    messages: List[Dict[str, str]]
    temperature: float = 0.7
    
    async def process(self):
        """Process text-to-text request"""
        result = await post_chat_completion(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )
        self.fulfill(result)
        return result

class Txt2ImgRequest(InferenceRequest):
    """Text-to-image generation request"""
    prompt: str
    height: int = 512
    width: int = 512
    
    async def process(self):
        """Process text-to-image request"""
        result = await post_image_generation(
            model=self.model,
            prompt=self.prompt,
            height=self.height,
            width=self.width
        )
        self.fulfill(result)
        return result

class Img2ImgRequest(InferenceRequest):
    """Image-to-image generation request"""
    file: str  # base64 encoded
    prompt: str
    height: int = 512
    width: int = 512
    
    async def process(self):
        """Process image-to-image request"""
        result = await post_image_generation(
            model=self.model,
            prompt=self.prompt,
            height=self.height,
            width=self.width
        )
        self.fulfill(result)
        return result

class Img2TxtRequest(InferenceRequest):
    """Image-to-text generation request"""
    messages: List[Dict[str, str]]
    temperature: float = 0.9
    
    async def process(self):
        """Process image-to-text request"""
        result = await post_chat_completion(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )
        self.fulfill(result)
        return result
