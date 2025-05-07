from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import List, Optional, Callable, Any, Dict
import asyncio
from pathlib import Path
from abc import abstractmethod

from src.utils.localai_client import post_chat_completion
from src.utils.stability_matrix_client import txt2img, img2img

class InferenceRequest(BaseModel):
    """Base request model with future support and processing"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: str
    future: asyncio.Future = Field(default_factory=asyncio.Future)
    fulfill: Callable[[Any], None] = None
    
    @model_validator(mode='after')
    def setup_fulfill(self) -> 'InferenceRequest':
        """Set up the fulfill function after model initialization"""
        self.fulfill = lambda result: self.future.set_result(result)
        return self
    
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
    steps: int = 50
    
    async def process(self):
        """Process text-to-image request"""
        result = await txt2img(
            prompt=self.prompt,
            height=self.height,
            width=self.width,
            steps=self.steps
        )
        self.fulfill(result)
        return result

class Img2ImgRequest(InferenceRequest):
    """Image-to-image generation request"""
    file: str  # base64 encoded
    prompt: str
    height: int = 512
    width: int = 512
    steps: int = 50
    
    async def process(self):
        """Process image-to-image request"""
        result = await img2img(
            init_images=[self.file],
            prompt=self.prompt,
            height=self.height,
            width=self.width,
            steps=self.steps
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
