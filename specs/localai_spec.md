# LocalAI Inference Manager - Developer Guide

## Goal
Build a queue-based inference server that processes requests by type and model, with clear separation between request handling and inference execution.

## Key Requirements
1. FastAPI server with MCP integration
2. Type-based queues with model sets
3. Pydantic models for each request type with futures
4. Abstract inference handler with concrete implementations
5. Docker integration with LocalAI

## File Structure
```
src/
├── main.py                 # FastAPI app entry point
├── models/
│   ├── __init__.py         # Model validation
│   ├── queues.py           # Queue management
│   └── schemas.py          # Pydantic models with processing
├── routes/
│   ├── api.py              # REST API endpoints
│   └── mcp.py              # MCP server implementation
└── utils/
    └── localai_client.py   # LocalAI API client
```

## Implementation Steps

### 1. Pydantic Models with Futures (src/models/schemas.py)
```python
from pydantic import BaseModel
from typing import List, Optional, Callable, Any
import asyncio

class InferenceRequest(BaseModel):
    """Base request model with future support and processing"""
    model: str
    future: asyncio.Future = None
    fulfill: Callable[[Any], None] = None

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
    messages: List[dict]
    temperature: float = 0.7
    
    async def process(self):
        result = await post_chat_completion(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )
        self.fulfill(result)
        return result

class Txt2ImgRequest(InferenceRequest):
    prompt: str
    height: int
    width: int
    
    async def process(self):
        result = await post_image_generation(
            model=self.model,
            prompt=self.prompt,
            height=self.height,
            width=self.width
        )
        self.fulfill(result)
        return result

class Img2ImgRequest(InferenceRequest):
    file: str  # base64 encoded
    prompt: str
    height: int
    width: int
    
    async def process(self):
        result = await post_image_generation(
            model=self.model,
            prompt=self.prompt,
            height=self.height,
            width=self.width
        )
        self.fulfill(result)
        return result

class Img2TxtRequest(InferenceRequest):
    messages: List[dict]
    temperature: float
    
    async def process(self):
        result = await post_chat_completion(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )
        self.fulfill(result)
        return result
```

### 2. Queue System (src/models/queues.py)
```python
from collections import deque
from typing import Deque, Dict, Set
from models.schemas import InferenceRequest

class InferenceQueues:
    def __init__(self):
        # Model sets for each type
        self.model_sets = {
            "txt2txt": {"Qwen2.5-32B-Instruct"},
            "txt2img": {"Flux-Dev"},
            "img2img": {"Flux-Dev"}, 
            "img2txt": {"Qwen2.5-VL-32B-Instruct"}
        }
        
        # Queues for each model
        self.queues: Dict[str, Deque[InferenceRequest]] = {}
        for models in self.model_sets.values():
            for model in models:
                self.queues[model] = deque()
        
        self.priority_order = ["txt2txt", "txt2img", "img2img", "img2txt"]
```

### 3. Model Validation (src/models/schemas.py)
```python
def validate_model(model_type: str, model: str) -> bool:
    """Validate model is supported for given type"""
    model_sets = {
        "txt2txt": {"Qwen2.5-32B-Instruct"},
        "txt2img": {"Flux-Dev"},
        "img2img": {"Flux-Dev"},
        "img2txt": {"Qwen2.5-VL-32B-Instruct"}
    }
    return model in model_sets.get(model_type, set())
```

### 4. MCP Server with Future Handling (src/routes/mcp.py)
```python
from mcp.server.fastmcp import FastMCP
from models.schemas import *
from models.queues import InferenceQueues

queues = InferenceQueues()

def setup_mcp_server():
    mcp = FastMCP("inference-manager")
    
    async def process_queue():
        """Process all requests in priority order"""
        for model_type in queues.priority_order:
            for model in queues.model_sets[model_type]:
                while queues.queues[model]:
                    request = queues.queues[model].popleft()
                    try:
                        await request.process()
                    except Exception as e:
                        request.future.set_exception(e)
    
    @mcp.tool()
    async def txt2txt(model: str, messages: List[dict], temperature: float = 0.7) -> txt:
        """Queue txt2txt request and await future"""
        req = Txt2TxtRequest(model=model, messages=messages, temperature=temperature)
        queues.queues[req.model].append(req)
        return await req.future
    
    @mcp.tool()
    async def txt2img(model: str, prompt: str, height: int = 256, width: int = 256) -> Any:
        req = Txt2ImgRequest(model=model, prompt=prompt, height=height, width=width)
        queues.queues[req.model].append(req)
        return await req.future
    
    @mcp.tool()
    async def img2img(model: str, file: str, prompt: str, height: int = 512, width: int = 512) -> Any:
        req = Img2ImgRequest(model=model, file=file, prompt=prompt, height=height, width=width)
        queues.queues[req.model].append(req)
        return await req.future
    
    @mcp.tool()
    async def img2txt(model: str, messages: List[dict], temperature: float = 0.9) -> txt:
        req = Img2TxtRequest(model=model, messages=messages, temperature=temperature)
        queues.queues[req.model].append(req)
        return await req.future
    
    return mcp
```


### 5. LocalAI Client (src/utils/localai_client.py)
```python
import httpx

async def post_chat_completion(model: str, messages: list, **kwargs):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8080/v1/chat/completions",
            json={"model": model, "messages": messages, **kwargs}
        )
        resp.raise_for_status()
        return resp.json()

async def post_image_generation(model: str, prompt: str, **kwargs):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "http://localhost:8080/v1/images/generations",
            json={"model": model, "prompt": prompt, **kwargs}
        )
        resp.raise_for_status()
        return resp.json()
```

## Coding Guidelines
1. Strict separation between:
   - Request validation (tools)
   - Queue management
   - Inference execution
2. Use futures for async result handling
3. All requests must use Pydantic models with fulfillment
4. Async throughout the stack
5. Comprehensive error handling
6. Type hints for all function signatures
7. Clear documentation of futures and fulfillment flow