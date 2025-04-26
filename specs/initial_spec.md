# Inference Manager MCP

> Create a Multi-Context Process (MCP) that efficiently manages various AI inference operations, optimizing resource usage by batching similar requests and handling model loading/unloading

## High Level Objective

* Develop a centralized inference service that efficiently manages AI model resources, provides a consistent API for all types of inference operations (text-to-text, text-to-image, image-to-text, image processing), and optimizes hardware utilization while ensuring reliable and stateful operation.

## Mid Level Objective

* Provide a unified API for all inference types
* Manage loading and unloading of models to optimize VRAM usage
* Batch similar inference requests to improve throughput
* Prioritize requests according to configured importance
* Handle state persistence for crash recovery (optional)
* Support different model backends (local models and potentially remote APIs)
* Provide logging and monitoring for inference operations

## Implementation Notes

* Use Python for all implementations
* Package as a containerized MCP (Multi-Context Process)
* Support GPU acceleration in WSL2 environment
* Use REST API for communication with other services
* Focus on robustness and error handling
* Implement request queueing with priority handling
* Design for future extension to cloud-based inference

## Context

### Beginning Context

* `config/` - Directory for configuration files
* `config/models.json` - Model configuration
* `config/inference_manager.json` - General configuration
* `src/` - Directory for source code

### Ending Context

* `config/` - Configuration files directory
* `config/models.json` - Model configuration
* `config/inference_manager.json` - General configuration
* `src/` - Source code directory
* `src/main.py` - Entry point for the MCP
* `src/api.py` - REST API implementation
* `src/inference_manager.py` - Core inference manager
* `src/model_manager.py` - Model loading/unloading
* `src/request_queue.py` - Request queue management
* `src/batch_processor.py` - Request batching
* `src/text_inference.py` - Text model inference
* `src/image_inference.py` - Image model inference
* `src/state_manager.py` - State persistence (optional)
* `src/datatypes.py` - Data structure definitions
* `src/utils.py` - Utility functions
* `Dockerfile` - Container definition

## Low Level Tasks

> Ordered from start to finish

1. Create core data structures in `datatypes.py`

```python
from pydantic import BaseModel
from typing import Dict, List, Optional, Literal, Union, Any
from enum import Enum
from datetime import datetime
from pathlib import Path

class InferenceType(str, Enum):
    """Types of inference operations supported."""
    TEXT_TO_TEXT = "text2text"
    TEXT_TO_IMAGE = "text2image"
    IMAGE_TO_TEXT = "image2text"
    IMAGE_PROCESSING = "image_processing"

class RequestStatus(str, Enum):
    """Status of inference requests."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"

class InferenceRequest(BaseModel):
    """Base class for inference requests."""
    request_id: str
    type: InferenceType
    priority: int
    created_at: datetime = datetime.now()
    status: RequestStatus = RequestStatus.PENDING
    callback_url: Optional[str] = None
    
class TextToTextRequest(InferenceRequest):
    """Request for text-to-text inference."""
    prompt: str
    parameters: Dict[str, Any] = {}
    
class TextToImageRequest(InferenceRequest):
    """Request for text-to-image inference."""
    prompt: str
    negative_prompt: Optional[str] = None
    parameters: Dict[str, Any] = {}
    
class ImageToTextRequest(InferenceRequest):
    """Request for image-to-text inference."""
    image_path: Path
    parameters: Dict[str, Any] = {}
    
class ImageProcessingRequest(InferenceRequest):
    """Request for image processing operations."""
    image_path: Path
    operation: str  # E.g., "upscale", "resize", etc.
    parameters: Dict[str, Any] = {}

class InferenceResponse(BaseModel):
    """Base class for inference responses."""
    request_id: str
    status: RequestStatus
    created_at: datetime = datetime.now()
    error_message: Optional[str] = None
    
class TextToTextResponse(InferenceResponse):
    """Response for text-to-text inference."""
    text: str
    
class TextToImageResponse(InferenceResponse):
    """Response for text-to-image inference."""
    image_paths: List[Path]
    
class ImageToTextResponse(InferenceResponse):
    """Response for image-to-text inference."""
    text: str
    
class ImageProcessingResponse(InferenceResponse):
    """Response for image processing operations."""
    image_path: Path

class ModelConfig(BaseModel):
    """Configuration for a model."""
    model_id: str
    model_type: InferenceType
    model_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    vram_required_mb: int
    loading_time_sec: int
    priority: int
    max_batch_size: int = 1
    parameters: Dict[str, Any] = {}

class ManagerState(BaseModel):
    """State of the inference manager for persistence."""
    active_models: Dict[str, str]  # model_id -> status
    request_queue: Dict[str, Union[TextToTextRequest, TextToImageRequest, 
                                   ImageToTextRequest, ImageProcessingRequest]]
    last_updated: datetime = datetime.now()
```

2. Implement model manager in `model_manager.py`

```python
from src.datatypes import *
import torch
import os
import json
import time
import logging
from typing import Dict, List, Optional, Union

class ModelManager:
    """Manages loading and unloading of AI models."""
    
    def __init__(self, config_path: str):
        """
        Initialize model manager with configuration.
        
        Args:
            config_path: Path to model configuration file
        """
        self.config_path = config_path
        self.models = {}  # model_id -> ModelConfig
        self.loaded_models = {}  # model_id -> model object
        self.load_configs()
        
    def load_configs(self) -> None:
        """Load model configurations from config file."""
        pass
        
    def get_model_config(self, model_id: str) -> Optional[ModelConfig]:
        """
        Get configuration for a specific model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            ModelConfig if found, None otherwise
        """
        pass
        
    def get_available_vram(self) -> int:
        """
        Get available VRAM in megabytes.
        
        Returns:
            Available VRAM in MB
        """
        pass
        
    def load_model(self, model_id: str) -> bool:
        """
        Load model into memory.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Success status
        """
        pass
        
    def unload_model(self, model_id: str) -> bool:
        """
        Unload model from memory.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Success status
        """
        pass
        
    def get_loaded_model(self, model_id: str) -> Optional[Any]:
        """
        Get loaded model object.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model object if loaded, None otherwise
        """
        pass
        
    def get_best_model_for_type(self, inference_type: InferenceType) -> Optional[str]:
        """
        Get best available model for inference type.
        
        Args:
            inference_type: Type of inference
            
        Returns:
            Model ID if available, None otherwise
        """
        pass
        
    def optimize_loaded_models(self, upcoming_requests: List[InferenceRequest]) -> Dict[str, str]:
        """
        Optimize loaded models based on upcoming requests.
        
        Args:
            upcoming_requests: List of upcoming requests
            
        Returns:
            Dict of actions taken (model_id -> action)
        """
        pass
```

3. Create request queue in `request_queue.py`

```python
from src.datatypes import *
import heapq
import threading
import time
from typing import Dict, List, Optional, Union, Tuple, Callable

class RequestPriorityQueue:
    """Priority queue for inference requests."""
    
    def __init__(self):
        """Initialize request priority queue."""
        self.queue = []  # heap queue
        self.requests = {}  # request_id -> request
        self.lock = threading.Lock()
        
    def add_request(self, request: Union[TextToTextRequest, TextToImageRequest, 
                                          ImageToTextRequest, ImageProcessingRequest]) -> None:
        """
        Add request to queue with priority.
        
        Args:
            request: Inference request
        """
        pass
        
    def get_next_request(self) -> Optional[Union[TextToTextRequest, TextToImageRequest, 
                                                 ImageToTextRequest, ImageProcessingRequest]]:
        """
        Get next request from queue.
        
        Returns:
            Next request if available, None otherwise
        """
        pass
        
    def get_request_by_id(self, request_id: str) -> Optional[Union[TextToTextRequest, TextToImageRequest, 
                                                                  ImageToTextRequest, ImageProcessingRequest]]:
        """
        Get request by ID.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Request if found, None otherwise
        """
        pass
        
    def update_request_status(self, request_id: str, status: RequestStatus) -> bool:
        """
        Update status of request.
        
        Args:
            request_id: Request identifier
            status: New status
            
        Returns:
            Success status
        """
        pass
        
    def get_requests_by_type(self, inference_type: InferenceType, 
                             max_count: int = 10) -> List[Union[TextToTextRequest, TextToImageRequest, 
                                                               ImageToTextRequest, ImageProcessingRequest]]:
        """
        Get requests of specific type for batching.
        
        Args:
            inference_type: Type of inference
            max_count: Maximum number of requests to return
            
        Returns:
            List of requests
        """
        pass
        
    def remove_requests(self, request_ids: List[str]) -> int:
        """
        Remove requests from queue.
        
        Args:
            request_ids: List of request identifiers
            
        Returns:
            Number of requests removed
        """
        pass
        
    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get statistics about queue.
        
        Returns:
            Dictionary of queue statistics
        """
        pass
```

4. Implement batch processor in `batch_processor.py`

```python
from src.datatypes import *
from typing import Dict, List, Optional, Union, Any
import time
import logging

class BatchProcessor:
    """Processes batches of similar inference requests."""
    
    def __init__(self, model_manager):
        """
        Initialize batch processor.
        
        Args:
            model_manager: Model manager instance
        """
        self.model_manager = model_manager
        
    def can_batch(self, requests: List[Union[TextToTextRequest, TextToImageRequest, 
                                            ImageToTextRequest, ImageProcessingRequest]]) -> bool:
        """
        Check if requests can be batched.
        
        Args:
            requests: List of requests
            
        Returns:
            Whether requests can be batched
        """
        pass
        
    def prepare_batch(self, requests: List[Union[TextToTextRequest, TextToImageRequest, 
                                                ImageToTextRequest, ImageProcessingRequest]]) -> Dict:
        """
        Prepare batch for processing.
        
        Args:
            requests: List of requests
            
        Returns:
            Batch data
        """
        pass
        
    def process_text_to_text_batch(self, 
                                  requests: List[TextToTextRequest], 
                                  model_id: str) -> Dict[str, TextToTextResponse]:
        """
        Process batch of text-to-text requests.
        
        Args:
            requests: List of requests
            model_id: Model identifier
            
        Returns:
            Dictionary of responses by request ID
        """
        pass
        
    def process_text_to_image_batch(self, 
                                   requests: List[TextToImageRequest], 
                                   model_id: str) -> Dict[str, TextToImageResponse]:
        """
        Process batch of text-to-image requests.
        
        Args:
            requests: List of requests
            model_id: Model identifier
            
        Returns:
            Dictionary of responses by request ID
        """
        pass
        
    def process_image_to_text_batch(self, 
                                   requests: List[ImageToTextRequest], 
                                   model_id: str) -> Dict[str, ImageToTextResponse]:
        """
        Process batch of image-to-text requests.
        
        Args:
            requests: List of requests
            model_id: Model identifier
            
        Returns:
            Dictionary of responses by request ID
        """
        pass
        
    def process_image_processing_batch(self, 
                                      requests: List[ImageProcessingRequest], 
                                      model_id: str) -> Dict[str, ImageProcessingResponse]:
        """
        Process batch of image processing requests.
        
        Args:
            requests: List of requests
            model_id: Model identifier
            
        Returns:
            Dictionary of responses by request ID
        """
        pass
```

5. Implement text inference in `text_inference.py`

```python
from src.datatypes import *
import torch
from typing import Dict, List, Optional, Any
import logging

class TextInferenceHandler:
    """Handles text-to-text inference operations."""
    
    def __init__(self, model_manager):
        """
        Initialize text inference handler.

        Args:
            model_manager: Model manager instance
        """
        self.model_manager = model_manager
        
    def perform_inference(self, request: TextToTextRequest) -> TextToTextResponse:
        """
        Perform text-to-text inference for a single request.
        
        Args:
            request: Text-to-text request
            
        Returns:
            Text-to-text response
        """
        pass
        
    def batch_inference(self, requests: List[TextToTextRequest], model_id: str) -> Dict[str, TextToTextResponse]:
        """
        Perform batch text-to-text inference.
        
        Args:
            requests: List of requests
            model_id: Model identifier
            
        Returns:
            Dictionary of responses by request ID
        """
        pass
        
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported text-to-text models.
        
        Returns:
            List of model identifiers
        """
        pass
        
    def estimate_inference_time(self, request: TextToTextRequest, model_id: str) -> float:
        """
        Estimate inference time for request.
        
        Args:
            request: Text-to-text request
            model_id: Model identifier
            
        Returns:
            Estimated time in seconds
        """
        pass
```

6. Implement image inference in `image_inference.py`

```python
from src.datatypes import *
import torch
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import os

class ImageInferenceHandler:
    """Handles image-related inference operations."""
    
    def __init__(self, model_manager, output_dir: str = "./output"):
        """
        Initialize image inference handler.
        
        Args:
            model_manager: Model manager instance
            output_dir: Directory for output images
        """
        self.model_manager = model_manager
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        
    def perform_text_to_image(self, request: TextToImageRequest) -> TextToImageResponse:
        """
        Perform text-to-image inference.
        
        Args:
            request: Text-to-image request
            
        Returns:
            Text-to-image response
        """
        pass
        
    def perform_image_to_text(self, request: ImageToTextRequest) -> ImageToTextResponse:
        """
        Perform image-to-text inference.
        
        Args:
            request: Image-to-text request
            
        Returns:
            Image-to-text response
        """
        pass
        
    def perform_image_processing(self, request: ImageProcessingRequest) -> ImageProcessingResponse:
        """
        Perform image processing.
        
        Args:
            request: Image processing request
            
        Returns:
            Image processing response
        """
        pass
        
    def batch_text_to_image(self, requests: List[TextToImageRequest], model_id: str) -> Dict[str, TextToImageResponse]:
        """
        Perform batch text-to-image inference.
        
        Args:
            requests: List of requests
            model_id: Model identifier
            
        Returns:
            Dictionary of responses by request ID
        """
        pass
        
    def batch_image_to_text(self, requests: List[ImageToTextRequest], model_id: str) -> Dict[str, ImageToTextResponse]:
        """
        Perform batch image-to-text inference.
        
        Args:
            requests: List of requests
            model_id: Model identifier
            
        Returns:
            Dictionary of responses by request ID
        """
        pass
        
    def batch_image_processing(self, requests: List[ImageProcessingRequest], model_id: str) -> Dict[str, ImageProcessingResponse]:
        """
        Perform batch image processing.
        
        Args:
            requests: List of requests
            model_id: Model identifier
            
        Returns:
            Dictionary of responses by request ID
        """
        pass
```

7. Create optional state manager in `state_manager.py`

```python
from src.datatypes import *
import json
import os
from datetime import datetime
from typing import Optional
from pathlib import Path

class StateManager:
    """Manages persistence of inference manager state."""
    
    def __init__(self, state_dir: str = "./state", save_interval: int = 600):
        """
        Initialize state manager.
        
        Args:
            state_dir: Directory for state files
            save_interval: Save interval in seconds
        """
        self.state_dir = Path(state_dir)
        self.save_interval = save_interval
        self.last_save = datetime.now()
        os.makedirs(self.state_dir, exist_ok=True)
        
    def load_state(self) -> Optional[ManagerState]:
        """
        Load state from disk.
        
        Returns:
            Manager state if available, None otherwise
        """
        pass
        
    def save_state(self, state: ManagerState) -> bool:
        """
        Save state to disk.
        
        Args:
            state: Manager state to save
            
        Returns:
            Success status
        """
        pass
        
    def periodic_save(self, state: ManagerState) -> bool:
        """
        Save state if save interval has passed.
        
        Args:
            state: Manager state to save
            
        Returns:
            Whether save was performed
        """
        pass
        
    def clear_state(self) -> bool:
        """
        Clear state files.
        
        Returns:
            Success status
        """
        pass
```

8. Implement core inference manager in `inference_manager.py`

```python
from src.datatypes import *
from src.model_manager import ModelManager
from src.request_queue import RequestPriorityQueue
from src.batch_processor import BatchProcessor
from src.text_inference import TextInferenceHandler
from src.image_inference import ImageInferenceHandler
from src.state_manager import StateManager
import threading
import time
import uuid
import logging
from typing import Dict, List, Optional, Union, Any

class InferenceManager:
    """Core inference manager that coordinates all operations."""
    
    def __init__(self, config_path: str, state_enabled: bool = False):
        """
        Initialize inference manager.
        
        Args:
            config_path: Path to configuration file
            state_enabled: Whether state persistence is enabled
        """
        self.config_path = config_path
        self.load_config()
        self.model_manager = ModelManager(self.model_config_path)
        self.request_queue = RequestPriorityQueue()
        self.batch_processor = BatchProcessor(self.model_manager)
        self.text_inference = TextInferenceHandler(self.model_manager)
        self.image_inference = ImageInferenceHandler(self.model_manager, self.output_dir)
        self.state_enabled = state_enabled
        if state_enabled:
            self.state_manager = StateManager(self.state_dir, self.save_interval)
        self.running = False
        self.worker_thread = None
        
    def load_config(self) -> None:
        """Load configuration from file."""
        pass
        
    def start(self) -> None:
        """Start inference manager."""
        pass
        
    def stop(self) -> None:
        """Stop inference manager."""
        pass
        
    def worker_loop(self) -> None:
        """Main worker loop for processing requests."""
        pass
        
    def submit_request(self, request: Union[TextToTextRequest, TextToImageRequest, 
                                            ImageToTextRequest, ImageProcessingRequest]) -> str:
        """
        Submit inference request.
        
        Args:
            request: Inference request
            
        Returns:
            Request ID
        """
        pass
        
    def get_request_status(self, request_id: str) -> Optional[RequestStatus]:
        """
        Get status of request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Request status if found, None otherwise
        """
        pass
        
    def get_request_result(self, request_id: str) -> Optional[Union[TextToTextResponse, TextToImageResponse, 
                                                                    ImageToTextResponse, ImageProcessingResponse]]:
        """
        Get result of request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Request result if available, None otherwise
        """
        pass
        
    def process_batch(self, inference_type: InferenceType) -> int:
        """
        Process batch of requests of specific type.
        
        Args:
            inference_type: Type of inference
            
        Returns:
            Number of requests processed
        """
        pass
        
    def create_text_to_text_request(self, prompt: str, parameters: Dict[str, Any] = None, 
                                   priority: int = None, callback_url: str = None) -> str:
        """
        Create and submit text-to-text request.
        
        Args:
            prompt: Input prompt
            parameters: Inference parameters
            priority: Request priority
            callback_url: Callback URL
            
        Returns:
            Request ID
        """
        pass
        
    def create_text_to_image_request(self, prompt: str, negative_prompt: str = None, 
                                    parameters: Dict[str, Any] = None, priority: int = None, 
                                    callback_url: str = None) -> str:
        """
        Create and submit text-to-image request.
        
        Args:
            prompt: Input prompt
            negative_prompt: Negative prompt
            parameters: Inference parameters
            priority: Request priority
            callback_url: Callback URL
            
        Returns:
            Request ID
        """
        pass
        
    def create_image_to_text_request(self, image_path: Union[str, Path], 
                                    parameters: Dict[str, Any] = None, priority: int = None, 
                                    callback_url: str = None) -> str:
        """
        Create and submit image-to-text request.
        
        Args:
            image_path: Path to input image
            parameters: Inference parameters
            priority: Request priority
            callback_url: Callback URL
            
        Returns:
            Request ID
        """
        pass
        
    def create_image_processing_request(self, image_path: Union[str, Path], operation: str, 
                                       parameters: Dict[str, Any] = None, priority: int = None, 
                                       callback_url: str = None) -> str:
        """
        Create and submit image processing request.
        
        Args:
            image_path: Path to input image
            operation: Processing operation
            parameters: Processing parameters
            priority: Request priority
            callback_url: Callback URL
            
        Returns:
            Request ID
        """
        pass
```

9. Implement API in `api.py`

```python
from src.datatypes import *
from src.inference_manager import InferenceManager
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import uvicorn
import logging
import os
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import json

app = FastAPI(title="Inference Manager API")

class InferenceAPI:
    """REST API for Inference Manager."""
    
    def __init__(self, inference_manager: InferenceManager):
        """
        Initialize API with inference manager.
        
        Args:
            inference_manager: Inference manager instance
        """
        self.inference_manager = inference_manager
        self.setup_routes()
        
    def setup_routes(self) -> None:
        """Set up API routes."""
        pass
        
    async def text_to_text(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle text-to-text request.
        
        Args:
            request: Request data
            
        Returns:
            Response data
        """
        pass
        
    async def text_to_image(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle text-to-image request.
        
        Args:
            request: Request data
            
        Returns:
            Response data
        """
        pass
        
    async def image_to_text(self, file: UploadFile, parameters: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle image-to-text request.
        
        Args:
            file: Uploaded image file
            parameters: JSON-encoded parameters
            
        Returns:
            Response data
        """
        pass
        
    async def image_processing(self, file: UploadFile, operation: str, parameters: Optional[str] = None) -> Dict[str, Any]:
        """
        Handle image processing request.
        
        Args:
            file: Uploaded image file
            operation: Processing operation
            parameters: JSON-encoded parameters
            
        Returns:
            Response data
        """
        pass
        
    async def get_request_status(self, request_id: str) -> Dict[str, Any]:
        """
        Get status of request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Status data
        """
        pass
        
    async def get_request_result(self, request_id: str) -> Union[Dict[str, Any], FileResponse]:
        """
        Get result of request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Result data or file
        """
        pass
        
    async def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get statistics about manager.
        
        Returns:
            Statistics data
        """
        pass
        
    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        """
        Run API server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
        """
        pass
```

10. Create main MCP entry point in `main.py`

```python
from src.inference_manager import InferenceManager
from src.api import InferenceAPI
import argparse
import logging
import os
import signal
import sys
from typing import Dict, List, Optional

def configure_logging(log_level: str) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level
    """
    pass

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    pass

def signal_handler(sig, frame):
    """
    Handle termination signals.
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    pass

def main():
    """Main entry point for the Inference Manager MCP."""
    pass

if __name__ == "__main__":
    main()
```
