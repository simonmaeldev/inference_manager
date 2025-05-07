from collections import deque
from typing import Deque, Dict
from src.models.schemas import InferenceRequest
import asyncio

class InferenceQueues:
    """Manages queues for different inference request types and models.
    
    Attributes:
        model_sets: Dictionary mapping request types to supported models
        queues: Dictionary mapping models to their request queues
        priority_order: List defining processing priority of request types
        request_event: Event that signals when new requests are available
    """
    def __init__(self):
        # Model sets for each request type
        self.model_sets = {
            "txt2txt": {"Qwen2.5-32B-Instruct"},
            "txt2img": {"Flux-Dev"},
            "img2img": {"Flux-Dev"},
            "img2txt": {"Qwen2.5-VL-32B-Instruct"}
        }
        
        # Initialize queues for each model
        self.queues: Dict[str, Deque[InferenceRequest]] = {}
        for models in self.model_sets.values():
            for model in models:
                self.queues[model] = deque()
        
        # Define processing priority order
        self.priority_order = ["txt2txt", "txt2img", "img2img", "img2txt"]
        
        # Event to signal when requests are available
        self.request_event = asyncio.Event()
    
    async def add_request(self, model: str, request: InferenceRequest):
        """Add a request to the queue and signal the processor.
        
        Args:
            model: The model to use for the request
            request: The inference request to add
        """
        self.queues[model].append(request)
        self.request_event.set()  # Signal that a request is available
    
    async def process_all_queues_once(self):
        for model_type in self.priority_order:
            for model in self.model_sets[model_type]:
                while self.queues[model]:
                    request = self.queues[model].popleft()
                    try:
                        await request.process()
                    except Exception as e:
                        request.future.set_exception(e)
    
    async def process_queues_forever(self):
        """Process queues when requests are available.
        
        This method implements a producer-consumer pattern:
        - It waits for the request_event to be set (by add_request)
        - When triggered, it processes all available requests in batch
        - It clears the event if all queues are empty
        - It yields control back to the event loop periodically
        """
        while True:
            # Wait for a request to be added
            await self.request_event.wait()
            
            # Process all available requests in batch (maintaining the priority order)
            await self.process_all_queues_once()
            
            # Check if all queues are empty
            all_empty = True
            for queue in self.queues.values():
                if queue:
                    all_empty = False
                    break
            
            # Reset the event if all queues are empty
            if all_empty:
                self.request_event.clear()
            
            # Yield control back to the event loop
            await asyncio.sleep(0.01)
