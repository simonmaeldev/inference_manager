from collections import deque
from typing import Deque, Dict
from src.models.schemas import InferenceRequest

class InferenceQueues:
    """Manages queues for different inference request types and models.
    
    Attributes:
        model_sets: Dictionary mapping request types to supported models
        queues: Dictionary mapping models to their request queues
        priority_order: List defining processing priority of request types
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
    
