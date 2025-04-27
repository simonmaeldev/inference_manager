from collections import deque
from typing import Deque, Dict, Set, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel
from models.schemas import InferenceRequest
from time import perf_counter

class ErrorRecord(BaseModel):
    """Record of an error that occurred during processing"""
    timestamp: datetime
    model: str
    request_type: str
    error_type: str
    error_message: str
    sanitized_params: Dict[str, Any]

class RequestMetrics(BaseModel):
    """Metrics for a processed request"""
    model: str
    request_type: str
    start_time: datetime
    end_time: datetime
    processing_time: float
    queue_wait_time: float
    success: bool

class InferenceQueues:
    """Manages queues for different inference request types and models.
    
    Attributes:
        model_sets: Dictionary mapping request types to supported models
        queues: Dictionary mapping models to their request queues
        priority_order: List defining processing priority of request types
        error_counts: Dictionary tracking error counts by type and model
        recent_errors: List of recent error records (max 100)
        request_metrics: List of recent request metrics (max 1000)
        throughput_counts: Dictionary tracking successful requests by type and model
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

        # Error tracking
        self.error_counts: Dict[str, Dict[str, int]] = {}
        self.recent_errors: List[ErrorRecord] = []
        
        # Metrics tracking
        self.request_metrics: List[RequestMetrics] = []
        self.throughput_counts: Dict[str, Dict[str, int]] = {}
    
    def record_error(self, request: InferenceRequest, error: Exception):
        """Record an error that occurred during processing"""
        # Get request type from class name
        request_type = request.__class__.__name__.replace("Request", "").lower()
        
        # Initialize error counts if needed
        if request_type not in self.error_counts:
            self.error_counts[request_type] = {}
        if request.model not in self.error_counts[request_type]:
            self.error_counts[request_type][request.model] = 0
        
        # Increment error count
        self.error_counts[request_type][request.model] += 1
        
        # Create error record
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            model=request.model,
            request_type=request_type,
            error_type=error.__class__.__name__,
            error_message=str(error),
            sanitized_params=self._sanitize_params(request)
        )
        
        # Add to recent errors (keep max 100)
        self.recent_errors.append(error_record)
        if len(self.recent_errors) > 100:
            self.recent_errors.pop(0)
    
    def _sanitize_params(self, request: InferenceRequest) -> Dict[str, Any]:
        """Sanitize request parameters for error logging"""
        params = request.dict()
        # Remove sensitive fields
        params.pop("future", None)
        params.pop("fulfill", None)
        # Special handling for different request types
        if hasattr(request, "messages"):
            params["messages"] = [{"role": m["role"], "content": "..."} for m in params["messages"]]
        if hasattr(request, "file"):
            params["file"] = "...[truncated]..."
        return params
    
    def record_metrics(self, request: InferenceRequest, start_time: float,
                       queue_wait_time: float, success: bool):
        """Record metrics for a processed request"""
        request_type = request.__class__.__name__.replace("Request", "").lower()
        end_time = perf_counter()
        
        # Initialize throughput counts if needed
        if request_type not in self.throughput_counts:
            self.throughput_counts[request_type] = {}
        if request.model not in self.throughput_counts[request_type]:
            self.throughput_counts[request_type][request.model] = 0
        
        # Increment throughput count if successful
        if success:
            self.throughput_counts[request_type][request.model] += 1
        
        # Create metrics record
        metrics = RequestMetrics(
            model=request.model,
            request_type=request_type,
            start_time=datetime.fromtimestamp(start_time),
            end_time=datetime.fromtimestamp(end_time),
            processing_time=end_time - start_time,
            queue_wait_time=queue_wait_time,
            success=success
        )
        
        # Add to metrics (keep max 1000)
        self.request_metrics.append(metrics)
        if len(self.request_metrics) > 1000:
            self.request_metrics.pop(0)
    
    def get_error_metrics(self) -> Dict[str, Any]:
        """Get error metrics and recent samples"""
        return {
            "counts_by_type": self.error_counts,
            "recent_errors": [err.dict() for err in self.recent_errors[-10:]],
            "total_errors": sum(
                count for model_counts in self.error_counts.values()
                for count in model_counts.values()
            )
        }
    
    def get_request_metrics(self) -> Dict[str, Any]:
        """Get request processing metrics"""
        if not self.request_metrics:
            return {}
            
        # Calculate averages
        avg_processing = sum(m.processing_time for m in self.request_metrics) / len(self.request_metrics)
        avg_wait = sum(m.queue_wait_time for m in self.request_metrics) / len(self.request_metrics)
        
        return {
            "throughput": self.throughput_counts,
            "avg_processing_time": avg_processing,
            "avg_wait_time": avg_wait,
            "recent_metrics": [m.dict() for m in self.request_metrics[-10:]],
            "total_requests": len(self.request_metrics),
            "success_rate": sum(1 for m in self.request_metrics if m.success) / len(self.request_metrics)
        }