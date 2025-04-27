import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from time import perf_counter
from collections import deque
from src.models.queues import InferenceQueues, ErrorRecord, RequestMetrics
from src.models.schemas import (
    Txt2TxtRequest,
    Txt2ImgRequest,
    Img2ImgRequest,
    Img2TxtRequest
)

@pytest.fixture
def queues():
    """Fixture providing initialized InferenceQueues"""
    return InferenceQueues()

@pytest.fixture
def mock_txt2txt_request():
    """Fixture for text-to-text request"""
    return Txt2TxtRequest(
        model="Qwen2.5-32B-Instruct",
        messages=[{"role": "user", "content": "Hello"}]
    )

@pytest.fixture
def mock_txt2img_request():
    """Fixture for text-to-image request"""
    return Txt2ImgRequest(
        model="Flux-Dev",
        prompt="A beautiful sunset"
    )

@pytest.fixture
def mock_img2img_request():
    """Fixture for image-to-image request"""
    return Img2ImgRequest(
        model="Flux-Dev",
        prompt="Make this more colorful",
        file="base64encodedimage"
    )

@pytest.fixture
def mock_img2txt_request():
    """Fixture for image-to-text request"""
    return Img2TxtRequest(
        model="Qwen2.5-VL-32B-Instruct",
        messages=[{"role": "user", "content": "Describe this image"}]
    )

@pytest.fixture
def mock_error():
    """Fixture for mock error"""
    return ValueError("Test error")

class TestInferenceQueues:
    """Test suite for InferenceQueues class"""

    def test_initialization(self, queues):
        """Test queue initialization"""
        assert len(queues.queues) == 2  # 2 models
        assert "Qwen2.5-32B-Instruct" in queues.queues
        assert "Flux-Dev" in queues.queues
        assert queues.priority_order == ["txt2txt", "txt2img", "img2img", "img2txt"]

    async def test_record_error(self, queues, mock_txt2txt_request, mock_error):
        """Test error recording"""
        initial_count = len(queues.recent_errors)
        queues.record_error(mock_txt2txt_request, mock_error)
        
        assert len(queues.recent_errors) == initial_count + 1
        assert queues.error_counts["txt2txt"]["Qwen2.5-32B-Instruct"] == 1
        assert queues.recent_errors[-1].error_type == "ValueError"
        assert queues.recent_errors[-1].error_message == "Test error"

    async def test_record_metrics_success(self, queues, mock_txt2img_request):
        """Test metrics recording for successful request"""
        start_time = perf_counter()
        queue_wait_time = 0.1
        queues.record_metrics(mock_txt2img_request, start_time, queue_wait_time, True)
        
        assert len(queues.request_metrics) == 1
        assert queues.request_metrics[-1].success
        assert queues.request_metrics[-1].processing_time > 0
        assert queues.request_metrics[-1].queue_wait_time == queue_wait_time
        assert queues.throughput_counts["txt2img"]["Flux-Dev"] == 1

    async def test_record_metrics_failure(self, queues, mock_img2img_request):
        """Test metrics recording for failed request"""
        start_time = perf_counter()
        queue_wait_time = 0.2
        queues.record_metrics(mock_img2img_request, start_time, queue_wait_time, False)
        
        assert len(queues.request_metrics) == 1
        assert not queues.request_metrics[-1].success
        assert "img2img" not in queues.throughput_counts

    async def test_get_error_metrics(self, queues, mock_txt2txt_request, mock_error):
        """Test error metrics retrieval"""
        queues.record_error(mock_txt2txt_request, mock_error)
        metrics = queues.get_error_metrics()
        
        assert metrics["counts_by_type"]["txt2txt"]["Qwen2.5-32B-Instruct"] == 1
        assert len(metrics["recent_errors"]) == 1
        assert metrics["total_errors"] == 1

    async def test_get_request_metrics(self, queues, mock_txt2img_request):
        """Test request metrics retrieval"""
        start_time = perf_counter()
        queues.record_metrics(mock_txt2img_request, start_time, 0.1, True)
        metrics = queues.get_request_metrics()
        
        assert metrics["throughput"]["txt2img"]["Flux-Dev"] == 1
        assert metrics["avg_processing_time"] > 0
        assert metrics["success_rate"] == 1.0
        assert metrics["total_requests"] == 1

    @pytest.mark.parametrize("request_type,model", [
        ("txt2txt", "Qwen2.5-32B-Instruct"),
        ("txt2img", "Flux-Dev"),
        ("img2img", "Flux-Dev"),
        ("img2txt", "Qwen2.5-VL-32B-Instruct")
    ])
    async def test_request_processing(self, queues, request_type, model):
        """Test request processing for all types"""
        # Create appropriate request based on type
        if request_type == "txt2txt":
            request = Txt2TxtRequest(model=model, messages=[{"role": "user", "content": "Hello"}])
        elif request_type == "txt2img":
            request = Txt2ImgRequest(model=model, prompt="Test prompt")
        elif request_type == "img2img":
            request = Img2ImgRequest(model=model, prompt="Test prompt", file="base64")
        else:
            request = Img2TxtRequest(model=model, messages=[{"role": "user", "content": "Describe"}])
        
        # Mock process method
        with patch.object(request, "process", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"result": "success"}
            await request.process()
            
            # Record metrics
            start_time = perf_counter()
            queues.record_metrics(request, start_time, 0.1, True)
            
            assert mock_process.called
            assert queues.request_metrics[-1].request_type == request_type
            assert queues.request_metrics[-1].model == model

    async def test_priority_ordering(self, queues, mock_txt2txt_request, mock_txt2img_request,
                                   mock_img2img_request, mock_img2txt_request):
        """Test queue priority ordering"""
        # Add requests in reverse priority order
        queues.queues["Qwen2.5-32B-Instruct"].append(mock_img2txt_request)
        queues.queues["Flux-Dev"].append(mock_img2img_request)
        queues.queues["Flux-Dev"].append(mock_txt2img_request)
        queues.queues["Qwen2.5-32B-Instruct"].append(mock_txt2txt_request)
        
        # Check processing order matches priority
        assert queues.priority_order.index("txt2txt") < queues.priority_order.index("txt2img")
        assert queues.priority_order.index("txt2img") < queues.priority_order.index("img2img")
        assert queues.priority_order.index("img2img") < queues.priority_order.index("img2txt")

    async def test_error_handling(self, queues, mock_txt2txt_request):
        """Test error handling scenarios"""
        # Test invalid model
        with pytest.raises(ValueError):
            Txt2TxtRequest(model="Invalid-Model", messages=[{"role": "user", "content": "Hello"}])
        
        # Test error recording
        error = RuntimeError("Processing failed")
        queues.record_error(mock_txt2txt_request, error)
        
        assert len(queues.recent_errors) == 1
        assert queues.error_counts["txt2txt"]["Qwen2.5-32B-Instruct"] == 1
        assert queues.recent_errors[-1].error_type == "RuntimeError"

    async def test_metrics_limits(self, queues, mock_txt2img_request):
        """Test metrics storage limits"""
        # Fill up metrics storage
        for _ in range(1001):
            queues.record_metrics(mock_txt2img_request, perf_counter(), 0.1, True)
        
        assert len(queues.request_metrics) == 1000
        assert len(queues.recent_errors) <= 100  # Also test error limit