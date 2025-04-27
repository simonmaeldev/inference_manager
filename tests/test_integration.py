import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from time import perf_counter
import asyncio
import psutil

from src.models.queues import InferenceQueues
from src.models.schemas import (
    Txt2TxtRequest,
    Txt2ImgRequest,
    Img2ImgRequest,
    Img2TxtRequest
)
from main import app
from src.routes.mcp import setup_mcp_server

@pytest.fixture
def test_client():
    """Fixture providing FastAPI test client"""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Fixture providing async test client"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def mock_queues():
    """Fixture providing mock queues"""
    return InferenceQueues()

@pytest.fixture
def mock_mcp_server(mock_queues):
    """Fixture providing mock MCP server"""
    return setup_mcp_server()

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

@pytest.mark.asyncio
class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    async def test_health_check(self, async_client):
        """Test health check endpoint"""
        response = await async_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "version": "0.1.0"
        }

    async def test_queue_status(self, async_client, mock_queues):
        """Test queue status endpoint"""
        with patch("src.routes.mcp.queues", mock_queues):
            response = await async_client.get("/queue-status")
            assert response.status_code == 200
            data = response.json()
            assert "queues" in data
            assert "system" in data
            assert isinstance(data["system"]["cpu_percent"], float)
            assert isinstance(data["system"]["memory_percent"], float)

@pytest.mark.asyncio
class TestMCPTools:
    """Test suite for MCP tools"""
    
    async def test_txt2txt_tool(self, mock_mcp_server, mock_txt2txt_request):
        """Test txt2txt tool"""
        with patch.object(mock_txt2txt_request, "process", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"result": "success"}
            result = await mock_mcp_server.tools["txt2txt"](
                model="Qwen2.5-32B-Instruct",
                messages=[{"role": "user", "content": "Hello"}]
            )
            assert result == {"result": "success"}
            assert mock_process.called

    async def test_txt2img_tool(self, mock_mcp_server, mock_txt2img_request):
        """Test txt2img tool"""
        with patch.object(mock_txt2img_request, "process", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"image": "base64encoded"}
            result = await mock_mcp_server.tools["txt2img"](
                model="Flux-Dev",
                prompt="A beautiful sunset"
            )
            assert result == {"image": "base64encoded"}
            assert mock_process.called

    async def test_img2img_tool(self, mock_mcp_server, mock_img2img_request):
        """Test img2img tool"""
        with patch.object(mock_img2img_request, "process", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"image": "base64encoded"}
            result = await mock_mcp_server.tools["img2img"](
                model="Flux-Dev",
                file="base64encodedimage",
                prompt="Make this more colorful"
            )
            assert result == {"image": "base64encoded"}
            assert mock_process.called

    async def test_img2txt_tool(self, mock_mcp_server, mock_img2txt_request):
        """Test img2txt tool"""
        with patch.object(mock_img2txt_request, "process", new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {"text": "description"}
            result = await mock_mcp_server.tools["img2txt"](
                model="Qwen2.5-VL-32B-Instruct",
                messages=[{"role": "user", "content": "Describe this image"}]
            )
            assert result == {"text": "description"}
            assert mock_process.called

    async def test_error_metrics_tool(self, mock_mcp_server, mock_queues, mock_txt2txt_request, mock_error):
        """Test error metrics tool"""
        mock_queues.record_error(mock_txt2txt_request, mock_error)
        with patch("src.routes.mcp.queues", mock_queues):
            metrics = await mock_mcp_server.tools["get_error_metrics"]()
            assert metrics["counts_by_type"]["txt2txt"]["Qwen2.5-32B-Instruct"] == 1
            assert metrics["total_errors"] == 1

    async def test_request_metrics_tool(self, mock_mcp_server, mock_queues, mock_txt2img_request):
        """Test request metrics tool"""
        start_time = perf_counter()
        mock_queues.record_metrics(mock_txt2img_request, start_time, 0.1, True)
        with patch("src.routes.mcp.queues", mock_queues):
            metrics = await mock_mcp_server.tools["get_request_metrics"]()
            assert metrics["throughput"]["txt2img"]["Flux-Dev"] == 1
            assert metrics["success_rate"] == 1.0

@pytest.mark.asyncio
class TestErrorScenarios:
    """Test suite for error scenarios"""
    
    async def test_invalid_model_error(self, async_client):
        """Test invalid model error"""
        with patch("src.models.schemas.validate_model", return_value=False):
            response = await async_client.post(
                "/mcp/txt2txt",
                json={
                    "model": "Invalid-Model",
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            )
            assert response.status_code == 422
            assert "Model Invalid-Model not found" in response.text

    async def test_processing_error(self, mock_mcp_server, mock_txt2txt_request):
        """Test processing error"""
        with patch.object(mock_txt2txt_request, "process", side_effect=Exception("Processing failed")):
            with pytest.raises(Exception):
                await mock_mcp_server.tools["txt2txt"](
                    model="Qwen2.5-32B-Instruct",
                    messages=[{"role": "user", "content": "Hello"}]
                )

@pytest.mark.asyncio
class TestQueueProcessing:
    """Test suite for queue processing"""
    
    async def test_queue_priority(self, mock_queues, mock_mcp_server):
        """Test queue priority processing"""
        # Add requests in reverse priority order
        mock_queues.queues["Qwen2.5-32B-Instruct"].append(
            Img2TxtRequest(
                model="Qwen2.5-VL-32B-Instruct",
                messages=[{"role": "user", "content": "Describe"}]
            )
        )
        mock_queues.queues["Flux-Dev"].append(
            Img2ImgRequest(
                model="Flux-Dev",
                prompt="Test",
                file="base64"
            )
        )
        mock_queues.queues["Flux-Dev"].append(
            Txt2ImgRequest(
                model="Flux-Dev",
                prompt="Test"
            )
        )
        mock_queues.queues["Qwen2.5-32B-Instruct"].append(
            Txt2TxtRequest(
                model="Qwen2.5-32B-Instruct",
                messages=[{"role": "user", "content": "Hello"}]
            )
        )

        # Process queue
        with patch("src.routes.mcp.queues", mock_queues):
            await mock_mcp_server.process_queue()
            assert len(mock_queues.request_metrics) == 4