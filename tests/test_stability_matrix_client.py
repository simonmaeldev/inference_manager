import pytest
import httpx
from src.utils.stability_matrix_client import STABILITY_MATRIX_BASE_URL

@pytest.mark.asyncio
async def test_get_prompt_styles():
    """Test GET request to /sdapi/v1/prompt-styles endpoint"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{STABILITY_MATRIX_BASE_URL}/sdapi/v1/prompt-styles",
            headers={"accept": "application/json"}
        )
        response.raise_for_status()
        
        # Basic response validation
        assert response.headers["content-type"] == "application/json"
        assert isinstance(response.json(), list)  # Expecting array of prompt styles