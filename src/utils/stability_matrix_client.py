import os
import httpx
import base64
from datetime import datetime
from typing import List, Any

# Base URL for Stability Matrix API - configured via STABILITY_MATRIX_URL environment variable
STABILITY_MATRIX_BASE_URL = os.getenv("STABILITY_MATRIX_URL", "http://localhost:7860")

async def txt2img(
    prompt: str,
    steps: int = 50,
    width: int = 640,
    height: int = 360,
    cfg_scale: float = 1,
    **kwargs
) -> List[str]:
    """Generate images from text prompt using txt2img API endpoint.
    
    Args:
        prompt: Text prompt for image generation
        steps: Number of diffusion steps (default: 50)
        width: Image width (default: 640)
        height: Image height (default: 360)
        cfg_scale: Classifier free guidance scale (default: 1)
        **kwargs: Additional parameters for the API
        
    Returns:
        List of file paths where generated images were saved
        
    Raises:
        httpx.HTTPStatusError: If request fails
    """
    payload = {
        "prompt": prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        **kwargs
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{STABILITY_MATRIX_BASE_URL}/sdapi/v1/txt2img",
            json=payload
        )
        resp.raise_for_status()
        response = resp.json()
        timestamp = datetime.now().strftime("%Y_%m_%d_%S")
        os.makedirs("generated_images", exist_ok=True)
        file_paths = []
        for i, img_data in enumerate(response['images']):
            img_bytes = base64.b64decode(img_data)
            filename = f"{timestamp}_{i}.png"
            filepath = os.path.join("generated_images", filename)
            with open(filepath, "wb") as f:
                f.write(img_bytes)
            file_paths.append(filepath)
        return file_paths

async def img2img(
    init_images: List[str],
    prompt: str,
    steps: int = 50,
    width: int = 640,
    height: int = 360,
    cfg_scale: float = 1,
    **kwargs
) -> List[str]:
    """Generate modified images from input images using img2img API endpoint.
    
    Args:
        init_images: List of base64 encoded input images
        prompt: Text prompt for image generation
        steps: Number of diffusion steps (default: 50)
        width: Image width (default: 640)
        height: Image height (default: 360)
        cfg_scale: Classifier free guidance scale (default: 1)
        **kwargs: Additional parameters for the API
        
    Returns:
        List of base64 encoded generated images
        
    Raises:
        httpx.HTTPStatusError: If request fails
    """
    payload = {
        "init_images": init_images,
        "prompt": prompt,
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        **kwargs
    }

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{STABILITY_MATRIX_BASE_URL}/sdapi/v1/img2img",
            json=payload
        )
        resp.raise_for_status()
        response = resp.json()
        return response['images']

