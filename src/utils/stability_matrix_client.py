import os
import httpx
import base64
from datetime import datetime
from typing import List, Any

# Base URL for Stability Matrix API - configured via STABILITY_MATRIX_URL environment variable
STABILITY_MATRIX_BASE_URL = os.getenv("STABILITY_MATRIX_URL", "http://localhost:7861")
# Timeout settings for API calls (10 minutes for read, 2 minutes for connect)
API_TIMEOUT = httpx.Timeout(600.0, connect=120.0)

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
        "negative_prompt": "",
        "steps": steps,
        "width": width,
        "height": height,
        "cfg_scale": cfg_scale,
        "batch_size": 1,
        "n_iter": 1,
        "sampler_name": "Euler",
        "scheduler": "Simple",
        "distilled_cfg_scale": 3.5,
        "denoising_strength": 0.7,
        "seed": -1,
        "seed_enable_extras": True,
        "subseed": -1,
        "subseed_strength": 0,
        "seed_resize_from_h": -1,
        "seed_resize_from_w": -1,
        "tiling": False,
        "restore_faces": False,
        "enable_hr": False,
        "hr_scale": 2,
        "hr_upscaler": "Latent",
        "hr_second_pass_steps": 0,
        "hr_resize_x": 0,
        "hr_resize_y": 0,
        "hr_prompt": "",
        "hr_negative_prompt": "",
        "hr_cfg": 1,
        "hr_distilled_cfg": 3.5,
        "styles": [],
        "override_settings": {},
        "override_settings_restore_afterwards": True,
        "script_name": None,
        "script_args": [],
        "do_not_save_samples": False,
        "do_not_save_grid": False,
        "disable_extra_networks": False,
        "comments": {},
        "s_churn": 0.0,
        "s_min_uncond": 0.0,
        "s_noise": 1.0,
        "s_tmin": 0.0,
        "s_tmax": None,
        "alwayson_scripts": {
            "API payload": {"args": []},
            "ControlNet": {"args": [
                {
                    "batch_image_dir": "", "batch_input_gallery": None, "batch_mask_dir": "", "batch_mask_gallery": None,
                    "control_mode": "Balanced", "enabled": False, "generated_image": None, "guidance_end": 1.0,
                    "guidance_start": 0.0, "hr_option": "Both", "image": None, "image_fg": None, "input_mode": "simple",
                    "mask_image": None, "mask_image_fg": None, "model": "None", "module": "None", "pixel_perfect": False,
                    "processor_res": -1, "resize_mode": "Crop and Resize", "save_detected_map": True, "threshold_a": -1,
                    "threshold_b": -1, "use_preview_as_input": False, "weight": 1
                },
                {
                    "batch_image_dir": "", "batch_input_gallery": None, "batch_mask_dir": "", "batch_mask_gallery": None,
                    "control_mode": "Balanced", "enabled": False, "generated_image": None, "guidance_end": 1.0,
                    "guidance_start": 0.0, "hr_option": "Both", "image": None, "image_fg": None, "input_mode": "simple",
                    "mask_image": None, "mask_image_fg": None, "model": "None", "module": "None", "pixel_perfect": False,
                    "processor_res": -1, "resize_mode": "Crop and Resize", "save_detected_map": True, "threshold_a": -1,
                    "threshold_b": -1, "use_preview_as_input": False, "weight": 1
                },
                {
                    "batch_image_dir": "", "batch_input_gallery": None, "batch_mask_dir": "", "batch_mask_gallery": None,
                    "control_mode": "Balanced", "enabled": False, "generated_image": None, "guidance_end": 1.0,
                    "guidance_start": 0.0, "hr_option": "Both", "image": None, "image_fg": None, "input_mode": "simple",
                    "mask_image": None, "mask_image_fg": None, "model": "None", "module": "None", "pixel_perfect": False,
                    "processor_res": -1, "resize_mode": "Crop and Resize", "save_detected_map": True, "threshold_a": -1,
                    "threshold_b": -1, "use_preview_as_input": False, "weight": 1
                }
            ]},
            "DynamicThresholding (CFG-Fix) Integrated": {"args": [False, 7, 1, "Constant", 0, "Constant", 0, 1, "enable", "MEAN", "AD", 1]},
            "Extra options": {"args": []},
            "FreeU Integrated (SD 1.x, SD 2.x, SDXL)": {"args": [False, 1.01, 1.02, 0.99, 0.95, 0, 1]},
            "Kohya HRFix Integrated": {"args": [False, 3, 2, 0, 0.35, True, "bicubic", "bicubic"]},
            "LatentModifier Integrated": {"args": [False, 0, "anisotropic", 0, "reinhard", 100, 0, "subtract", 0, 0, "gaussian", "add", 0, 100, 127, 0, "hard_clamp", 5, 0, "None", "None"]},
            "MultiDiffusion Integrated": {"args": [False, "MultiDiffusion", 768, 768, 64, 4]},
            "Never OOM Integrated": {"args": [False, False]},
            "PerturbedAttentionGuidance Integrated": {"args": [False, 3, 0, 0, 1]},
            "Refiner": {"args": [False, "", 0.8]},
            "Sampler": {"args": [steps, "Euler", "Simple"]},
            "Seed": {"args": [-1, False, -1, 0, 0, 0]},
            "SelfAttentionGuidance Integrated (SD 1.x, SD 2.x, SDXL)": {"args": [False, 0.5, 2, 1]},
            "StyleAlign Integrated": {"args": [False, 1]}
        }
    }
    
    # Override with any additional kwargs
    payload.update(kwargs)

    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        resp = await client.post(
            f"{STABILITY_MATRIX_BASE_URL}/sdapi/v1/txt2img",
            json=payload,
            timeout=API_TIMEOUT
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

    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        resp = await client.post(
            f"{STABILITY_MATRIX_BASE_URL}/sdapi/v1/img2img",
            json=payload,
            timeout=API_TIMEOUT
        )
        resp.raise_for_status()
        response = resp.json()
        return response['images']

