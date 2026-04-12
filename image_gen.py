"""
image_gen.py — Local b-roll image generation via Stable Diffusion
Uses diffusers + CUDA. Generates vertical 9:16 b-roll frames from Gemma's prompts.
"""

import logging
import torch
from pathlib import Path
from PIL import Image

log = logging.getLogger(__name__)

_pipe = None   # lazy load — only initialise when first needed


def _load_pipeline():
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    log.info("[imggen] Loading Stable Diffusion (SDXL-turbo)...")
    model_id = "stabilityai/sdxl-turbo"

    _pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    _pipe.scheduler = DPMSolverMultistepScheduler.from_config(_pipe.scheduler.config)
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _pipe = _pipe.to(_device)
    if _device == "cuda":
        _pipe.enable_xformers_memory_efficient_attention()
    log.info(f"[imggen] Stable Diffusion ready on {_device}.")
    return _pipe


# Negative prompt — applied to every generation to avoid bad outputs
NEGATIVE_PROMPT = (
    "blurry, low quality, watermark, text, logo, distorted, ugly, "
    "bad anatomy, extra limbs, cropped, worst quality, low resolution"
)


def generate_broll_image(prompt: str, output_path: str,
                          width: int = 576, height: int = 1024,
                          steps: int = 4) -> str:
    """
    Generate a single vertical b-roll image.
    steps=4 is fast (SDXL-turbo optimised); increase to 8 for quality.
    Returns output_path.
    """
    pipe = _load_pipeline()

    log.info(f"[imggen] Generating: '{prompt[:60]}...'")

    # Enhance prompt for cinematic vertical b-roll
    enhanced = (
        f"{prompt}, "
        "cinematic lighting, sharp focus, 9:16 vertical aspect ratio, "
        "professional photography, vibrant colors, high detail, 4K"
    )

    with torch.inference_mode():
        result = pipe(
            prompt          = enhanced,
            negative_prompt = NEGATIVE_PROMPT,
            width           = width,
            height          = height,
            num_inference_steps = steps,
            guidance_scale  = 0.0,   # SDXL-turbo works with guidance=0
        )

    img: Image.Image = result.images[0]
    img.save(output_path)
    log.info(f"[imggen] Saved: {output_path}")
    return output_path


def generate_all_broll(broll_prompts: list[dict], work_dir: str) -> list[dict]:
    """
    Generate all b-roll images from Gemma's prompts.
    broll_prompts: [{time: float, prompt: str, duration: float}, ...]
    Returns: [{time, duration, image_path}, ...]
    """
    work = Path(work_dir)
    results = []

    for i, item in enumerate(broll_prompts):
        out = str(work / f"broll_{i:02d}.png")
        try:
            generate_broll_image(item["prompt"], out)
            results.append({
                "time":       item["time"],
                "duration":   item.get("duration", 2.0),
                "image_path": out,
            })
        except Exception as e:
            log.warning(f"[imggen] Skipped b-roll {i}: {e}")

    return results
