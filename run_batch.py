# run_batch.py
# Windows-safe batch runner for SDXL + ControlNet (depth) + IP-Adapter (XL)
# - Cross-platform paths (pathlib)
# - CPU fallback (smaller size & steps)
# - Handles diffusers API differences: tries dtype=..., falls back to torch_dtype=...
# - Valid default image encoder repo (CLIP ViT-H)
# - Clear checks + helpful errors

from __future__ import annotations

import contextlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from rembg import remove
import torch

from diffusers import (
    StableDiffusionXLControlNetInpaintPipeline,
    ControlNetModel,
)

from ip_adapter import IPAdapterXL
from ip_adapter.utils import register_cross_attention_hook


# ---------------------------- Config ---------------------------- #
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROLNET_REPO = "diffusers/controlnet-depth-sdxl-1.0"

# Use a valid CLIP Vision model (public) OR switch to a local folder later.
IMAGE_ENCODER = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

# Must exist locally (put your .bin here)
IP_ADAPTER_WEIGHTS = Path("sdxl_models/ip-adapter_sdxl_vit-h.bin")

# I/O folders
ROOT = Path(".")
IN_DIR = ROOT / "demo_assets" / "input_imgs"
TEX_DIR = ROOT / "demo_assets" / "material_exemplars"
DEPTH_DIR = ROOT / "demo_assets" / "depths"
OUT_DIR = ROOT / "demo_assets" / "output_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Runtime knobs
SEED = 42
STEPS_CUDA = 30
STEPS_CPU = 8
SIZE_CUDA = 1024
SIZE_CPU = 512


# ------------------------- Utilities --------------------------- #
def assert_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at: {path.resolve()}")


def list_required_pngs(directory: Path):
    items = list(directory.glob("*.png"))
    if not items:
        raise FileNotFoundError(
            f"No PNGs in {directory.resolve()}\n"
            f"Expected structure:\n"
            f"  {IN_DIR}/AAA1111.png\n"
            f"  {TEX_DIR}/brick_red.png\n"
            f"  {DEPTH_DIR}/AAA1111.png\n"
        )
    return items


def from_pretrained_compat(cls, repo_id: str, device: str, *, want_fp16_variant: bool, **common):
    """
    Load diffusers models across versions:
    - Try dtype=... (newer diffusers)
    - Fallback to torch_dtype=... (older diffusers)
    """
    if device == "cuda":
        dtype = torch.float16
        kwargs = dict(common)
        if want_fp16_variant:
            kwargs.update(variant="fp16")
        # Try new API
        try:
            return cls.from_pretrained(repo_id, dtype=dtype, **kwargs)
        except TypeError:
            # Old API
            return cls.from_pretrained(repo_id, torch_dtype=dtype, **kwargs)
    else:
        dtype = torch.float32
        kwargs = dict(common)
        # CPU: no fp16 variant
        try:
            return cls.from_pretrained(repo_id, dtype=dtype, **kwargs)
        except TypeError:
            return cls.from_pretrained(repo_id, torch_dtype=dtype, **kwargs)


# ------------------------ Pipeline build ------------------------ #
def build_pipe(device: str):
    # ControlNet (depth)
    controlnet = from_pretrained_compat(
        ControlNetModel,
        CONTROLNET_REPO,
        device,
        want_fp16_variant=(device == "cuda"),
        use_safetensors=True,
    )

    # SDXL pipeline
    pipe = from_pretrained_compat(
        StableDiffusionXLControlNetInpaintPipeline,
        BASE_MODEL,
        device,
        want_fp16_variant=(device == "cuda"),
        controlnet=controlnet,
        use_safetensors=True,
        add_watermarker=False,
    )

    # Place on device. (Avoid mixing enable_model_cpu_offload with manual .to)
    pipe.to(device)

    # Light memory helpers
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    # Hook for IP-Adapter cross-attention
    pipe.unet = register_cross_attention_hook(pipe.unet)

    return pipe


# -------------------- Target/mask preparation ------------------- #
def build_init_and_mask(target_image: Image.Image, target_size: int):
    """Return (init_img RGB, mask L) resized to target_size."""
    # Remove background (RGBA)
    rm_bg = remove(target_image)
    alpha = rm_bg.split()[3]
    mask_L = alpha.point(lambda a: 255 if a > 0 else 0)  # binary mask L

    invert_mask_RGB = ImageChops.invert(mask_L.convert("RGB"))

    gray_rgb = target_image.convert("L").convert("RGB")
    gray_rgb = ImageEnhance.Brightness(gray_rgb).enhance(1.0)

    grayscale_img = ImageChops.darker(gray_rgb, mask_L.convert("RGB"))
    img_black_mask = ImageChops.darker(target_image, invert_mask_RGB)
    init_img = ImageChops.lighter(img_black_mask, grayscale_img)

    init_img = init_img.resize((target_size, target_size), Image.BICUBIC)
    mask_L = mask_L.resize((target_size, target_size), Image.NEAREST)
    return init_img, mask_L


# --------------------------- One job ---------------------------- #
def run_one(ip_model: IPAdapterXL, obj_png: Path, tex_png: Path, depth_png: Path, out_png: Path, device: str):
    target_image = Image.open(obj_png).convert("RGB")
    ip_image = Image.open(tex_png)

    depth_np = np.array(Image.open(depth_png))
    if depth_np.dtype != np.uint8:  # compress 16-bit depth to 8-bit like your original code
        depth_np = (depth_np / 256).astype("uint8")

    target_size = SIZE_CUDA if device == "cuda" else SIZE_CPU
    depth_map = Image.fromarray(depth_np).resize((target_size, target_size))

    init_img, mask_L = build_init_and_mask(target_image, target_size)

    steps = STEPS_CUDA if device == "cuda" else STEPS_CPU
    autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()

    with autocast_ctx:
        images = ip_model.generate(
            pil_image=ip_image,
            image=init_img,
            control_image=depth_map,
            mask_image=mask_L,
            controlnet_conditioning_scale=0.9,
            num_samples=1,
            num_inference_steps=steps,
            seed=SEED,
        )

    images[0].save(out_png)


# ----------------------------- Main ----------------------------- #
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("⚠️  CUDA GPU not detected — running in CPU fallback mode (slower).")
    else:
        torch.cuda.empty_cache()

    # Critical files / dirs
    if not IP_ADAPTER_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Missing IP-Adapter weights at: {IP_ADAPTER_WEIGHTS.resolve()}\n"
            f"Expected file: 'ip-adapter_sdxl_vit-h.bin' under 'sdxl_models/'."
        )

    list_required_pngs(IN_DIR)
    list_required_pngs(TEX_DIR)
    list_required_pngs(DEPTH_DIR)

    # Build pipeline and IP-Adapter
    pipe = build_pipe(device)

    # IMAGE_ENCODER can be a HF repo id (string) or a local folder path (string)
    image_encoder_path = IMAGE_ENCODER  # e.g., "models/image_encoder" if you download locally
    ip_model = IPAdapterXL(pipe, image_encoder_path, str(IP_ADAPTER_WEIGHTS), device)

    # Iterate texture × object
    textures = sorted(TEX_DIR.glob("*.png"))
    objs = sorted(IN_DIR.glob("*.png"))

    for tex_png in textures:
        for obj_png in objs:
            depth_png = DEPTH_DIR / f"{obj_png.stem}.png"
            if not depth_png.exists():
                print(f"❌ Missing depth for {obj_png.name} (expected {depth_png.name}). Skipping.")
                continue

            out_png = OUT_DIR / f"{obj_png.stem}_{tex_png.stem}.png"
            print(f"→ {obj_png.name} × {tex_png.name} → {out_png.name}")
            try:
                run_one(ip_model, obj_png, tex_png, depth_png, out_png, device)
            except Exception as e:
                print(f"⚠️  Failed on {obj_png.name} × {tex_png.name}: {e}")

    print("✅ Done.")


if __name__ == "__main__":
    main()
