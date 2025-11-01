# apply_material_to_wall.py
# Use 800x600 for masks/annotations, run SDXL at 1024x1024 (letterboxed), then crop back.

from __future__ import annotations
import json, contextlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageChops, ImageEnhance
import torch
from diffusers import StableDiffusionXLControlNetInpaintPipeline, ControlNetModel, AutoencoderKL
from ip_adapter import IPAdapterXL
from ip_adapter.utils import register_cross_attention_hook
from diffusers import EulerAncestralDiscreteScheduler

# ─── Config ──────────────────────────────────────────────────────────────────
HOUSE_IMG        = Path("demo_assets/input_imgs/AAA1111.png")
# MATERIAL_IMG     = Path("demo_assets/material_exemplars/Belden__830.png")
MATERIAL_IMG     = Path("demo_assets/input_imgs/stone_wall.jpg")
DEPTH_IMG_OPT    = Path("demo_assets/depths/AAA1111.png")  # optional
DETECTIONS_JSON  = Path("detections.json")
OUT_IMG          = Path("demo_assets/output_images/AAA1111_stone_on_wall.png")

# Annotation canvas (what your frontend uses)
ANN_W, ANN_H = 800, 600

# SDXL working canvas (stable)
WORK_SIZE = 1024  # square for SDXL+ControlNet

BASE_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
CONTROLNET_REPO     = "diffusers/controlnet-depth-sdxl-1.0"
IMAGE_ENCODER       = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
IP_ADAPTER_WEIGHTS  = Path("sdxl_models/ip-adapter_sdxl_vit-h.bin")

SEED = 42
STEPS_GPU = 14
STEPS_CPU = 12

fallback_prompt = "a realistic wall with applied material texture"
fallback_negative_prompt = "blurry, distorted, unrealistic"

# ─── Helpers ─────────────────────────────────────────────────────────────────
def from_pretrained_compat(cls, repo_id: str, device: str, *, want_fp16_variant: bool, **common):
    if device == "cuda":
        dtype = torch.float16
        kwargs = dict(common)
        if want_fp16_variant:
            kwargs.update(variant="fp16")
        try:
            return cls.from_pretrained(repo_id, dtype=dtype, **kwargs)
        except TypeError:
            return cls.from_pretrained(repo_id, torch_dtype=dtype, **kwargs)
    else:
        dtype = torch.float32
        kwargs = dict(common)
        try:
            return cls.from_pretrained(repo_id, dtype=dtype, **kwargs)
        except TypeError:
            return cls.from_pretrained(repo_id, torch_dtype=dtype, **kwargs)

def polygon_to_mask(size: Tuple[int,int], polygon: List[Tuple[int,int]]) -> Image.Image:
    w, h = size
    m = Image.new("L", (w, h), 0)
    if polygon:
        ImageDraw.Draw(m).polygon(polygon, outline=255, fill=255)
    return m

def build_wall_mask_from_json(size: Tuple[int,int], json_path: Path) -> Image.Image:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    W, H = size
    wall_mask = Image.new("L", (W, H), 0)
    door_mask = Image.new("L", (W, H), 0)

    for r in data.get("results", []):
        label = (r.get("label") or "").lower().strip().rstrip(".")
        poly  = r.get("polygon") or []
        poly  = [(int(x), int(y)) for x,y in poly]
        if not poly:
            continue
        if label == "wall":
            wall_mask = ImageChops.lighter(wall_mask, polygon_to_mask((W,H), poly))
        elif label == "door":
            door_mask = ImageChops.lighter(door_mask, polygon_to_mask((W,H), poly))

    # White = inpaint, black = keep
    return ImageChops.subtract(wall_mask, door_mask)

def compute_depth_dpt_800x600(house_800x600: Image.Image) -> Image.Image:
    """Return an 8-bit 'L' depth image at 800x600."""
    try:
        from DPT.dpt.models import DPTDepthModel
        from DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
        from torchvision.transforms import Compose
        import cv2

        weights = Path("DPT/weights/dpt_hybrid-midas-501f0c75.pt")
        if not weights.exists():
            raise FileNotFoundError("Missing DPT weights at DPT/weights/dpt_hybrid-midas-501f0c75.pt")

        model = DPTDepthModel(
            path=str(weights),
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        transform = Compose([
            Resize(384, 384, resize_target=None, keep_aspect_ratio=True, ensure_multiple_of=32,
                   resize_method="minimal", image_interpolation_method=cv2.INTER_CUBIC),
            NormalizeImage(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            PrepareForNet(),
        ])
        model.eval()

        img = np.array(house_800x600)
        with torch.no_grad():
            sample = torch.from_numpy(transform({"image": img})["image"]).unsqueeze(0)
            pred = model.forward(sample)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=img.shape[:2], mode="bicubic", align_corners=False
            ).squeeze().cpu().numpy()

        dmin, dmax = float(np.nanmin(pred)), float(np.nanmax(pred))
        if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax - dmin <= 1e-8:
            dep8 = np.zeros_like(pred, dtype=np.uint8)
        else:
            dep8 = (np.clip((pred - dmin) / (dmax - dmin), 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(dep8, mode="L").resize((ANN_W, ANN_H), Image.BICUBIC)
    except Exception as e:
        raise RuntimeError(f"Depth computation failed: {e}")

def make_init_from_mask(house_800x600: Image.Image, wall_mask_L: Image.Image) -> Image.Image:
    inv = ImageChops.invert(wall_mask_L.convert("RGB"))
    gray = ImageEnhance.Brightness(house_800x600.convert("L").convert("RGB")).enhance(1.0)
    grayscale_img = ImageChops.darker(gray, wall_mask_L.convert("RGB"))
    img_black_mask = ImageChops.darker(house_800x600, inv)
    return ImageChops.lighter(img_black_mask, grayscale_img)  # 800x600

def letterbox_to_square(img: Image.Image, size: int) -> tuple[Image.Image, tuple[int,int,int,int]]:
    """Return square canvas and (x,y,w,h) of the pasted region."""
    w, h = img.size
    scale = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    new_w -= new_w % 8
    new_h -= new_h % 8
    if new_w == 0: new_w = 8
    if new_h == 0: new_h = 8
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    canvas = Image.new(img.mode, (size, size), 0 if img.mode == "L" else (0,0,0))
    x = (size - new_w) // 2
    y = (size - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas, (x, y, new_w, new_h)

def unletterbox_from_square(img_sq: Image.Image, roi: tuple[int,int,int,int], out_wh: tuple[int,int]) -> Image.Image:
    x, y, w, h = roi
    crop = img_sq.crop((x, y, x+w, y+h))
    return crop.resize(out_wh, Image.BICUBIC)

def save_debug(tag: str, im: Image.Image):
    dbg = Path("demo_assets/output_images/_debug_latest")
    dbg.mkdir(parents=True, exist_ok=True)
    im.save(dbg / f"{tag}.png")

def looks_invalid(pil_img: Image.Image) -> bool:
    arr = np.asarray(pil_img, dtype=np.float32)
    if not np.isfinite(arr).all():  # NaNs or infs → invalid
        return True
    if arr.max() <= 1 and arr.mean() < 0.5:
        return True
    return False

def build_pipe(device: str):
    # ControlNet (depth SDXL)
    controlnet = from_pretrained_compat(
        ControlNetModel,
        CONTROLNET_REPO,
        device,
        want_fp16_variant=(device == "cuda"),
        use_safetensors=True,
    )

    pipe_kwargs = dict(controlnet=controlnet, use_safetensors=True)

    # Try the stable fp32 VAE; only pass it if it loads
    try:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            use_safetensors=True,
        )
        vae = vae.to(dtype=torch.float32)
        pipe_kwargs["vae"] = vae
        print("✔ Using madebyollin/sdxl-vae-fp16-fix (fp32)")
    except Exception as e:
        print(f"⚠ Could not load fixed VAE, using model's default VAE: {e}")

    # IMPORTANT: do NOT pass vae=None
    pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        **pipe_kwargs,
    )

    pipe.to(device)

    # Ensure a VAE is present and fp32
    assert pipe.vae is not None, "Pipeline VAE did not load"
    pipe.vae.to(dtype=torch.float32)

    # Optional stabilizers
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    print("UNet in_channels:", getattr(pipe.unet.config, "in_channels", "?"))  # should be 9 or 10 for inpaint
    try:
        print("VAE dtype:", next(pipe.vae.parameters()).dtype)
    except StopIteration:
        pass

    return pipe

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    assert HOUSE_IMG.exists(), f"Missing house image: {HOUSE_IMG}"
    assert MATERIAL_IMG.exists(), f"Missing material image: {MATERIAL_IMG}"
    assert IP_ADAPTER_WEIGHTS.exists(), f"Missing IP-Adapter weights: {IP_ADAPTER_WEIGHTS}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps  = STEPS_GPU if device == "cuda" else STEPS_CPU

    # 1) Resize the house to your annotation canvas (800x600)
    house = Image.open(HOUSE_IMG).convert("RGB").resize((ANN_W, ANN_H), Image.BICUBIC)
    ip_image = Image.open(MATERIAL_IMG).convert("RGB")  # swatch; will be used as-is by IP-Adapter
    save_debug("house_800x600", house)

    # 2) Build mask from annotations (white = paint)
    wall_mask = build_wall_mask_from_json((ANN_W, ANN_H), DETECTIONS_JSON)
    coverage = (np.array(wall_mask) > 0).mean() * 100
    print(f"Mask coverage ~ {coverage:.2f}%")
    save_debug("mask_800x600", wall_mask)

    if coverage < 0.05:
        raise ValueError("Mask is ~empty; check your polygon coordinates / scaling.")

    # 3) Depth at 800x600
    if DEPTH_IMG_OPT.exists():
        dep = Image.open(DEPTH_IMG_OPT)
        dep = dep.convert("L") if dep.mode != "L" else dep
        depth_800x600 = dep.resize((ANN_W, ANN_H), Image.BICUBIC)
    else:
        depth_800x600 = compute_depth_dpt_800x600(house)

    save_debug("depth_800x600", depth_800x600)

    house_1024, roi = letterbox_to_square(house, WORK_SIZE)
    mask_1024, _    = letterbox_to_square(wall_mask, WORK_SIZE)
    depth_1024, _   = letterbox_to_square(depth_800x600, WORK_SIZE)

    # Make mask strictly binary: 0 or 255 (no gray)
    def to_binary_mask(m: Image.Image) -> Image.Image:
        m = m.convert("L")
        return m.point(lambda p: 255 if p >= 128 else 0)

    mask_1024 = to_binary_mask(mask_1024)

    # Depth: ensure 3-channel for ControlNet-depth robustness
    depth_1024 = depth_1024.convert("RGB")

    save_debug("house_1024", house_1024)
    save_debug("mask_1024_bin", mask_1024)
    save_debug("depth_1024_rgb", depth_1024)

    # 6) Build pipeline + IP-Adapter
    pipe = build_pipe(device)
    orig_unet = pipe.unet
    ip_model = IPAdapterXL(pipe, IMAGE_ENCODER, str(IP_ADAPTER_WEIGHTS), device)

    # 7) Run generation
    autocast_ctx = torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()
    def run_once(mask_img):
        with autocast_ctx:
            return ip_model.generate(
                pil_image=ip_image,
                image=house_1024,
                control_image=depth_1024,
                mask_image=mask_img,                       # white = inpaint region
                controlnet_conditioning_scale=0.7,
                num_samples=1,
                num_inference_steps=steps,
                seed=SEED,
            )[0]
        
    def run_ip_adapter(mask_img):
        # Temporarily hook UNet only while using IP-Adapter
        try:
            pipe.unet = register_cross_attention_hook(orig_unet)
            with autocast_ctx:
                return ip_model.generate(
                    pil_image=ip_image,
                    image=house_1024,
                    control_image=depth_1024,
                    mask_image=mask_img,                 # white = inpaint region
                    controlnet_conditioning_scale=0.8,   # a bit stronger for depth
                    num_samples=1,
                    num_inference_steps=steps,
                    seed=SEED,
                )[0]
        finally:
            pipe.unet = orig_unet 
    # out_1024 = run_once(mask_1024)
    out_1024 = run_ip_adapter(mask_1024)
    save_debug("raw_out_1024", out_1024)

    # 8) Fallbacks for black/NaN outputs
    if looks_invalid(out_1024):
        print("⚠️ Output looks invalid. Retrying with INVERTED mask…")
        inv_mask_1024 = ImageChops.invert(mask_1024)
        out_1024 = run_once(inv_mask_1024)
        save_debug("raw_out_1024_retry_invMask", out_1024)

    if looks_invalid(out_1024):
        print("⚠️ Still invalid. Retrying WITHOUT IP-Adapter (plain inpaint)…")
        # Plain inpaint without IP-Adapter:
        with autocast_ctx:
            out_1024 = pipe(
                prompt=fallback_prompt,
                negative_prompt=fallback_negative_prompt,
                image=house_1024,
                mask_image=mask_1024,
                control_image=depth_1024,
                controlnet_conditioning_scale=0.8,
                strength=0.98,
                # num_inference_steps=steps,
                generator=torch.Generator(device).manual_seed(SEED),
                num_inference_steps=STEPS_GPU,
                guidance_scale=7.5,
            ).images[0]
        save_debug("raw_out_1024_plain_inpaint", out_1024)

    # 9) Unletterbox back to 800x600 and composite with original
    out_800x600 = unletterbox_from_square(out_1024, roi, (ANN_W, ANN_H))
    save_debug("raw_out_800x600", out_800x600)
    final = Image.composite(out_800x600, house, wall_mask)
    final.save(OUT_IMG)
    print(f"✅ saved: {OUT_IMG.resolve()}")

if __name__ == "__main__":
    main()



