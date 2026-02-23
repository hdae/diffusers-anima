"""Full-featured Anima pipeline sample.

Demonstrates every major API surface in one runnable script:
  - from_pretrained and from_single_file loading
  - Memory optimisations: VAE slicing, VAE tiling, xformers, CPU offload
  - Scheduler configuration (sampler, sigma_schedule, eta, s_noise)
  - Text-to-image generation
  - encode_prompt pre-computation for prompt reuse
  - Image-to-image (img2img) with strength
  - Inpainting with a mask
  - Step callback
  - LoRA weight loading
  - Batch generation (num_images_per_prompt)

Usage:
    uv run python scripts/sample_full.py
    uv run python scripts/sample_full.py --model hdae/diffusers-anima-preview
    uv run python scripts/sample_full.py --model path/to/model.safetensors
    uv run python scripts/sample_full.py --cpu-offload
    uv run python scripts/sample_full.py --vae-slicing --vae-tiling
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from diffusers_anima import AnimaFlowMatchEulerDiscreteScheduler, AnimaPipeline

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = os.getenv(
    "ANIMA_PRETRAINED_MODEL_PATH", "hdae/diffusers-anima-preview"
)
_PROMPT = (
    "masterpiece, best quality, score 9, score 8, newest, absurdres, very aesthetic, "
    "highres, 1girl, solo, long hair, blue eyes, white blouse, pleated skirt, "
    "looking at viewer, gentle smile, smiling"
)
_NEG_PROMPT = (
    "worst quality, low quality, score_1, score_2, score_3, monochrome, bad anatomy, "
    "bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, "
    "normal quality, jpeg artifacts, nsfw, nude"
)
_WIDTH, _HEIGHT = 1024, 1024
_STEPS = 28
_GUIDANCE = 4.0
_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _make_inpaint_mask(width: int, height: int) -> Image.Image:
    """Create a simple ellipse mask for inpainting demo (white = inpaint region)."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    cx, cy = width // 2, height // 3
    rx, ry = width // 5, height // 6
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    return mask


def _step_callback(
    pipeline: Any, step_index: int, timestep: torch.Tensor, kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    """Print progress every 5 steps; return None to leave kwargs unchanged."""
    if step_index % 5 == 0:
        total = pipeline.scheduler.config.get("num_train_timesteps", "?")
        print(f"  step {step_index:3d}  timestep={float(timestep):.3f}  (of {total})")
    return None


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------


def load_pipeline(args: argparse.Namespace) -> AnimaPipeline:
    model: str = args.model

    # from_single_file for raw .safetensors; from_pretrained for HF repos / dirs
    if model.endswith(".safetensors") or model.endswith(".ckpt"):
        print(f"Loading from single file: {model}")
        single_file_kwargs: dict[str, Any] = dict(torch_dtype=torch.bfloat16)
        if args.device != "auto":
            single_file_kwargs["device"] = args.device
        pipe = AnimaPipeline.from_single_file(model, **single_file_kwargs)
    else:
        print(f"Loading from pretrained: {model}")
        pipe = AnimaPipeline.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            local_files_only=args.local_files_only,
        )
        # Resolve target device and move models
        device = args.device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        if device != "cpu":
            pipe.to(device)

    print(f"  execution_device = {pipe.execution_device}")

    # -----------------------------------------------------------------------
    # Memory optimisations (choose what fits your VRAM budget)
    # -----------------------------------------------------------------------

    if args.cpu_offload:
        # Move models to CPU between forward passes; keeps peak VRAM minimal.
        # Incompatible with pipe.to("cuda") — do not call both.
        pipe.enable_model_cpu_offload()
        print("  CPU offload enabled")

    if args.vae_slicing:
        # Decode one frame at a time in the VAE; trades VRAM for speed.
        pipe.enable_vae_slicing()
        print("  VAE slicing enabled")

    if args.vae_tiling:
        # Decode the image in spatial tiles; useful for very large outputs.
        pipe.enable_vae_tiling()
        print("  VAE tiling enabled")

    if args.vae_xformers:
        # xformers memory-efficient attention for the VAE (requires xformers).
        pipe.enable_vae_xformers_memory_efficient_attention()
        print("  VAE xformers enabled")

    # -----------------------------------------------------------------------
    # Scheduler / sampler configuration
    # -----------------------------------------------------------------------
    # set_sampling_config persists values into the scheduler's config dict so
    # they survive scheduler reconstruction with from_config().
    # Option A: mutate the existing scheduler in-place
    pipe.scheduler.set_sampling_config(
        sampler=args.sampler,
        sigma_schedule=args.sigma_schedule,
        eta=args.eta,
        s_noise=args.s_noise,
    )
    # Option B: replace with a freshly-built scheduler (useful when overriding
    # sigma-schedule options not exposed by set_sampling_config)
    pipe.scheduler = AnimaFlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        sampler=args.sampler,
        sigma_schedule=args.sigma_schedule,
        eta=args.eta,
        s_noise=args.s_noise,
    )
    print(
        f"  sampler={args.sampler}  sigma_schedule={args.sigma_schedule}"
        f"  eta={args.eta}  s_noise={args.s_noise}"
    )

    # -----------------------------------------------------------------------
    # LoRA
    # -----------------------------------------------------------------------
    if args.lora:
        print(f"  Loading LoRA: {args.lora}")
        pipe.load_lora_weights(args.lora, adapter_name="user_lora")
        # Optionally set adapter scale after loading:
        #   pipe.set_adapters(["user_lora"], adapter_weights=[0.8])

    return pipe


# ---------------------------------------------------------------------------
# Generation examples
# ---------------------------------------------------------------------------


def run_text2img(pipe: AnimaPipeline, output_dir: Path) -> Image.Image:
    """Basic text-to-image generation."""
    print("\n[1] Text-to-image")
    result = pipe(
        _PROMPT,
        negative_prompt=_NEG_PROMPT,
        width=_WIDTH,
        height=_HEIGHT,
        num_inference_steps=_STEPS,
        guidance_scale=_GUIDANCE,
        generator=_make_generator(_SEED),
        cfg_batch_mode="split",  # "split" (memory-efficient) or "concat" (faster)
        callback_on_step_end=_step_callback,
        output_type="pil",
        return_dict=True,
    )
    image = result.images[0]
    out_path = output_dir / "01_text2img.png"
    image.save(out_path)
    print(f"  Saved: {out_path}")
    return image


def run_encode_prompt_reuse(pipe: AnimaPipeline, output_dir: Path) -> None:
    """Pre-compute prompt embeddings once, reuse across two seed variations.

    encode_prompt returns (pos_cond, neg_cond) tensors.  Pass them to __call__
    via prompt_embeds / negative_prompt_embeds to skip tokenisation and
    text-encoding on each call — useful when sweeping seeds or steps.
    """
    print("\n[2] encode_prompt + prompt_embeds reuse (2 seeds)")

    pos_cond, neg_cond = pipe.encode_prompt(
        _PROMPT,
        negative_prompt=_NEG_PROMPT,
        num_images_per_prompt=1,
    )
    print(f"  pos_cond shape: {pos_cond.shape}  dtype: {pos_cond.dtype}")

    for seed in [_SEED, _SEED + 1]:
        result = pipe(
            _PROMPT,               # still required for metadata; ignored for encoding
            prompt_embeds=pos_cond,
            negative_prompt_embeds=neg_cond,
            width=_WIDTH,
            height=_HEIGHT,
            num_inference_steps=_STEPS,
            guidance_scale=_GUIDANCE,
            generator=_make_generator(seed),
            output_type="pil",
            return_dict=True,
        )
        out_path = output_dir / f"02_encode_prompt_seed{seed}.png"
        result.images[0].save(out_path)
        print(f"  Saved: {out_path}")


def run_batch(pipe: AnimaPipeline, output_dir: Path) -> None:
    """Generate multiple images from one prompt call (num_images_per_prompt)."""
    print("\n[3] Batch generation (num_images_per_prompt=2)")
    result = pipe(
        _PROMPT,
        negative_prompt=_NEG_PROMPT,
        width=_WIDTH // 2,
        height=_HEIGHT // 2,
        num_inference_steps=_STEPS,
        guidance_scale=_GUIDANCE,
        num_images_per_prompt=2,
        generator=_make_generator(_SEED),
        output_type="pil",
        return_dict=True,
    )
    for i, img in enumerate(result.images):
        out_path = output_dir / f"03_batch_{i}.png"
        img.save(out_path)
        print(f"  Saved: {out_path}")


def run_img2img(
    pipe: AnimaPipeline, init_image: Image.Image, output_dir: Path
) -> Image.Image:
    """Image-to-image generation: denoise from an initial image."""
    print("\n[4] Image-to-image  (strength=0.7)")
    result = pipe(
        _PROMPT,
        negative_prompt=_NEG_PROMPT,
        image=init_image,
        strength=0.7,   # 0.0 = keep init_image, 1.0 = full text-to-image
        width=_WIDTH,
        height=_HEIGHT,
        num_inference_steps=_STEPS,
        guidance_scale=_GUIDANCE,
        generator=_make_generator(_SEED),
        output_type="pil",
        return_dict=True,
    )
    image = result.images[0]
    out_path = output_dir / "04_img2img.png"
    image.save(out_path)
    print(f"  Saved: {out_path}")
    return image


def run_inpaint(
    pipe: AnimaPipeline, init_image: Image.Image, output_dir: Path
) -> None:
    """Inpainting: regenerate the masked region while preserving the rest."""
    print("\n[5] Inpainting")
    mask = _make_inpaint_mask(_WIDTH, _HEIGHT)
    mask.save(output_dir / "05_inpaint_mask.png")

    result = pipe(
        _PROMPT,
        negative_prompt=_NEG_PROMPT,
        image=init_image,
        mask_image=mask,  # white pixels = region to regenerate
        strength=1.0,
        width=_WIDTH,
        height=_HEIGHT,
        num_inference_steps=_STEPS,
        guidance_scale=_GUIDANCE,
        generator=_make_generator(_SEED),
        output_type="pil",
        return_dict=True,
    )
    out_path = output_dir / "05_inpaint.png"
    result.images[0].save(out_path)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full Anima pipeline demo")
    p.add_argument(
        "--model", default=_DEFAULT_MODEL,
        help="HF repo ID, local directory, or .safetensors path",
    )
    p.add_argument(
        "--device", default="auto",
        help="Target device: cuda / cpu / mps / auto (default)",
    )
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--cpu-offload", action="store_true",
                   help="Enable model CPU offload (lower VRAM, slower)")
    p.add_argument("--vae-slicing", action="store_true",
                   help="Enable VAE slicing")
    p.add_argument("--vae-tiling", action="store_true",
                   help="Enable VAE tiling")
    p.add_argument("--vae-xformers", action="store_true",
                   help="Enable VAE xformers attention")
    p.add_argument("--sampler", default="euler_a_rf",
                   choices=["flowmatch_euler", "euler", "euler_a_rf", "euler_ancestral_rf"])
    p.add_argument("--sigma-schedule", default="beta",
                   choices=["beta", "uniform", "simple", "normal"])
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--s-noise", type=float, default=1.0)
    p.add_argument("--lora", default=None,
                   help="Optional LoRA path or HF repo ID to load")
    p.add_argument("--skip-img2img", action="store_true",
                   help="Skip img2img and inpainting examples")
    p.add_argument("--output-dir", default="outputs/sample_full",
                   help="Directory to save generated images")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args)

    text2img_image = run_text2img(pipe, output_dir)
    run_encode_prompt_reuse(pipe, output_dir)
    run_batch(pipe, output_dir)

    if not args.skip_img2img:
        img2img_image = run_img2img(pipe, text2img_image, output_dir)
        run_inpaint(pipe, img2img_image, output_dir)

    print(f"\nDone. Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
