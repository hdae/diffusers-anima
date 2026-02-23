"""Regression baseline tests for Anima pipeline (1-girl reference case).

Loads the pipeline via ``AnimaPipeline.from_pretrained``.
By default uses the HuggingFace repo ``hdae/diffusers-anima-preview``.
Set ``ANIMA_PRETRAINED_MODEL_PATH`` to a local directory or alternative HF repo ID
to override (e.g. when validating a freshly converted model).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch

from diffusers_anima import AnimaFlowMatchEulerDiscreteScheduler, AnimaPipeline


pytestmark = [pytest.mark.integration]


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ASSET_DIR = _REPO_ROOT / "tests" / "integration" / "assets" / "regression_1girl"
_CASE_SETTINGS_PATH = _ASSET_DIR / "case_settings.json"

_BASELINE_TEXT2IMG_PATH = _ASSET_DIR / "baseline_1girl_text2img.png"
_BASELINE_IMG2IMG_PATH = _ASSET_DIR / "baseline_1girl_img2img.png"
_BASELINE_INPAINT_PATH = _ASSET_DIR / "baseline_1girl_inpaint.png"
_INPAINT_MASK_PATH = _ASSET_DIR / "input_1girl_inpaint_mask.png"

_OUTPUT_DIR = _REPO_ROOT / "outputs" / "integration_regression_1girl"
_REPORT_PATH = _OUTPUT_DIR / "report.json"

_DEFAULT_MODEL_PATH = "hdae/diffusers-anima-preview"


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _build_metrics(reference: Path, candidate: Path) -> dict[str, object]:
    reference_image = _load_rgb(reference)
    candidate_image = _load_rgb(candidate)
    if reference_image.shape != candidate_image.shape:
        raise ValueError(
            f"Image shape mismatch: reference={reference_image.shape}, candidate={candidate_image.shape}"
        )

    diff = candidate_image.astype(np.int16) - reference_image.astype(np.int16)
    abs_diff = np.abs(diff)
    mse = float((diff.astype(np.float64) ** 2).mean())
    rmse = math.sqrt(mse)

    return {
        "reference_path": str(reference),
        "candidate_path": str(candidate),
        "reference_sha256": _sha256(reference),
        "candidate_sha256": _sha256(candidate),
        "shape_hwc": list(reference_image.shape),
        "mae": float(abs_diff.mean()),
        "mse": mse,
        "rmse": rmse,
        "max_abs": int(abs_diff.max()),
        "nonzero_values": int(np.count_nonzero(abs_diff)),
        "nonzero_pixels": int(np.count_nonzero(np.any(abs_diff != 0, axis=2))),
        "psnr": float("inf") if mse == 0.0 else float(20.0 * math.log10(255.0 / rmse)),
    }


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _build_inpaint_mask(case_settings: dict[str, object]) -> Image.Image:
    common = case_settings["common"]
    width = int(common["width"])
    height = int(common["height"])

    mask = Image.new("L", (width, height), 0)
    mask_spec = case_settings["inpaint_mask"]
    draw = ImageDraw.Draw(mask)
    if mask_spec.get("shape") == "ellipse":
        draw.ellipse(
            (
                int(mask_spec["x0"]),
                int(mask_spec["y0"]),
                int(mask_spec["x1"]),
                int(mask_spec["y1"]),
            ),
            fill=255,
        )
    else:
        draw.rectangle(
            (
                int(mask_spec["x0"]),
                int(mask_spec["y0"]),
                int(mask_spec["x1"]),
                int(mask_spec["y1"]),
            ),
            fill=255,
        )
    return mask


def _load_pipeline() -> AnimaPipeline:
    pretrained_path = os.getenv("ANIMA_PRETRAINED_MODEL_PATH", "").strip() or _DEFAULT_MODEL_PATH

    device = os.getenv("ANIMA_DEVICE", "auto").strip()
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    pipe = AnimaPipeline.from_pretrained(
        pretrained_path,
        torch_dtype=torch.bfloat16,
        local_files_only=_env_flag("ANIMA_LOCAL_FILES_ONLY"),
    )
    if device != "cpu":
        pipe.to(device)
    if _env_flag("ANIMA_CPU_OFFLOAD"):
        pipe.enable_model_cpu_offload()
    if _env_flag("ANIMA_VAE_SLICING"):
        pipe.enable_vae_slicing()
    if _env_flag("ANIMA_VAE_TILING"):
        pipe.enable_vae_tiling()
    if _env_flag("ANIMA_VAE_XFORMERS"):
        pipe.enable_vae_xformers_memory_efficient_attention()
    return pipe


def _generate_from_case(
    *,
    pipe: AnimaPipeline,
    common: dict[str, object],
    case: dict[str, object],
    init_image: Image.Image | None = None,
    mask_image: Image.Image | None = None,
) -> Image.Image:
    generator_seed = int(common["generator_seed"])
    generator = torch.Generator(device="cpu").manual_seed(generator_seed)

    output = pipe(
        str(case["prompt"]),
        negative_prompt=str(common["negative_prompt"]),
        image=init_image,
        mask_image=mask_image,
        strength=float(case.get("strength", 1.0)),
        width=int(common["width"]),
        height=int(common["height"]),
        num_inference_steps=int(common["num_inference_steps"]),
        guidance_scale=float(common["guidance_scale"]),
        generator=generator,
        cfg_batch_mode=str(common["cfg_batch_mode"]),
        output_type="pil",
        return_dict=True,
    )
    return output.images[0]


def _configure_case_scheduler(
    *,
    pipe: AnimaPipeline,
    common: dict[str, object],
) -> None:
    scheduler = AnimaFlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        sampler=str(common["sampler"]),
        sigma_schedule=str(common["sigma_schedule"]),
        beta_alpha=float(common["beta_alpha"]),
        beta_beta=float(common["beta_beta"]),
        eta=float(common["eta"]),
        s_noise=float(common["s_noise"]),
    )
    pipe.scheduler = scheduler


def _assert_baselines_exist() -> None:
    required = [
        _BASELINE_TEXT2IMG_PATH,
        _BASELINE_IMG2IMG_PATH,
        _BASELINE_INPAINT_PATH,
        _INPAINT_MASK_PATH,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            "Regression baselines are missing. "
            "Generate them with ANIMA_UPDATE_BASELINE=1."
            f" Missing: {joined}"
        )


def test_regression_1girl_baselines() -> None:
    update_baseline = _env_flag("ANIMA_UPDATE_BASELINE")
    max_abs_threshold = int(os.getenv("ANIMA_MAX_ABS_THRESHOLD", "0"))
    nonzero_pixels_threshold = int(os.getenv("ANIMA_NONZERO_PIXELS_THRESHOLD", "0"))

    case_settings = _load_json(_CASE_SETTINGS_PATH)
    common = case_settings["common"]

    if not update_baseline:
        _assert_baselines_exist()

    pipe = _load_pipeline()
    _configure_case_scheduler(pipe=pipe, common=common)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mask_image = _build_inpaint_mask(case_settings)
    if update_baseline:
        mask_image.save(_INPAINT_MASK_PATH)
    else:
        mask_image = Image.open(_INPAINT_MASK_PATH).convert("L")

    results: dict[str, dict[str, object]] = {}

    text2img_image = _generate_from_case(
        pipe=pipe,
        common=common,
        case=case_settings["text2img"],
    )
    text2img_candidate_path = _OUTPUT_DIR / "current_1girl_text2img.png"
    text2img_image.save(text2img_candidate_path)
    if update_baseline:
        text2img_image.save(_BASELINE_TEXT2IMG_PATH)
        results["text2img"] = {"updated_baseline": str(_BASELINE_TEXT2IMG_PATH)}
    else:
        metrics = _build_metrics(_BASELINE_TEXT2IMG_PATH, text2img_candidate_path)
        results["text2img"] = metrics
        assert metrics["max_abs"] <= max_abs_threshold
        assert metrics["nonzero_pixels"] <= nonzero_pixels_threshold

    if update_baseline:
        init_image = text2img_image
    else:
        init_image = Image.open(_BASELINE_TEXT2IMG_PATH).convert("RGB")

    img2img_image = _generate_from_case(
        pipe=pipe,
        common=common,
        case=case_settings["img2img"],
        init_image=init_image,
    )
    img2img_candidate_path = _OUTPUT_DIR / "current_1girl_img2img.png"
    img2img_image.save(img2img_candidate_path)
    if update_baseline:
        img2img_image.save(_BASELINE_IMG2IMG_PATH)
        results["img2img"] = {"updated_baseline": str(_BASELINE_IMG2IMG_PATH)}
    else:
        metrics = _build_metrics(_BASELINE_IMG2IMG_PATH, img2img_candidate_path)
        results["img2img"] = metrics
        assert metrics["max_abs"] <= max_abs_threshold
        assert metrics["nonzero_pixels"] <= nonzero_pixels_threshold

    inpaint_image = _generate_from_case(
        pipe=pipe,
        common=common,
        case=case_settings["inpaint"],
        init_image=init_image,
        mask_image=mask_image,
    )
    inpaint_candidate_path = _OUTPUT_DIR / "current_1girl_inpaint.png"
    inpaint_image.save(inpaint_candidate_path)
    if update_baseline:
        inpaint_image.save(_BASELINE_INPAINT_PATH)
        results["inpaint"] = {"updated_baseline": str(_BASELINE_INPAINT_PATH)}
    else:
        metrics = _build_metrics(_BASELINE_INPAINT_PATH, inpaint_candidate_path)
        results["inpaint"] = metrics
        assert metrics["max_abs"] <= max_abs_threshold
        assert metrics["nonzero_pixels"] <= nonzero_pixels_threshold

    report = {
        "update_baseline": update_baseline,
        "thresholds": {
            "max_abs": max_abs_threshold,
            "nonzero_pixels": nonzero_pixels_threshold,
        },
        "results": results,
    }
    _write_json(_REPORT_PATH, report)
