"""Tests for image preprocessing and utility functions."""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from diffusers_anima.pipelines.anima.image_processing import (
    _ensure_finite,
    _normalize_tensor_to_unit_interval,
    _reshape_image_tensor_to_bchw,
    align_tensor_batch_size,
    latent_hw,
    prepare_init_image_tensor,
    prepare_inpaint_mask_tensor,
)


# ---------------------------------------------------------------------------
# prepare_init_image_tensor
# ---------------------------------------------------------------------------


def test_prepare_init_image_tensor_accepts_mixed_list_inputs() -> None:
    pil_image = Image.new("RGB", (16, 16), color=(255, 0, 0))
    np_image = np.zeros((16, 16, 3), dtype=np.uint8)
    tensor_image = torch.ones((3, 16, 16), dtype=torch.float32)

    prepared = prepare_init_image_tensor(
        [pil_image, np_image, tensor_image],
        width=16,
        height=16,
    )

    assert tuple(prepared.shape) == (3, 3, 16, 16)
    assert prepared.dtype == torch.float32
    assert float(prepared.min().item()) >= -1.0
    assert float(prepared.max().item()) <= 1.0


def test_prepare_init_image_tensor_single_pil() -> None:
    pil_image = Image.new("RGB", (32, 32), color=(128, 64, 0))
    prepared = prepare_init_image_tensor(pil_image, width=16, height=16)
    assert tuple(prepared.shape) == (1, 3, 16, 16)
    assert prepared.dtype == torch.float32
    assert float(prepared.min().item()) >= -1.0
    assert float(prepared.max().item()) <= 1.0


def test_prepare_init_image_tensor_single_numpy() -> None:
    np_image = np.full((16, 16, 3), 128, dtype=np.uint8)
    prepared = prepare_init_image_tensor(np_image, width=16, height=16)
    assert tuple(prepared.shape) == (1, 3, 16, 16)


def test_prepare_init_image_tensor_single_tensor_chw() -> None:
    tensor = torch.rand((3, 32, 32))
    prepared = prepare_init_image_tensor(tensor, width=16, height=16)
    assert tuple(prepared.shape) == (1, 3, 16, 16)


def test_prepare_init_image_tensor_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        prepare_init_image_tensor([], width=16, height=16)


def test_prepare_init_image_tensor_rejects_nested_list() -> None:
    pil = Image.new("RGB", (16, 16))
    with pytest.raises(ValueError, match="Nested list"):
        prepare_init_image_tensor([[pil]], width=16, height=16)  # type: ignore[list-item]


def test_prepare_init_image_tensor_resizes_to_target() -> None:
    pil_image = Image.new("RGB", (64, 64), color=(255, 255, 255))
    prepared = prepare_init_image_tensor(pil_image, width=32, height=16)
    assert tuple(prepared.shape) == (1, 3, 16, 32)


# ---------------------------------------------------------------------------
# prepare_inpaint_mask_tensor
# ---------------------------------------------------------------------------


def test_prepare_inpaint_mask_tensor_accepts_mixed_list_inputs() -> None:
    pil_mask = Image.new("L", (16, 16), color=255)
    np_mask = np.zeros((16, 16), dtype=np.uint8)
    tensor_mask = torch.full((1, 16, 16), 0.5, dtype=torch.float32)

    prepared = prepare_inpaint_mask_tensor(
        [pil_mask, np_mask, tensor_mask],
        width=16,
        height=16,
    )

    assert tuple(prepared.shape) == (3, 1, 16, 16)
    assert prepared.dtype == torch.float32
    assert float(prepared.min().item()) >= 0.0
    assert float(prepared.max().item()) <= 1.0


def test_prepare_inpaint_mask_tensor_single_pil() -> None:
    pil_mask = Image.new("L", (32, 32), color=128)
    prepared = prepare_inpaint_mask_tensor(pil_mask, width=16, height=16)
    assert tuple(prepared.shape) == (1, 1, 16, 16)


def test_prepare_inpaint_mask_tensor_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        prepare_inpaint_mask_tensor([], width=16, height=16)


def test_prepare_inpaint_mask_tensor_3channel_rgb_reduced_to_single() -> None:
    """3-channel tensor masks are averaged to single channel."""
    tensor_mask = torch.rand((3, 16, 16))
    prepared = prepare_inpaint_mask_tensor(tensor_mask, width=16, height=16)
    assert prepared.shape[1] == 1


# ---------------------------------------------------------------------------
# _reshape_image_tensor_to_bchw
# ---------------------------------------------------------------------------


def test_reshape_2d_tensor_to_bchw() -> None:
    tensor = torch.rand((16, 16))
    result = _reshape_image_tensor_to_bchw(tensor, input_label="test")
    assert tuple(result.shape) == (1, 1, 16, 16)


def test_reshape_hwc_tensor_to_bchw() -> None:
    tensor = torch.rand((16, 16, 3))
    result = _reshape_image_tensor_to_bchw(tensor, input_label="test")
    assert tuple(result.shape) == (1, 3, 16, 16)


def test_reshape_chw_tensor_to_bchw() -> None:
    tensor = torch.rand((3, 16, 16))
    result = _reshape_image_tensor_to_bchw(tensor, input_label="test")
    assert tuple(result.shape) == (1, 3, 16, 16)


def test_reshape_rejects_5d_tensor() -> None:
    tensor = torch.rand((1, 1, 1, 16, 16))
    with pytest.raises(ValueError, match="2D/3D/4D"):
        _reshape_image_tensor_to_bchw(tensor, input_label="test")


def test_reshape_rejects_invalid_channel_count() -> None:
    tensor = torch.rand((5, 16, 16))  # 5 channels
    with pytest.raises(ValueError, match="channel size 1 or 3"):
        _reshape_image_tensor_to_bchw(tensor, input_label="test")


# ---------------------------------------------------------------------------
# _normalize_tensor_to_unit_interval
# ---------------------------------------------------------------------------


def test_normalize_uint8_range() -> None:
    tensor = torch.tensor([0.0, 128.0, 255.0])
    result = _normalize_tensor_to_unit_interval(tensor, input_label="test")
    assert float(result[0].item()) == pytest.approx(0.0)
    assert float(result[2].item()) == pytest.approx(1.0)


def test_normalize_negative_one_to_one_range() -> None:
    tensor = torch.tensor([-1.0, 0.0, 1.0])
    result = _normalize_tensor_to_unit_interval(tensor, input_label="test")
    assert float(result[0].item()) == pytest.approx(0.0)
    assert float(result[2].item()) == pytest.approx(1.0)


def test_normalize_already_unit_interval() -> None:
    tensor = torch.tensor([0.0, 0.5, 1.0])
    result = _normalize_tensor_to_unit_interval(tensor, input_label="test")
    assert torch.allclose(result, tensor)


def test_normalize_rejects_unsupported_range() -> None:
    tensor = torch.tensor([-2.0, 0.0, 500.0])
    with pytest.raises(ValueError, match="unsupported"):
        _normalize_tensor_to_unit_interval(tensor, input_label="test")


# ---------------------------------------------------------------------------
# _ensure_finite
# ---------------------------------------------------------------------------


def test_ensure_finite_passes_for_normal_tensor() -> None:
    tensor = torch.randn(4, 4)
    _ensure_finite(tensor, name="test", runtime_dtype=torch.float32)


def test_ensure_finite_raises_on_nan() -> None:
    tensor = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        _ensure_finite(tensor, name="test", runtime_dtype=torch.float32)


def test_ensure_finite_raises_on_inf() -> None:
    tensor = torch.tensor([1.0, float("inf"), 3.0])
    with pytest.raises(RuntimeError, match="NaN/Inf"):
        _ensure_finite(tensor, name="test", runtime_dtype=torch.float32)


def test_ensure_finite_fp16_includes_dtype_hint() -> None:
    tensor = torch.tensor([float("nan")])
    with pytest.raises(RuntimeError, match="float16 is unstable"):
        _ensure_finite(tensor, name="test", runtime_dtype=torch.float16)


# ---------------------------------------------------------------------------
# latent_hw
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("height", "width", "expected_h", "expected_w"),
    [
        (1024, 1024, 1024, 1024),
        (1023, 1023, 1008, 1008),  # floor to nearest step (8*2=16)
        (16, 16, 16, 16),
    ],
)
def test_latent_hw_aligns_dimensions(
    height: int, width: int, expected_h: int, expected_w: int,
) -> None:
    h, w, lh, lw = latent_hw(height, width)
    assert h == expected_h
    assert w == expected_w
    assert lh == expected_h // 8
    assert lw == expected_w // 8


# ---------------------------------------------------------------------------
# align_tensor_batch_size
# ---------------------------------------------------------------------------


def test_align_tensor_batch_size_exact_match() -> None:
    tensor = torch.zeros((4, 3, 2, 2))
    result = align_tensor_batch_size(tensor, target_batch_size=4, input_name="test")
    assert result is tensor  # identity when already correct


def test_align_tensor_batch_size_repeat_interleave() -> None:
    tensor = torch.zeros((2, 3, 2, 2))
    result = align_tensor_batch_size(tensor, target_batch_size=4, input_name="test")
    assert tuple(result.shape) == (4, 3, 2, 2)
