from __future__ import annotations

import pytest
import torch

from diffusers_anima.pipelines.anima.image_processing import (
    align_tensor_batch_size as _align_tensor_batch_size,
)
from diffusers_anima.pipelines.anima.prompt_utils import (
    _resolve_prompt_batches,
)


# ---------------------------------------------------------------------------
# _resolve_prompt_batches
# ---------------------------------------------------------------------------


def test_resolve_prompt_batches_expands_num_images_per_prompt() -> None:
    prompts, negatives = _resolve_prompt_batches(
        prompt=["a", "b"],
        negative_prompt=["na", "nb"],
        num_images_per_prompt=2,
    )

    assert prompts == ["a", "a", "b", "b"]
    assert negatives == ["na", "na", "nb", "nb"]


def test_resolve_prompt_batches_none_negative_prompt_fills_empty_strings() -> None:
    prompts, negatives = _resolve_prompt_batches(
        prompt=["a", "b"],
        negative_prompt=None,
        num_images_per_prompt=1,
    )

    assert prompts == ["a", "b"]
    assert negatives == ["", ""]


def test_resolve_prompt_batches_string_prompt_wraps_to_list() -> None:
    prompts, negatives = _resolve_prompt_batches(
        prompt="single prompt",
        negative_prompt=None,
        num_images_per_prompt=1,
    )

    assert prompts == ["single prompt"]
    assert negatives == [""]


def test_resolve_prompt_batches_string_negative_prompt_broadcasts() -> None:
    prompts, negatives = _resolve_prompt_batches(
        prompt=["a", "b"],
        negative_prompt="bad quality",
        num_images_per_prompt=1,
    )

    assert prompts == ["a", "b"]
    assert negatives == ["bad quality", "bad quality"]


def test_resolve_prompt_batches_num_images_per_prompt_one_returns_unchanged() -> None:
    prompts, negatives = _resolve_prompt_batches(
        prompt=["a", "b"],
        negative_prompt=["na", "nb"],
        num_images_per_prompt=1,
    )

    assert prompts == ["a", "b"]
    assert negatives == ["na", "nb"]


def test_resolve_prompt_batches_rejects_negative_prompt_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length must match"):
        _resolve_prompt_batches(
            prompt=["a", "b"],
            negative_prompt=["na"],
            num_images_per_prompt=1,
        )


def test_resolve_prompt_batches_rejects_zero_num_images_per_prompt() -> None:
    with pytest.raises(ValueError, match="num_images_per_prompt"):
        _resolve_prompt_batches(
            prompt=["a"],
            negative_prompt=None,
            num_images_per_prompt=0,
        )


# ---------------------------------------------------------------------------
# align_tensor_batch_size
# ---------------------------------------------------------------------------


def test_align_tensor_batch_size_repeats_single_batch() -> None:
    tensor = torch.zeros((1, 3, 2, 2))

    aligned = _align_tensor_batch_size(
        tensor,
        target_batch_size=4,
        input_name="image",
    )

    assert tuple(aligned.shape) == (4, 3, 2, 2)


def test_align_tensor_batch_size_exact_match_returns_identity() -> None:
    tensor = torch.zeros((4, 3, 2, 2))

    aligned = _align_tensor_batch_size(
        tensor,
        target_batch_size=4,
        input_name="image",
    )

    assert aligned is tensor


def test_align_tensor_batch_size_repeat_interleave() -> None:
    """Batch size 2 -> target 4 uses repeat_interleave."""
    tensor = torch.tensor([[1.0], [2.0]])

    aligned = _align_tensor_batch_size(
        tensor,
        target_batch_size=4,
        input_name="image",
    )

    assert tuple(aligned.shape) == (4, 1)
    assert aligned[0].item() == 1.0
    assert aligned[1].item() == 1.0
    assert aligned[2].item() == 2.0
    assert aligned[3].item() == 2.0


def test_align_tensor_batch_size_rejects_incompatible_batch() -> None:
    tensor = torch.zeros((3, 3, 2, 2))

    with pytest.raises(ValueError, match="incompatible"):
        _align_tensor_batch_size(
            tensor,
            target_batch_size=4,
            input_name="image",
        )
