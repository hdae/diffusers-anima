from __future__ import annotations

import pytest
import torch

from diffusers_anima.pipelines.anima.generator_utils import (
    _normalize_generator,
    _resolve_noise_runtime,
)
from diffusers_anima.pipelines.anima.prompt_utils import (
    _resolve_prompt_batches,
)
from diffusers_anima.pipelines.anima.sampling import (
    randn_like as _randn_like,
    randn_tensor as _randn_tensor,
)


# ---------------------------------------------------------------------------
# _normalize_generator
# ---------------------------------------------------------------------------


def test_normalize_generator_accepts_single_generator() -> None:
    generator = torch.Generator(device="cpu").manual_seed(123)

    resolved = _normalize_generator(generator, batch_size=1)

    assert resolved is generator


def test_normalize_generator_accepts_single_item_list() -> None:
    generator = torch.Generator(device="cpu").manual_seed(123)

    resolved = _normalize_generator([generator], batch_size=1)

    assert isinstance(resolved, list)
    assert len(resolved) == 1
    assert resolved[0] is generator


def test_normalize_generator_rejects_batch_size_mismatch() -> None:
    generators = [
        torch.Generator(device="cpu").manual_seed(1),
        torch.Generator(device="cpu").manual_seed(2),
    ]

    with pytest.raises(ValueError, match="batch size"):
        _normalize_generator(generators, batch_size=1)


def test_normalize_generator_rejects_invalid_type() -> None:
    with pytest.raises(ValueError, match="torch.Generator"):
        _normalize_generator(generator=42, batch_size=1)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _resolve_noise_runtime
# ---------------------------------------------------------------------------


def test_resolve_noise_runtime_prioritizes_generator() -> None:
    provided = torch.Generator(device="cpu").manual_seed(999)

    init_generator, step_generator, noise_device, noise_dtype = _resolve_noise_runtime(
        execution_device="cpu",
        generator=provided,
        batch_size=1,
    )

    assert init_generator is provided
    assert step_generator is provided
    assert noise_device == "cpu"
    assert noise_dtype == torch.float32


def test_resolve_noise_runtime_without_generator_returns_default_runtime() -> None:
    init_generator, step_generator, noise_device, noise_dtype = _resolve_noise_runtime(
        execution_device="cpu",
        generator=None,
        batch_size=1,
    )

    assert init_generator is None
    assert step_generator is None
    assert noise_device == "cpu"
    assert noise_dtype == torch.float32


def test_resolve_noise_runtime_accepts_generator_list_for_batch() -> None:
    generators = [
        torch.Generator(device="cpu").manual_seed(111),
        torch.Generator(device="cpu").manual_seed(222),
    ]

    init_generator, step_generator, noise_device, noise_dtype = _resolve_noise_runtime(
        execution_device="cpu",
        generator=generators,
        batch_size=2,
    )

    assert isinstance(init_generator, list)
    assert isinstance(step_generator, list)
    assert init_generator == generators
    assert step_generator == generators
    assert noise_device == "cpu"
    assert noise_dtype == torch.float32


# ---------------------------------------------------------------------------
# randn_tensor
# ---------------------------------------------------------------------------


def test_randn_tensor_supports_generator_list() -> None:
    generators = [
        torch.Generator(device="cpu").manual_seed(1),
        torch.Generator(device="cpu").manual_seed(2),
    ]

    sample = _randn_tensor(
        (2, 4),
        device="cpu",
        dtype=torch.float32,
        generator=generators,
    )

    assert tuple(sample.shape) == (2, 4)


def test_randn_tensor_deterministic_with_seed() -> None:
    """Same seed must produce identical noise."""
    g1 = torch.Generator(device="cpu").manual_seed(42)
    g2 = torch.Generator(device="cpu").manual_seed(42)

    a = _randn_tensor((4, 16), device="cpu", dtype=torch.float32, generator=g1)
    b = _randn_tensor((4, 16), device="cpu", dtype=torch.float32, generator=g2)

    assert torch.equal(a, b)


def test_randn_tensor_different_seeds_produce_different_noise() -> None:
    g1 = torch.Generator(device="cpu").manual_seed(42)
    g2 = torch.Generator(device="cpu").manual_seed(99)

    a = _randn_tensor((4, 16), device="cpu", dtype=torch.float32, generator=g1)
    b = _randn_tensor((4, 16), device="cpu", dtype=torch.float32, generator=g2)

    assert not torch.equal(a, b)


def test_randn_tensor_none_generator() -> None:
    sample = _randn_tensor(
        (2, 4), device="cpu", dtype=torch.float32, generator=None
    )
    assert tuple(sample.shape) == (2, 4)


def test_randn_tensor_generator_list_rejects_size_mismatch() -> None:
    generators = [
        torch.Generator(device="cpu").manual_seed(1),
        torch.Generator(device="cpu").manual_seed(2),
    ]

    with pytest.raises(ValueError, match="batch size"):
        _randn_tensor(
            (3, 4), device="cpu", dtype=torch.float32, generator=generators,
        )


# ---------------------------------------------------------------------------
# randn_like
# ---------------------------------------------------------------------------


def test_randn_like_matches_shape_and_dtype() -> None:
    sample = torch.randn(2, 16, 8, 8, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(42)

    noise = _randn_like(sample, generator=generator)

    assert noise.shape == sample.shape
    assert noise.dtype == sample.dtype
    assert noise.device == sample.device


def test_randn_like_none_generator() -> None:
    sample = torch.randn(1, 4)
    noise = _randn_like(sample, generator=None)
    assert noise.shape == sample.shape


# ---------------------------------------------------------------------------
# Generator + prompt batch integration
# ---------------------------------------------------------------------------


def test_generator_list_matches_expanded_batch_size_for_multi_image_prompts() -> None:
    prompts, _ = _resolve_prompt_batches(
        prompt=["a", "b"],
        negative_prompt=None,
        num_images_per_prompt=2,
    )
    generators = [
        torch.Generator(device="cpu").manual_seed(1),
        torch.Generator(device="cpu").manual_seed(2),
        torch.Generator(device="cpu").manual_seed(3),
        torch.Generator(device="cpu").manual_seed(4),
    ]

    resolved = _normalize_generator(generators, batch_size=len(prompts))

    assert isinstance(resolved, list)
    assert len(resolved) == 4


def test_generator_list_rejects_expanded_batch_size_mismatch() -> None:
    """A 2-element generator list is rejected when batch_size=4 (2 prompts x 2 images)."""
    prompts, _ = _resolve_prompt_batches(
        prompt=["a", "b"],
        negative_prompt=None,
        num_images_per_prompt=2,
    )
    generators = [
        torch.Generator(device="cpu").manual_seed(1),
        torch.Generator(device="cpu").manual_seed(2),
    ]

    with pytest.raises(ValueError, match="batch size"):
        _normalize_generator(generators, batch_size=len(prompts))
