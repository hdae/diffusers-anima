"""Tests for strength-based step trimming."""

from __future__ import annotations

import pytest
import torch

from diffusers_anima.pipelines.anima.strength_utils import (
    _resolve_strength_start_step,
    _trim_flowmatch_timesteps_by_strength,
    _trim_sigmas_by_strength,
)


class _DummyScheduler:
    def __init__(self, *, order: int = 1):
        self.order = order
        self.timesteps = torch.tensor([], dtype=torch.float32)
        self.begin_index: int | None = None

    def set_timesteps(self, num_inference_steps: int, device: str) -> None:
        self.timesteps = torch.arange(
            num_inference_steps - 1,
            -1,
            -1,
            dtype=torch.float32,
            device=device,
        )

    def set_begin_index(self, begin_index: int) -> None:
        self.begin_index = begin_index


class _DummyPipe:
    def __init__(self, *, order: int = 1):
        self.execution_device = "cpu"
        self.scheduler = _DummyScheduler(order=order)


# ---------------------------------------------------------------------------
# _resolve_strength_start_step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("total_steps", "strength", "expected"),
    [
        (20, 1.0, 0),
        (20, 0.5, 10),
        (20, 0.1, 18),
        (20, 0.0, 20),
        (20, 0.25, 15),
        (20, 0.75, 5),
        (10, 1.0, 0),
        (10, 0.5, 5),
        (1, 1.0, 0),
        (1, 0.5, 0),
    ],
)
def test_resolve_strength_start_step_parametrized(
    total_steps: int, strength: float, expected: int,
) -> None:
    assert _resolve_strength_start_step(total_steps=total_steps, strength=strength) == expected


def test_resolve_strength_start_step_rejects_zero_total_steps() -> None:
    with pytest.raises(ValueError, match="total_steps must be >= 1"):
        _resolve_strength_start_step(total_steps=0, strength=1.0)


# ---------------------------------------------------------------------------
# _trim_sigmas_by_strength
# ---------------------------------------------------------------------------


def test_trim_sigmas_by_strength_trims_prefix() -> None:
    sigmas = torch.tensor([9.0, 7.0, 5.0, 0.0], dtype=torch.float32)

    trimmed = _trim_sigmas_by_strength(sigmas=sigmas, strength=0.5)

    assert torch.equal(trimmed, torch.tensor([7.0, 5.0, 0.0], dtype=torch.float32))


def test_trim_sigmas_by_strength_noop_when_strength_is_one() -> None:
    sigmas = torch.tensor([3.0, 2.0, 1.0, 0.0], dtype=torch.float32)

    trimmed = _trim_sigmas_by_strength(sigmas=sigmas, strength=1.0)

    assert torch.equal(trimmed, sigmas)


def test_trim_sigmas_by_strength_raises_when_too_few_steps_remain() -> None:
    """strength=0.0 trims all steps, leaving only the trailing zero."""
    sigmas = torch.tensor([3.0, 1.0, 0.0], dtype=torch.float32)  # 2 steps

    with pytest.raises(ValueError, match="fewer than 1 denoising step"):
        _trim_sigmas_by_strength(sigmas=sigmas, strength=0.0)


@pytest.mark.parametrize("strength", [0.3, 0.5, 0.7, 0.9, 1.0])
def test_trim_sigmas_by_strength_always_ends_with_zero(strength: float) -> None:
    sigmas = torch.tensor([9.0, 7.0, 5.0, 3.0, 1.0, 0.0], dtype=torch.float32)

    trimmed = _trim_sigmas_by_strength(sigmas=sigmas, strength=strength)

    assert trimmed[-1].item() == 0.0


# ---------------------------------------------------------------------------
# _trim_flowmatch_timesteps_by_strength
# ---------------------------------------------------------------------------


def test_trim_flowmatch_timesteps_by_strength_sets_begin_index() -> None:
    pipe = _DummyPipe(order=1)

    trimmed = _trim_flowmatch_timesteps_by_strength(
        pipe,
        num_inference_steps=10,
        strength=0.5,
    )

    assert torch.equal(
        trimmed, torch.tensor([4.0, 3.0, 2.0, 1.0, 0.0], dtype=torch.float32)
    )
    assert pipe.scheduler.begin_index == 5


def test_trim_flowmatch_timesteps_by_strength_noop_when_strength_is_one() -> None:
    pipe = _DummyPipe(order=1)

    trimmed = _trim_flowmatch_timesteps_by_strength(
        pipe,
        num_inference_steps=6,
        strength=1.0,
    )

    assert torch.equal(
        trimmed, torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], dtype=torch.float32)
    )


@pytest.mark.parametrize("strength", [0.3, 0.5, 0.7, 1.0])
def test_trim_flowmatch_timesteps_returns_fewer_steps_with_lower_strength(
    strength: float,
) -> None:
    pipe = _DummyPipe(order=1)

    trimmed = _trim_flowmatch_timesteps_by_strength(
        pipe,
        num_inference_steps=20,
        strength=strength,
    )

    full_pipe = _DummyPipe(order=1)
    full_trimmed = _trim_flowmatch_timesteps_by_strength(
        full_pipe,
        num_inference_steps=20,
        strength=1.0,
    )

    assert len(trimmed) <= len(full_trimmed)
