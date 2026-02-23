"""Tests for sigma schedule builders."""

from __future__ import annotations

import pytest
import torch

from diffusers_anima.pipelines.anima.sigma_schedules import (
    _time_snr_shift,
    build_beta_sigmas,
    build_normal_sigmas,
    build_sampling_sigmas,
    build_simple_sigmas,
)
from diffusers_anima.schedulers import AnimaFlowMatchEulerDiscreteScheduler


# ---------------------------------------------------------------------------
# _time_snr_shift
# ---------------------------------------------------------------------------


def test_time_snr_shift_identity_when_alpha_is_one() -> None:
    t = torch.linspace(0.0, 1.0, 10)
    result = _time_snr_shift(1.0, t)
    assert torch.allclose(result, t)


def test_time_snr_shift_applies_shift_when_alpha_is_not_one() -> None:
    t = torch.tensor([0.0, 0.5, 1.0])
    result = _time_snr_shift(3.0, t)

    # At t=0: 0/(0+1) = 0
    assert result[0].item() == pytest.approx(0.0, abs=1e-6)
    # At t=1: 3/(3-1+1)=3/3=1
    assert result[2].item() == pytest.approx(1.0, abs=1e-6)
    # At t=0.5: 1.5/(0.5*2+1)=1.5/2=0.75
    assert result[1].item() == pytest.approx(0.75, abs=1e-5)


# ---------------------------------------------------------------------------
# build_simple_sigmas
# ---------------------------------------------------------------------------


def test_build_simple_sigmas_output_length() -> None:
    base_sigmas = torch.linspace(0.0, 1.0, 1000)
    result = build_simple_sigmas(base_sigmas, steps=20)
    # steps + 1 (trailing zero)
    assert len(result) == 21


def test_build_simple_sigmas_ends_with_zero() -> None:
    base_sigmas = torch.linspace(0.0, 1.0, 100)
    result = build_simple_sigmas(base_sigmas, steps=10)
    assert result[-1].item() == 0.0


def test_build_simple_sigmas_is_monotonically_decreasing() -> None:
    base_sigmas = torch.linspace(0.01, 1.0, 1000)
    result = build_simple_sigmas(base_sigmas, steps=20)
    # Excluding the trailing zero, should be monotonically decreasing
    for i in range(len(result) - 2):
        assert result[i].item() >= result[i + 1].item(), (
            f"Not monotonically decreasing at index {i}: {result[i].item()} < {result[i + 1].item()}"
        )


def test_build_simple_sigmas_rejects_zero_steps() -> None:
    base_sigmas = torch.linspace(0.0, 1.0, 100)
    with pytest.raises(ValueError, match="steps must be >= 1"):
        build_simple_sigmas(base_sigmas, steps=0)


def test_build_simple_sigmas_single_step() -> None:
    base_sigmas = torch.linspace(0.0, 1.0, 100)
    result = build_simple_sigmas(base_sigmas, steps=1)
    assert len(result) == 2
    assert result[-1].item() == 0.0


# ---------------------------------------------------------------------------
# build_beta_sigmas
# ---------------------------------------------------------------------------


def test_build_beta_sigmas_output_ends_with_zero() -> None:
    result = build_beta_sigmas(
        num_inference_steps=20,
        num_train_timesteps=1000,
        shift=3.0,
        beta_alpha=0.6,
        beta_beta=0.6,
        device="cpu",
    )
    assert result[-1].item() == 0.0


def test_build_beta_sigmas_deduplicates_indices() -> None:
    """Beta distribution with extreme parameters may map multiple steps to the same index."""
    result = build_beta_sigmas(
        num_inference_steps=50,
        num_train_timesteps=1000,
        shift=3.0,
        beta_alpha=0.6,
        beta_beta=0.6,
        device="cpu",
    )
    # Deduplicated length may differ from num_inference_steps + 1
    # but must have at least 2 entries (one sigma + trailing zero)
    assert len(result) >= 2
    assert result[-1].item() == 0.0


def test_build_beta_sigmas_respects_step_count_upper_bound() -> None:
    result = build_beta_sigmas(
        num_inference_steps=10,
        num_train_timesteps=1000,
        shift=3.0,
        beta_alpha=0.6,
        beta_beta=0.6,
        device="cpu",
    )
    # After deduplication + trailing zero, must be <= steps + 1
    assert len(result) <= 11


@pytest.mark.parametrize(
    ("alpha", "beta_param"),
    [
        (0.6, 0.6),  # Forge default
        (0.5, 0.5),  # Symmetric U-shape
        (2.0, 2.0),  # Bell-shaped
        (1.0, 1.0),  # Uniform
    ],
)
def test_build_beta_sigmas_various_parameters(alpha: float, beta_param: float) -> None:
    result = build_beta_sigmas(
        num_inference_steps=20,
        num_train_timesteps=1000,
        shift=3.0,
        beta_alpha=alpha,
        beta_beta=beta_param,
        device="cpu",
    )
    assert result[-1].item() == 0.0
    assert len(result) >= 2
    assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# build_normal_sigmas
# ---------------------------------------------------------------------------


def test_build_normal_sigmas_output_ends_with_zero() -> None:
    result = build_normal_sigmas(
        num_inference_steps=20,
        num_train_timesteps=1000,
        shift=3.0,
        device="cpu",
    )
    assert result[-1].item() == pytest.approx(0.0, abs=1e-5)


def test_build_normal_sigmas_first_value_is_largest() -> None:
    result = build_normal_sigmas(
        num_inference_steps=20,
        num_train_timesteps=1000,
        shift=3.0,
        device="cpu",
    )
    assert result[0].item() >= result[1].item()


def test_build_normal_sigmas_all_finite() -> None:
    result = build_normal_sigmas(
        num_inference_steps=30,
        num_train_timesteps=1000,
        shift=1.0,
        device="cpu",
    )
    assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# build_sampling_sigmas (dispatcher)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _scheduler() -> AnimaFlowMatchEulerDiscreteScheduler:
    from diffusers import FlowMatchEulerDiscreteScheduler
    base = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False,
    )
    return AnimaFlowMatchEulerDiscreteScheduler.from_config(base.config)


@pytest.mark.parametrize("sigma_schedule", ["beta", "simple", "normal"])
def test_build_sampling_sigmas_dispatches_custom_schedules(
    _scheduler: AnimaFlowMatchEulerDiscreteScheduler,
    sigma_schedule: str,
) -> None:
    result = build_sampling_sigmas(
        _scheduler,
        num_inference_steps=20,
        sigma_schedule=sigma_schedule,
        device="cpu",
    )
    assert result[-1].item() == pytest.approx(0.0, abs=1e-5)
    assert len(result) >= 2
    assert torch.isfinite(result).all()


def test_build_sampling_sigmas_uniform_delegates_to_scheduler(
    _scheduler: AnimaFlowMatchEulerDiscreteScheduler,
) -> None:
    result = build_sampling_sigmas(
        _scheduler,
        num_inference_steps=20,
        sigma_schedule="uniform",
        device="cpu",
    )
    assert len(result) >= 2
    assert torch.isfinite(result).all()
