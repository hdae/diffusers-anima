from __future__ import annotations

import warnings

import pytest

from diffusers_anima.pipelines.anima.validation import (
    _validate_sampling_modes,
    _warn_ignored_sampling_arguments,
)


# ---------------------------------------------------------------------------
# _validate_sampling_modes — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sampler", "sigma_schedule"),
    [
        ("flowmatch_euler", "uniform"),
        ("euler", "beta"),
        ("euler", "normal"),
        ("euler", "simple"),
        ("euler_a_rf", "beta"),
        ("euler_a_rf", "normal"),
        ("euler_a_rf", "simple"),
        ("euler_ancestral_rf", "beta"),
        ("euler_ancestral_rf", "normal"),
        ("euler_ancestral_rf", "simple"),
    ],
)
def test_validate_sampling_modes_accepts_supported_combinations(
    sampler: str, sigma_schedule: str,
) -> None:
    _validate_sampling_modes(
        sampler=sampler,
        sigma_schedule=sigma_schedule,
        cfg_batch_mode="split",
        output_type="pil",
    )


@pytest.mark.parametrize("output_type", ["pil", "np", "latent"])
def test_validate_sampling_modes_accepts_supported_output_types(
    output_type: str,
) -> None:
    _validate_sampling_modes(
        sampler="euler_a_rf",
        sigma_schedule="beta",
        cfg_batch_mode="split",
        output_type=output_type,
    )


@pytest.mark.parametrize("cfg_batch_mode", ["split", "concat"])
def test_validate_sampling_modes_accepts_supported_cfg_batch_modes(
    cfg_batch_mode: str,
) -> None:
    _validate_sampling_modes(
        sampler="euler_a_rf",
        sigma_schedule="beta",
        cfg_batch_mode=cfg_batch_mode,
        output_type="pil",
    )


# ---------------------------------------------------------------------------
# _validate_sampling_modes — error paths
# ---------------------------------------------------------------------------


def test_validate_sampling_modes_rejects_unknown_sampler() -> None:
    with pytest.raises(ValueError, match="`sampler` must be one of"):
        _validate_sampling_modes(
            sampler="dpmpp_2m",
            sigma_schedule="beta",
            cfg_batch_mode="split",
            output_type="pil",
        )


def test_validate_sampling_modes_requires_uniform_for_flowmatch() -> None:
    with pytest.raises(ValueError, match="requires `sigma_schedule='uniform'`"):
        _validate_sampling_modes(
            sampler="flowmatch_euler",
            sigma_schedule="beta",
            cfg_batch_mode="split",
            output_type="pil",
        )


def test_validate_sampling_modes_rejects_unknown_cfg_batch_mode() -> None:
    with pytest.raises(ValueError, match="cfg_batch_mode"):
        _validate_sampling_modes(
            sampler="euler_a_rf",
            sigma_schedule="beta",
            cfg_batch_mode="interleave",
            output_type="pil",
        )


def test_validate_sampling_modes_rejects_unknown_output_type() -> None:
    with pytest.raises(ValueError, match="output_type"):
        _validate_sampling_modes(
            sampler="euler_a_rf",
            sigma_schedule="beta",
            cfg_batch_mode="split",
            output_type="pt",
        )


def test_validate_sampling_modes_rejects_unknown_sigma_schedule() -> None:
    with pytest.raises(ValueError):
        _validate_sampling_modes(
            sampler="euler",
            sigma_schedule="cosine",
            cfg_batch_mode="split",
            output_type="pil",
        )


# ---------------------------------------------------------------------------
# _warn_ignored_sampling_arguments
# ---------------------------------------------------------------------------


def test_warn_ignored_sampling_arguments_for_flowmatch() -> None:
    with pytest.warns(UserWarning, match="eta, s_noise"):
        _warn_ignored_sampling_arguments(
            sampler="flowmatch_euler",
            sigma_schedule="uniform",
            beta_alpha=0.6,
            beta_beta=0.6,
            eta=0.5,
            s_noise=1.2,
        )


def test_warn_ignored_sampling_arguments_for_non_beta_schedule() -> None:
    with pytest.warns(UserWarning, match="beta_alpha, beta_beta"):
        _warn_ignored_sampling_arguments(
            sampler="euler_a_rf",
            sigma_schedule="normal",
            beta_alpha=0.4,
            beta_beta=0.9,
            eta=1.0,
            s_noise=1.0,
        )


def test_warn_ignored_sampling_arguments_no_warning_when_effective() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_ignored_sampling_arguments(
            sampler="euler_a_rf",
            sigma_schedule="beta",
            beta_alpha=0.6,
            beta_beta=0.6,
            eta=1.0,
            s_noise=1.0,
        )
