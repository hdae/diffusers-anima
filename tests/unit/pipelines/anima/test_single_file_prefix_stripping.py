"""Tests for state-dict wrapping-prefix handling used by ``from_single_file``."""

from __future__ import annotations

import torch

from diffusers_anima.pipelines.anima.loading import _strip_wrapping_prefixes


def _sample_anima_like_state_dict() -> dict[str, torch.Tensor]:
    """Keys representative of a real Anima transformer state dict (root + blocks + adapter)."""
    return {
        "x_embedder.proj.1.weight": torch.zeros(1),
        "t_embedder.1.linear_1.weight": torch.zeros(1),
        "blocks.0.self_attn.q_proj.weight": torch.zeros(1),
        "blocks.27.mlp.layer2.weight": torch.zeros(1),
        "llm_adapter.embed.weight": torch.zeros(1),
    }


def _prefix_keys(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {f"{prefix}{key}": value for key, value in state_dict.items()}


def test_strip_wrapping_prefixes_passes_through_unprefixed_state_dict() -> None:
    base = _sample_anima_like_state_dict()

    result = _strip_wrapping_prefixes(base)

    assert set(result.keys()) == set(base.keys())


def test_strip_wrapping_prefixes_removes_net_wrapper() -> None:
    base = _sample_anima_like_state_dict()
    wrapped = _prefix_keys(base, "net.")

    result = _strip_wrapping_prefixes(wrapped)

    assert set(result.keys()) == set(base.keys())


def test_strip_wrapping_prefixes_removes_model_wrapper() -> None:
    base = _sample_anima_like_state_dict()
    wrapped = _prefix_keys(base, "model.")

    result = _strip_wrapping_prefixes(wrapped)

    assert set(result.keys()) == set(base.keys())


def test_strip_wrapping_prefixes_removes_diffusion_model_wrapper() -> None:
    base = _sample_anima_like_state_dict()
    wrapped = _prefix_keys(base, "diffusion_model.")

    result = _strip_wrapping_prefixes(wrapped)

    assert set(result.keys()) == set(base.keys())


def test_strip_wrapping_prefixes_removes_composite_comfyui_wrapper() -> None:
    """waiANIMA_v10 ships keys like ``model.diffusion_model.x_embedder.proj.1.weight``."""
    base = _sample_anima_like_state_dict()
    wrapped = _prefix_keys(base, "model.diffusion_model.")

    result = _strip_wrapping_prefixes(wrapped)

    assert set(result.keys()) == set(base.keys())


def test_strip_wrapping_prefixes_preserves_tensors() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    wrapped = {"model.diffusion_model.x_embedder.proj.1.weight": tensor}

    result = _strip_wrapping_prefixes(wrapped)

    assert "x_embedder.proj.1.weight" in result
    assert torch.equal(result["x_embedder.proj.1.weight"], tensor)


def test_strip_wrapping_prefixes_does_not_strip_when_prefix_is_not_shared() -> None:
    """If only some keys start with a prefix, leave all keys alone to avoid silent merges."""
    mixed = {
        "model.x_embedder.proj.1.weight": torch.zeros(1),
        "x_embedder.proj.1.weight": torch.zeros(1),  # same key would result after stripping
    }

    result = _strip_wrapping_prefixes(mixed)

    assert set(result.keys()) == set(mixed.keys())


def test_strip_wrapping_prefixes_on_empty_state_dict() -> None:
    assert _strip_wrapping_prefixes({}) == {}
