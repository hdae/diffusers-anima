"""Tests for AnimaTransformerModel Diffusers interface conformance."""
from __future__ import annotations

import inspect

from diffusers.models.modeling_outputs import Transformer2DModelOutput
import pytest
import torch

from diffusers_anima.models.transformers.modeling_anima_transformer import (
    AnimaTransformerModel,
)


def test_forward_uses_diffusers_parameter_names() -> None:
    """forward() must use hidden_states / timestep / encoder_hidden_states."""
    sig = inspect.signature(AnimaTransformerModel.forward)
    params = list(sig.parameters)
    assert "hidden_states" in params, "expected 'hidden_states', not 'x'"
    assert "timestep" in params, "expected 'timestep' (singular), not 'timesteps'"
    assert "encoder_hidden_states" in params, "expected 'encoder_hidden_states', not 'context'"
    assert "x" not in params
    assert "timesteps" not in params
    assert "context" not in params


def test_forward_has_return_dict_parameter() -> None:
    sig = inspect.signature(AnimaTransformerModel.forward)
    assert "return_dict" in sig.parameters
    assert sig.parameters["return_dict"].default is True


def test_forward_return_dict_true_returns_model_output() -> None:
    model = AnimaTransformerModel()
    model.eval()
    B, C, T, H, W = 1, 16, 1, 16, 16
    hidden_states = torch.randn(B, C, T, H, W)
    timestep = torch.tensor([0.5]).expand(B)
    encoder_hidden_states = torch.randn(B, 512, 1024)

    with torch.no_grad():
        out = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
        )

    assert isinstance(out, Transformer2DModelOutput)
    assert hasattr(out, "sample")
    assert out.sample.shape == hidden_states.shape


def test_init_exposes_architecture_config_parameters() -> None:
    """__init__ must expose architecture params so config.json is self-describing."""
    import inspect

    sig = inspect.signature(AnimaTransformerModel.__init__)
    expected = {
        "in_channels",
        "out_channels",
        "num_attention_heads",
        "attention_head_dim",
        "num_layers",
        "mlp_ratio",
        "text_embed_dim",
        "adapter_vocab_size",
        "adapter_dim",
        "adapter_layers",
        "adapter_heads",
    }
    params = set(sig.parameters)
    assert expected.issubset(params), f"Missing config params: {expected - params}"


def test_config_roundtrip_recreates_same_architecture() -> None:
    """from_config(model.config) must produce an identical architecture."""
    original = AnimaTransformerModel()
    config = original.config

    # Verify key architecture fields are present in the config
    assert config.in_channels == 16
    assert config.num_layers == 28
    assert config.num_attention_heads == 16

    recreated = AnimaTransformerModel.from_config(config)
    assert recreated.config.in_channels == original.config.in_channels
    assert recreated.config.num_layers == original.config.num_layers
    assert recreated.config.num_attention_heads == original.config.num_attention_heads
    assert recreated.config.adapter_vocab_size == original.config.adapter_vocab_size


def test_forward_return_dict_false_returns_tuple() -> None:
    model = AnimaTransformerModel()
    model.eval()
    B, C, T, H, W = 1, 16, 1, 16, 16
    hidden_states = torch.randn(B, C, T, H, W)
    timestep = torch.tensor([0.5]).expand(B)
    encoder_hidden_states = torch.randn(B, 512, 1024)

    with torch.no_grad():
        out = model(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )

    assert isinstance(out, tuple)
    assert len(out) == 1
    assert out[0].shape == hidden_states.shape
