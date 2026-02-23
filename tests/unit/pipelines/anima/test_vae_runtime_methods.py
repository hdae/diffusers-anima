from __future__ import annotations

import types

import pytest
import torch

from diffusers_anima.pipelines.anima.pipeline_anima import AnimaPipeline


class _DummyVAE:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def enable_slicing(self) -> None:
        self.calls.append("enable_slicing")

    def disable_slicing(self) -> None:
        self.calls.append("disable_slicing")

    def enable_tiling(self) -> None:
        self.calls.append("enable_tiling")

    def disable_tiling(self) -> None:
        self.calls.append("disable_tiling")


def _build_pipeline_with_vae(vae: object) -> AnimaPipeline:
    pipeline = object.__new__(AnimaPipeline)
    pipeline.vae = vae  # type: ignore[assignment]
    return pipeline


def test_vae_runtime_methods_toggle_supported_features() -> None:
    vae = _DummyVAE()
    pipe = _build_pipeline_with_vae(vae)

    pipe.enable_vae_slicing()
    pipe.disable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.disable_vae_tiling()

    assert vae.calls == [
        "enable_slicing",
        "disable_slicing",
        "enable_tiling",
        "disable_tiling",
    ]


def _make_offload_stub(*, execution_device: str = "auto") -> types.SimpleNamespace:
    """Return a minimal namespace that satisfies enable_model_cpu_offload's reads/writes."""
    stub = types.SimpleNamespace()
    stub._anima_execution_device = execution_device
    stub.use_module_cpu_offload = False
    return stub


def test_enable_model_cpu_offload_sets_flag_and_device() -> None:
    stub = _make_offload_stub(execution_device="auto")

    AnimaPipeline.enable_model_cpu_offload(stub, device="cuda")  # type: ignore[arg-type]

    assert stub.use_module_cpu_offload is True
    assert stub._anima_execution_device == "cuda"


def test_enable_model_cpu_offload_accepts_torch_device() -> None:
    stub = _make_offload_stub(execution_device="auto")

    AnimaPipeline.enable_model_cpu_offload(stub, device=torch.device("cuda"))  # type: ignore[arg-type]

    assert stub.use_module_cpu_offload is True
    assert stub._anima_execution_device == "cuda"


def test_enable_model_cpu_offload_preserves_explicit_device() -> None:
    """When execution_device was already set explicitly, the device arg is ignored."""
    stub = _make_offload_stub(execution_device="cuda:1")

    AnimaPipeline.enable_model_cpu_offload(stub, device="cuda")  # type: ignore[arg-type]

    assert stub.use_module_cpu_offload is True
    assert stub._anima_execution_device == "cuda:1"


def test_vae_runtime_methods_warn_when_feature_is_not_supported() -> None:
    pipe = _build_pipeline_with_vae(object())

    with pytest.warns(UserWarning, match="VAE slicing"):
        pipe.enable_vae_slicing()
    with pytest.warns(UserWarning, match="VAE tiling"):
        pipe.enable_vae_tiling()
    with pytest.warns(UserWarning, match="VAE slicing"):
        pipe.disable_vae_slicing()
    with pytest.warns(UserWarning, match="VAE tiling"):
        pipe.disable_vae_tiling()
