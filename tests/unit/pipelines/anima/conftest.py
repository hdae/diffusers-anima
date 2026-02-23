"""Shared fixtures for Anima pipeline unit tests."""

from __future__ import annotations

import types

import pytest
import torch

from diffusers import FlowMatchEulerDiscreteScheduler

from diffusers_anima.schedulers import AnimaFlowMatchEulerDiscreteScheduler


@pytest.fixture()
def base_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """A vanilla Diffusers FlowMatchEulerDiscreteScheduler."""
    return FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=False,
    )


@pytest.fixture()
def anima_scheduler(base_scheduler: FlowMatchEulerDiscreteScheduler) -> AnimaFlowMatchEulerDiscreteScheduler:
    """An AnimaFlowMatchEulerDiscreteScheduler with default Forge settings."""
    return AnimaFlowMatchEulerDiscreteScheduler.from_config(base_scheduler.config)


@pytest.fixture()
def cpu_generator() -> torch.Generator:
    """A deterministic CPU generator seeded at 42."""
    return torch.Generator(device="cpu").manual_seed(42)
