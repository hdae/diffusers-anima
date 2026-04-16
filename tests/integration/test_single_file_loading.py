"""Integration test: load an Anima transformer checkpoint from a raw ``.safetensors`` file.

Opt-in via env var — the test requires a real checkpoint on disk. Typical uses:

```bash
# Validate waiANIMA v10 / any ComfyUI-saved Anima checkpoint
ANIMA_SINGLE_FILE_CHECKPOINT_PATH=/abs/path/to/waiANIMA_v10.safetensors \
  uv run pytest tests/integration/test_single_file_loading.py -m integration -s

# Regression check against a Preview 1 raw checkpoint
ANIMA_SINGLE_FILE_CHECKPOINT_PATH=/abs/path/to/anima-preview1.safetensors \
  uv run pytest tests/integration/test_single_file_loading.py -m integration -s
```
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from diffusers_anima.models.transformers.modeling_anima_transformer import (
    AnimaTransformerModel,
)
from diffusers_anima.pipelines.anima.loading import load_transformer_native


pytestmark = [pytest.mark.integration]

_ENV_VAR = "ANIMA_SINGLE_FILE_CHECKPOINT_PATH"


def _resolve_checkpoint_path() -> Path:
    raw = os.getenv(_ENV_VAR, "").strip()
    if not raw:
        pytest.skip(f"{_ENV_VAR} not set; skipping single-file load test.")
    path = Path(raw).expanduser()
    if not path.is_file():
        pytest.skip(f"{_ENV_VAR}={raw} does not point to an existing file.")
    return path


def test_load_transformer_from_single_file() -> None:
    """Single-file load should succeed regardless of ComfyUI-style key wrappers."""
    checkpoint_path = _resolve_checkpoint_path()

    transformer = load_transformer_native(
        model_path=str(checkpoint_path),
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert isinstance(transformer, AnimaTransformerModel)
    # Every parameter should have been populated by the checkpoint; any parameter
    # left at a default init value would indicate a missed key mapping.
    assert sum(p.numel() for p in transformer.parameters()) > 0
