# Development Guide

## Environment Setup

```bash
# Install Python dependencies (requires uv)
uv sync
```

## Core Commands

| Command | Description |
|---|---|
| `uv run ruff format src/` | Auto-format source files |
| `uv run ruff check src/` | Lint source files |
| `uv run pytest tests/unit/ -q` | Run unit tests |
| `uv run pytest tests/unit/ -x -q` | Run unit tests, stop on first failure |

## Unit Tests

Unit tests are fast and require no GPU or model weights:

```bash
uv run pytest tests/unit/ -q
```

Tests live in `tests/unit/`. They cover:
- Pipeline argument validation and interface contracts
- Generator and random tensor utilities
- Image preprocessing and mask handling
- Prompt batch expansion
- Sampler/schedule validation
- Sigma schedule construction

## Integration Regression Tests

The integration tests load the full Anima model and run inference. A GPU and
model weights are required.

```bash
ANIMA_RUN_INTEGRATION=1 ANIMA_VAE_SLICING=1 \
  uv run pytest tests/integration/test_regression_1girl.py -m integration -s
```

To update the stored baselines after an intentional output change:

```bash
ANIMA_RUN_INTEGRATION=1 ANIMA_UPDATE_BASELINE=1 ANIMA_VAE_SLICING=1 \
  uv run pytest tests/integration/test_regression_1girl.py -m integration -s
```

Set `ANIMA_PRETRAINED_MODEL_PATH` to a local path to avoid downloading from
HuggingFace:

```bash
ANIMA_RUN_INTEGRATION=1 ANIMA_VAE_SLICING=1 \
  ANIMA_PRETRAINED_MODEL_PATH=./converted_pipeline \
  uv run pytest tests/integration/test_regression_1girl.py -m integration -s
```

If you regenerate baselines, include a justification in the commit message
explaining why the outputs legitimately changed (e.g. a corrected VAE
conversion key, not numerical drift from an algorithm change).

## Model Conversion

To convert a raw Anima checkpoint to Diffusers format:

```bash
uv run python scripts/convert_models.py \
  path/to/anima.safetensors \
  ./converted_pipeline
```

This produces a directory compatible with `AnimaPipeline.from_pretrained`.

## Shared GPU Operation

When working on a shared GPU server:

- Check GPU availability before running integration tests: `nvidia-smi`
- Avoid `dtype=float16` on Anima — use `bfloat16` or `auto` instead, as
  float16 may produce NaN/Inf with this model.
- Release GPU memory with `torch.cuda.empty_cache()` or restart the process
  after OOM errors.

## Project Structure

```
src/diffusers_anima/
  __init__.py                    # Public API, _register_local_diffusers_loadables
  loaders/
    lora_pipeline.py             # AnimaLoraLoaderMixin
  models/
    transformers/
      modeling_anima_transformer.py  # AnimaTransformerModel
  pipelines/
    anima/
      constants.py               # Default repos, VAE config, dtype maps
      generator_utils.py         # Generator normalization, noise runtime
      image_processing.py        # Image/mask prep, VAE encode/decode
      loading.py                 # Component loaders, device/dtype resolution
      options.py                 # AnimaComponents / AnimaLoaderOptions / AnimaRuntimeOptions
      pipeline_anima.py          # AnimaPipeline class
      pipeline_output.py         # AnimaPipelineOutput
      prompt_utils.py            # Prompt normalization, batch expansion
      sampling.py                # Euler, ancestral RF, flowmatch samplers
      sigma_schedules.py         # Sigma schedule construction
      strength_utils.py          # Strength-based step trimming (img2img)
      text_encoding.py           # AnimaPromptTokenizer, conditioning utilities
      validation.py              # Input validation, type aliases, constants
      vae_conversion.py          # Anima VAE state_dict → Diffusers format
  schedulers/
    anima_flow_match_euler.py    # AnimaFlowMatchEulerDiscreteScheduler
```

## Code Style

- Python ≥ 3.12 with full type annotations.
- Format and lint with `ruff` (configured in `pyproject.toml`).
- All public APIs should have docstrings.
- Avoid `Any` where a concrete type or protocol can be used.

## Custom Implementations

See [`docs/custom_implementations.md`](custom_implementations.md) for a
catalogue of intentional deviations from Diffusers upstream patterns and the
rationale behind each one.
