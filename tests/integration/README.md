# Integration Regression Tests (Public)

This directory contains public integration regression tests using a `1girl` baseline.
The scenario is expression edit regression (`cry -> smile`) and covers:

- text-to-image
- image-to-image
- inpaint

Model bundle, case settings, and runnable command snapshots are stored in:

- `tests/integration/assets/regression_1girl/model_bundle.json`
- `tests/integration/assets/regression_1girl/case_settings.json`
- `tests/integration/assets/regression_1girl/commands.json`

## Run regression check

```bash
ANIMA_RUN_INTEGRATION=1 uv run pytest tests/integration/test_regression_1girl.py -m integration -s
```

## Update baselines

```bash
ANIMA_RUN_INTEGRATION=1 ANIMA_UPDATE_BASELINE=1 uv run pytest tests/integration/test_regression_1girl.py -m integration -s
```

## Optional environment variables

- `ANIMA_MODEL`
- `ANIMA_TEXT_ENCODER_WEIGHTS`
- `ANIMA_TEXT_ENCODER_CONFIG_REPO`
- `ANIMA_QWEN_TOKENIZER_REPO`
- `ANIMA_T5_TOKENIZER_REPO`
- `ANIMA_VAE_REPO`
- `ANIMA_DEVICE`
- `ANIMA_DTYPE`
- `ANIMA_TEXT_ENCODER_DTYPE`
- `ANIMA_LOCAL_FILES_ONLY`
- `ANIMA_CPU_OFFLOAD`
- `ANIMA_VAE_SLICING`
- `ANIMA_VAE_TILING`
- `ANIMA_VAE_XFORMERS`
- `ANIMA_MAX_ABS_THRESHOLD` (default `0`)
- `ANIMA_NONZERO_PIXELS_THRESHOLD` (default `0`)
