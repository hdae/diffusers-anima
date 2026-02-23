# Custom Implementations

This document records the intentional deviations from standard Diffusers-upstream
patterns used in `diffusers-anima`. Each entry explains the rationale so future
maintainers can evaluate whether upstream changes make the custom code obsolete.

---

## `randn_tensor` in `sampling.py`

**Location:** `src/diffusers_anima/pipelines/anima/sampling.py`

**Why custom?**
Diffusers' `diffusers.utils.torch_utils.randn_tensor` raises a `ValueError` when
a CUDA generator is paired with a CPU target device. The Anima sampling loops
pass a user-supplied `generator` that may live on a different device than the
latent tensors (for example, a seeded CPU generator used during CUDA inference).
The custom `randn_tensor` handles this gracefully by generating noise on the
generator's device in `float32` and transferring it afterwards, matching the
documented semantics of `diffusers.utils.torch_utils.randn_tensor` for the
CPU-generator-to-CUDA-device path without raising.

**Differences from Diffusers:**
- No `layout` parameter (not needed for dense latent tensors).
- Does not emit an info-level log when cross-device generation occurs.
- CUDA-generator + CPU-device is handled (moved silently) rather than raising.

**Upstream candidate:** Yes, `layout` support and better error messages could be
added to align with upstream. However replacing with the Diffusers version would
introduce a `ValueError` regression for any user who already holds a pre-seeded
CUDA generator and runs inference on CPU.

---

## `_AnimaRMSNorm` in `modeling_anima_transformer.py`

**Location:** `src/diffusers_anima/models/transformers/modeling_anima_transformer.py`

**Why custom?**
`diffusers.models.normalization.RMSNorm` computes variance manually in float32:

```python
variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + eps)
```

`_AnimaRMSNorm` delegates directly to `torch.nn.functional.rms_norm`, which is
a fused CUDA kernel (PyTorch ≥ 2.1) and avoids the float32 upcast overhead during
forward passes in bfloat16/float16 inference.

Additionally, `_AnimaRMSNorm.from_diffusers()` is used by
`_patch_diffusers_rmsnorm_to_anima()` to transparently replace Diffusers-loaded
`RMSNorm` modules inside `CosmosTransformer3DModel` with the faster variant after
checkpoint loading.

**Upstream candidate:** If Diffusers' `RMSNorm` switches to `F.rms_norm` internally
(tracked at diffusers issue #…), this wrapper can be removed.

---

## CPU Offload Context Manager (`_module_execution_context`)

**Location:** `src/diffusers_anima/pipelines/anima/pipeline_anima.py`

**Why custom?**
Diffusers' standard `enable_model_cpu_offload` is designed for sequential hooks
attached to the forward call. The Anima pipeline manually orchestrates inference
across three modules (text encoder → transformer → VAE) in a specific order that
differs from the `model_cpu_offload_seq` default cadence because the adapter
conditioning (`build_condition`) runs inside the transformer context before the
main denoising loop.

`_module_execution_context` is a simple context manager that moves a single
module to the execution device for the duration of a `with` block and moves it
back to CPU immediately afterwards, releasing VRAM between stages.

**Upstream alignment:** The pipeline does declare `model_cpu_offload_seq =
"text_encoder->transformer->vae"` so Diffusers' standard `enable_model_cpu_offload`
method is still callable and works for simple cases. The custom context manager
is used internally when `use_module_cpu_offload=True` is set directly (e.g. by
`from_single_file`).

**Note on 2-stage text conditioning:** `build_condition` runs the text encoder
*and* then immediately feeds its output into the transformer's LLM adapter
(`preprocess_text_embeds`). This means the text encoder and transformer must
both be on the execution device at the same time during the conditioning phase,
which conflicts with the linear `text_encoder→transformer→vae` hook sequence
that Diffusers' standard `enable_model_cpu_offload` assumes. The custom context
manager solves this by explicitly moving the transformer to the device early
(for the adapter pass), keeping it resident through the denoising loop, and
only then moving the VAE in for decoding.

---

## `AnimaFlowMatchEulerDiscreteScheduler`

**Location:** `src/diffusers_anima/schedulers/anima_flow_match_euler.py`

**Why custom?**
Extends Diffusers' `FlowMatchEulerDiscreteScheduler` to make Anima-specific
sampling parameters (`sampler`, `sigma_schedule`, `eta`, `s_noise`,
`beta_alpha`, `beta_beta`) first-class serialisable config fields via
`register_to_config`. This allows checkpoint round-tripping: `save_pretrained`
/ `from_pretrained` preserves the full sampling configuration without requiring
callers to re-specify it.

**Upstream candidate:** Sampling parameter serialisation could be contributed
to the upstream scheduler directly.

---

## Custom Sigma Schedules (`sigma_schedules.py`)

**Location:** `src/diffusers_anima/pipelines/anima/sigma_schedules.py`

**Why custom?**
Anima supports four sigma trajectories beyond what Diffusers' scheduler exposes:

| Schedule | Description |
|---|---|
| `uniform` | Delegates to `FlowMatchEulerDiscreteScheduler.set_timesteps` |
| `beta` | Beta-distribution-sampled timesteps (ComfyUI/Forge compatible) |
| `simple` | Stride-selected sigmas from a linear base grid |
| `normal` | Linearly-spaced sigmas with SNR-shift applied |

The `beta` schedule requires `scipy.stats.beta.ppf` and produces a density of
timesteps near the middle of the trajectory, matching community samplers such
as those from Comfy/Forge. These are not available in upstream Diffusers.

**Upstream candidate:** `beta` schedule in particular could be contributed as a
general-purpose sigma schedule option for flow-match schedulers.

---

## Custom VAE Key Mapping (`vae_conversion.py`)

**Location:** `src/diffusers_anima/pipelines/anima/vae_conversion.py`

**Why custom?**
The Anima VAE (based on `AutoencoderKLQwenImage`) ships with a non-standard
state dict layout in the original checkpoint. `convert_anima_vae_state_dict`
maps the Anima key conventions (e.g. `encoder.downsamples.*`, `decoder.upsamples.*`,
`residual.*.gamma`) to the Diffusers `AutoencoderKLQwenImage` key conventions
(e.g. `encoder.down_blocks.*`, `decoder.up_blocks.*`, `norm1.gamma`).

This conversion runs once at load time during `from_single_file` and is not
needed when loading from a pre-converted Diffusers-format checkpoint.

**Upstream candidate:** Could be contributed as a standalone conversion script
or integrated into `from_original_config` + `from_single_file` for
`AutoencoderKLQwenImage`, once the mapping is stable.

---

## `AnimaPromptTokenizer`

**Location:** `src/diffusers_anima/pipelines/anima/text_encoding.py`

**Why custom?**
Anima uses a dual-encoder conditioning scheme: Qwen3-0.6B hidden states (for
semantic richness) plus T5-XXL token IDs and per-token weights (for positional
conditioning via LLM adapter). Diffusers has no off-the-shelf class for
heterogeneous multi-tokenizer conditioning of this form.

`AnimaPromptTokenizer` wraps both tokenizers and produces a structured output
(`{"qwen3_06b": ..., "t5xxl": ...}`) consumed by `prepare_condition_inputs`.
All token weights are currently fixed at `1.0` (no parenthesis-weighted syntax
is supported).

**Upstream candidate:** A generic multi-tokenizer abstraction could be
contributed upstream as the dual-encoder pattern becomes more common.

---

## `LOADABLE_CLASSES` registration in `__init__.py`

**Location:** `src/diffusers_anima/__init__.py`

**Why custom?**
Diffusers resolves `model_index.json` component entries by looking up the module
path against an internal registry called `LOADABLE_CLASSES`
(`diffusers.pipelines.pipeline_loading_utils.LOADABLE_CLASSES`).
Third-party packages are not registered automatically, so when a converted Anima
repo references `diffusers_anima.schedulers.anima_flow_match_euler` or
`diffusers_anima.models.transformers.modeling_anima_transformer` in its
`model_index.json`, Diffusers rejects those entries during `from_pretrained`
unless the package self-registers.

`_register_loadable_classes()` (called at import time from `__init__.py`) patches
this registry via `setdefault` so that existing entries are never overwritten.
The call is wrapped in `try/except ImportError` to survive Diffusers version
changes that may rename or remove `pipeline_loading_utils`.

**Upstream candidate:** Diffusers does not yet expose a public API for third-party
plugin registration. A `register_loadable_module()` public function would
eliminate this workaround. Tracking issue to be filed upstream.
