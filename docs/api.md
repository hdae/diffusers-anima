# API Reference

## Loading

### `AnimaPipeline.from_pretrained`

```python
AnimaPipeline.from_pretrained(
    pretrained_model_name_or_path: str,
    **kwargs,
) -> AnimaPipeline
```

Loads from a Diffusers-format pipeline directory or Hugging Face Hub repository.
The repository must contain a `model_index.json` file.

Accepts standard Diffusers `from_pretrained` kwargs such as `cache_dir`,
`token`, `revision`, `local_files_only`, and `scheduler`.

> **Note:** Runtime options such as `device`, `dtype`, `text_encoder_dtype`,
> and VAE/offload toggles are **not** accepted here. Configure them after
> loading with the standard pipeline methods.

### `AnimaPipeline.from_single_file`

```python
AnimaPipeline.from_single_file(
    pretrained_model_link_or_path: str,
    *,
    # Component sources (defaults to hdae/diffusers-anima-preview)
    text_encoder_weights: str = ...,
    text_encoder_config_repo: str = ...,
    qwen_tokenizer_repo: str = ...,
    t5_tokenizer_repo: str = ...,
    vae_repo: str = ...,
    # Runtime
    device: str = "auto",
    dtype: str = "auto",
    text_encoder_dtype: str = "auto",
    # Loader options
    local_files_only: bool = False,
    cache_dir: str | None = None,
    token: str | bool | None = None,
    scheduler: FlowMatchEulerDiscreteScheduler | None = None,
) -> AnimaPipeline
```

Loads from a raw `.safetensors` checkpoint. The transformer is loaded from the
given path; all other components default to the Anima preview repository on
Hugging Face.

**Path formats accepted for `pretrained_model_link_or_path`:**
- Local file: `"/path/to/anima.safetensors"`
- HF URL: `"https://huggingface.co/owner/repo/blob/main/model.safetensors"`
- Repo+filename: `"owner/repo::path/to/model.safetensors"`

The same `repo::filename` shorthand applies to all component source args.

---

## Generation — `AnimaPipeline.__call__`

```python
pipe(
    prompt: str | list[str],
    negative_prompt: str | list[str] | None = None,
    image: ImageInput | None = None,
    mask_image: ImageInput | None = None,
    strength: float = 1.0,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 32,
    num_images_per_prompt: int = 1,
    guidance_scale: float = 4.0,
    generator: torch.Generator | list[torch.Generator] | None = None,
    cfg_batch_mode: str = "split",
    output_type: str = "pil",
    return_dict: bool = True,
    callback_on_step_end: Callable | None = None,
    callback_on_step_end_tensor_inputs: list[str] | None = None,
) -> AnimaPipelineOutput
```

### Key parameters

| Parameter | Description |
|---|---|
| `prompt` | Positional argument. Single string or list of strings. |
| `image` | PIL Image / ndarray / tensor for img2img or inpainting. |
| `mask_image` | Inpaint mask: white pixels are inpainted, black are preserved. |
| `strength` | How much noise to add for img2img (0.0–1.0]. 1.0 = full generation. |
| `cfg_batch_mode` | `"split"` (two separate forwards) or `"concat"` (one batched forward). |
| `output_type` | `"pil"` (default), `"np"` (uint8 array), or `"latent"`. |

### Width / Height constraints

`width` and `height` must be divisible by `pipe.spatial_step`
(= `vae_scale_factor × patch_size`, typically **16**).

---

## Sampling Configuration

Sampling parameters are stored in the scheduler config and are serialised with
`save_pretrained`. Change them with:

```python
pipe.scheduler.set_sampling_config(
    sampler="euler_a_rf",      # flowmatch_euler | euler | euler_a_rf | euler_ancestral_rf
    sigma_schedule="beta",     # uniform | beta | simple | normal
    beta_alpha=0.5,
    beta_beta=0.5,
    eta=1.0,
    s_noise=1.0,
)
```

### Sampler/schedule matrix

| `sampler` | Compatible `sigma_schedule` | Notes |
|---|---|---|
| `flowmatch_euler` | `uniform` only | Uses Diffusers scheduler step |
| `euler` | `beta`, `simple`, `normal` | Classic Euler; `eta`/`s_noise` ignored |
| `euler_a_rf` | `beta`, `simple`, `normal` | Ancestral RF Euler |
| `euler_ancestral_rf` | `beta`, `simple`, `normal` | Alias for `euler_a_rf` |

---

## Runtime Helpers

```python
pipe.enable_vae_slicing()      # Reduce peak VRAM at slight speed cost
pipe.disable_vae_slicing()

pipe.enable_vae_tiling()       # For very large images
pipe.disable_vae_tiling()

pipe.enable_model_cpu_offload()  # Standard Diffusers CPU offload
```

---

## LoRA

```python
pipe.load_lora_weights("owner/repo", weight_name="lora.safetensors")
pipe.set_adapters(["lora_name"], adapter_weights=[0.8])
pipe.fuse_lora()
pipe.unfuse_lora()
pipe.unload_lora_weights()
```

See `AnimaLoraLoaderMixin` for the full API surface.
