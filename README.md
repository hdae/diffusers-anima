# diffusers-anima

`diffusers-anima` provides an Anima pipeline implementation designed to align with [Diffusers](https://github.com/huggingface/diffusers) patterns.

## Install

```bash
uv sync
```

## Quick Start

### Text-to-image

```python
import torch
from diffusers_anima import AnimaPipeline

pipe = AnimaPipeline.from_pretrained(
    "hdae/diffusers-anima-preview",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

result = pipe(
    "masterpiece, best quality, score 9, score 8, newest, absurdres, very aesthetic, highres, 1girl, solo, long hair, blue eyes, white blouse, pleated skirt, looking at viewer, gentle smile, smiling",
    negative_prompt="worst quality, low quality, score_1, score_2, score_3, monochrome, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, nsfw, nude, nipples, areola, cleavage, breasts, large breasts, suggestive, erotic, explicit",
    width=1024,
    height=1024,
    num_inference_steps=32,
    guidance_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
)
result.images[0].save("output.png")
```

### From a single-file checkpoint

```python
pipe = AnimaPipeline.from_single_file("/path/to/anima.safetensors")
```

### Img2Img / Inpaint

```python
from PIL import Image

init_image = Image.open("input.png").convert("RGB")
mask_image = Image.open("mask.png").convert("L")  # white = repaint area

# Img2Img:
result = pipe(
    "masterpiece, best quality, score 9, score 8, newest, absurdres, very aesthetic, highres, 1girl, solo, long hair, blue eyes, white blouse, pleated skirt, looking at viewer, gentle smile, smiling",
    negative_prompt="worst quality, low quality, score_1, score_2, score_3, monochrome, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, nsfw, nude, nipples, areola, cleavage, breasts, large breasts, suggestive, erotic, explicit",
    image=init_image,
    strength=0.65,
    width=1024,
    height=1024,
    num_inference_steps=32,
    guidance_scale=4.0,
)

# Inpaint:
result = pipe(
    "masterpiece, best quality, score 9, score 8, newest, absurdres, very aesthetic, highres, 1girl, solo, long hair, blue eyes, white blouse, pleated skirt, looking at viewer, gentle smile, smiling",
    negative_prompt="worst quality, low quality, score_1, score_2, score_3, monochrome, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, nsfw, nude, nipples, areola, cleavage, breasts, large breasts, suggestive, erotic, explicit",
    image=init_image,
    mask_image=mask_image,
    strength=0.75,
    width=1024,
    height=1024,
    num_inference_steps=32,
    guidance_scale=4.0,
)
```

### LoRA

```python
pipe.load_lora_weights("/path/to/lora.safetensors", adapter_name="style")
pipe.set_adapters("style", adapter_weights=[0.8])

result = pipe(
    "masterpiece, best quality, score 9, score 8, newest, absurdres, very aesthetic, highres, 1girl, solo, long hair, blue eyes, white blouse, pleated skirt, looking at viewer, gentle smile, smiling",
    negative_prompt="worst quality, low quality, score_1, score_2, score_3, monochrome, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, nsfw, nude, nipples, areola, cleavage, breasts, large breasts, suggestive, erotic, explicit",
    width=1024, 
    height=1024,
)
```

## Documentation

| Document | Description |
|---|---|
| [`docs/api.md`](docs/api.md) | Full API reference (loading, generation, sampling config) |
| [`docs/development.md`](docs/development.md) | Development setup, test commands, project structure |
| [`docs/custom_implementations.md`](docs/custom_implementations.md) | Intentional deviations from Diffusers upstream |
