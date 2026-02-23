from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import PIL.Image
from diffusers.utils import BaseOutput
import torch


@dataclass
class AnimaPipelineOutput(BaseOutput):
    """Output class for Anima image generation pipelines."""

    images: list[PIL.Image.Image] | np.ndarray | torch.Tensor
