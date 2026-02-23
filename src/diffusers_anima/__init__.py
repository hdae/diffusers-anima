"""Anima support utilities for Diffusers."""

from .pipelines.anima import AnimaPipeline, AnimaPipelineOutput
from .schedulers import AnimaFlowMatchEulerDiscreteScheduler, AnimaSamplingConfig


def _register_local_diffusers_loadables() -> None:
    """Register local custom component modules for Diffusers pipeline download.

    Converted Anima repos reference installed `diffusers_anima` component modules in
    `model_index.json`. Diffusers checks `LOADABLE_CLASSES` during the download phase
    to decide whether those modules are valid custom components.
    """

    try:
        from diffusers.pipelines.pipeline_loading_utils import LOADABLE_CLASSES
    except ImportError:
        return

    LOADABLE_CLASSES.setdefault(
        "diffusers_anima.schedulers.anima_flow_match_euler",
        {"SchedulerMixin": ["save_pretrained", "from_pretrained"]},
    )
    LOADABLE_CLASSES.setdefault(
        "diffusers_anima.models.transformers.modeling_anima_transformer",
        {"ModelMixin": ["save_pretrained", "from_pretrained"]},
    )


_register_local_diffusers_loadables()

__all__ = [
    "AnimaFlowMatchEulerDiscreteScheduler",
    "AnimaPipeline",
    "AnimaPipelineOutput",
    "AnimaSamplingConfig",
]
