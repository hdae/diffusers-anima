from __future__ import annotations

import importlib
import inspect

from diffusers import FlowMatchEulerDiscreteScheduler
import pytest
import torch
from torch import nn

from diffusers_anima.pipelines.anima.loading import _recast_module_to_parameter_dtype
from diffusers_anima.pipelines.anima.pipeline_anima import AnimaPipeline
from diffusers_anima.pipelines.anima.validation import (
    _partition_single_file_from_pretrained_kwargs,
)
from diffusers_anima.schedulers import AnimaFlowMatchEulerDiscreteScheduler


def test_call_signature_exposes_supported_arguments_only() -> None:
    signature = inspect.signature(AnimaPipeline.__call__)
    supported = {
        "prompt",
        "negative_prompt",
        "prompt_embeds",
        "negative_prompt_embeds",
        "image",
        "mask_image",
        "strength",
        "width",
        "height",
        "num_inference_steps",
        "num_images_per_prompt",
        "guidance_scale",
        "generator",
        "cfg_batch_mode",
        "output_type",
        "return_dict",
        "callback_on_step_end",
        "callback_on_step_end_tensor_inputs",
    }
    unsupported = {
        "prompt_2",
        "negative_prompt_2",
        "timesteps",
        "sigmas",
        "denoising_end",
        "true_cfg_scale",
        "sampler",
        "sigma_schedule",
        "beta_alpha",
        "beta_beta",
        "eta",
        "s_noise",
        "latents",
        "pooled_prompt_embeds",
        "negative_pooled_prompt_embeds",
        "ip_adapter_image",
        "ip_adapter_image_embeds",
        "negative_ip_adapter_image",
        "negative_ip_adapter_image_embeds",
        "cross_attention_kwargs",
        "joint_attention_kwargs",
        "guidance_rescale",
        "original_size",
        "crops_coords_top_left",
        "target_size",
        "negative_original_size",
        "negative_crops_coords_top_left",
        "negative_target_size",
        "clip_skip",
        "max_sequence_length",
        "kwargs",
    }

    params = set(signature.parameters)
    assert supported.issubset(params)
    assert unsupported.isdisjoint(params)
    assert signature.parameters["prompt"].default is inspect.Parameter.empty


def test_anima_scheduler_updates_sampling_config() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=False,
    )
    anima_scheduler = AnimaFlowMatchEulerDiscreteScheduler.from_config(
        scheduler.config,
        sampler="flowmatch_euler",
        sigma_schedule="uniform",
        beta_alpha=0.7,
        beta_beta=0.9,
        eta=1.2,
        s_noise=0.8,
    )
    sampling = anima_scheduler.get_sampling_config()

    assert sampling.sampler == "flowmatch_euler"
    assert sampling.sigma_schedule == "uniform"
    assert sampling.beta_alpha == 0.7
    assert sampling.beta_beta == 0.9
    assert sampling.eta == 1.2
    assert sampling.s_noise == 0.8
    assert anima_scheduler.config.sampler == "flowmatch_euler"
    assert anima_scheduler.config.sigma_schedule == "uniform"
    assert anima_scheduler.config.beta_alpha == 0.7
    assert anima_scheduler.config.beta_beta == 0.9
    assert anima_scheduler.config.eta == 1.2
    assert anima_scheduler.config.s_noise == 0.8


def test_anima_scheduler_rejects_invalid_sampler_schedule_pair() -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=False,
    )
    with pytest.raises(ValueError, match="requires `sigma_schedule='uniform'`"):
        AnimaFlowMatchEulerDiscreteScheduler.from_config(
            scheduler.config,
            sampler="flowmatch_euler",
            sigma_schedule="beta",
        )


def test_anima_scheduler_exposes_only_current_sampling_api() -> None:
    signature = inspect.signature(AnimaFlowMatchEulerDiscreteScheduler.__init__)
    params = set(signature.parameters)

    assert {
        "sampler",
        "sigma_schedule",
        "beta_alpha",
        "beta_beta",
        "eta",
        "s_noise",
    }.issubset(params)
    assert {
        "anima_sampler",
        "anima_sigma_schedule",
        "anima_beta_alpha",
        "anima_beta_beta",
        "anima_eta",
        "anima_s_noise",
    }.isdisjoint(params)
    assert not hasattr(
        AnimaFlowMatchEulerDiscreteScheduler, "set_anima_sampling_config"
    )
    assert not hasattr(
        AnimaFlowMatchEulerDiscreteScheduler, "get_anima_sampling_config"
    )


def test_partition_single_file_from_pretrained_kwargs_splits_ignored_and_unknown() -> (
    None
):
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=False,
    )
    ignored, unknown = _partition_single_file_from_pretrained_kwargs(
        {
            "local_files_only": True,
            "scheduler": scheduler,
            "variant": "fp16",
            "use_safetensors": True,
            "unexpected_arg": 1,
        }
    )

    assert ignored == ["use_safetensors", "variant"]
    assert unknown == ["unexpected_arg"]


def test_from_single_file_ignores_compatibility_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=False,
    )
    captured: dict[str, object] = {}

    def _fake_single_file_loader(
        cls: type[AnimaPipeline], path: str, *, kwargs: object
    ) -> str:
        captured["cls"] = cls
        captured["path"] = path
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(
        AnimaPipeline, "_from_single_file_source", classmethod(_fake_single_file_loader)
    )

    with pytest.warns(UserWarning, match="from_single_file"):
        result = AnimaPipeline.from_single_file(
            "dummy.safetensors",
            config="CompVis/stable-diffusion-v1-4",
            disable_mmap=True,
            original_config="dummy.yaml",
            variant="fp16",
            local_files_only=True,
            scheduler=scheduler,
        )

    assert result == "ok"
    assert captured["path"] == "dummy.safetensors"
    assert captured["kwargs"] == {"local_files_only": True, "scheduler": scheduler}


def test_from_pretrained_routes_repo_id_to_diffusers_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=False,
    )
    captured: dict[str, object] = {}

    def _fake_diffusers_repo_loader(
        cls: type[AnimaPipeline], path: str, *, kwargs: dict[str, object]
    ) -> str:
        captured["cls"] = cls
        captured["path"] = path
        captured["kwargs"] = dict(kwargs)
        return "ok"

    monkeypatch.setattr(
        AnimaPipeline,
        "_from_pretrained_diffusers_repo",
        classmethod(_fake_diffusers_repo_loader),
    )

    result = AnimaPipeline.from_pretrained(
        "hdae/diffusers-anima-preview",
        local_files_only=True,
        scheduler=scheduler,
    )

    assert result == "ok"
    assert captured["path"] == "hdae/diffusers-anima-preview"
    assert captured["kwargs"]["local_files_only"] is True
    assert isinstance(
        captured["kwargs"]["scheduler"], AnimaFlowMatchEulerDiscreteScheduler
    )


@pytest.mark.parametrize(
    "source",
    [
        "dummy.safetensors",
        "repo-id::weights/model.safetensors",
        "https://huggingface.co/org/repo/blob/main/model.safetensors",
    ],
)
def test_from_pretrained_rejects_single_file_sources(source: str) -> None:
    with pytest.raises(ValueError, match="from_single_file"):
        AnimaPipeline.from_pretrained(
            source,
        )


def test_recast_module_to_parameter_dtype_updates_non_persistent_buffers() -> None:
    class _DummyModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
            self.register_buffer(
                "runtime_buffer",
                torch.ones(1, dtype=torch.float32),
                persistent=False,
            )

    module = _DummyModule()

    assert module.weight.dtype == torch.bfloat16
    assert module.runtime_buffer.dtype == torch.float32

    _recast_module_to_parameter_dtype(module)

    assert module.runtime_buffer.dtype == torch.bfloat16


def test_from_pretrained_rejects_non_flowmatch_scheduler() -> None:
    with pytest.raises(ValueError, match="FlowMatchEulerDiscreteScheduler"):
        AnimaPipeline.from_pretrained(
            "hdae/diffusers-anima-preview",
            scheduler=object(),
        )


def test_loading_module_does_not_import_diffusers_private_apis() -> None:
    """loading.py must not import private Diffusers symbols (names starting with _)."""
    import diffusers.utils.hub_utils as hub_utils
    import diffusers_anima.pipelines.anima.loading as loading_module

    # Collect everything imported from diffusers.utils.hub_utils into loading.py
    hub_utils_members = {name for name in dir(hub_utils)}
    loading_globals = vars(loading_module)
    private_diffusers_imports = [
        name
        for name, obj in loading_globals.items()
        if name.startswith("_") and not name.startswith("__")
        and name in hub_utils_members
        and getattr(hub_utils, name, None) is obj
    ]
    assert private_diffusers_imports == [], (
        f"loading.py imports private Diffusers symbols: {private_diffusers_imports}"
    )


def test_encode_prompt_is_public_method() -> None:
    """AnimaPipeline must expose encode_prompt as a public method."""
    assert hasattr(AnimaPipeline, "encode_prompt"), (
        "AnimaPipeline is missing the encode_prompt public method"
    )
    sig = inspect.signature(AnimaPipeline.encode_prompt)
    params = set(sig.parameters)
    assert "prompt" in params
    assert "negative_prompt" in params
    assert "num_images_per_prompt" in params


def test_optional_components_is_empty() -> None:
    """prompt_tokenizer must NOT appear in _optional_components.

    It is a plain instance attribute managed outside the Diffusers component
    registry. Putting it in _optional_components without registering it via
    register_modules would mislead Diffusers' CPU-offload and component
    introspection machinery.
    """
    assert AnimaPipeline._optional_components == []


def test_prepare_prompt_embedding_inputs_raises_runtime_error_without_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing prompt_tokenizer must raise RuntimeError, not AssertionError.
    AssertionError is silenced by Python -O; RuntimeError is not.
    """
    from diffusers_anima.pipelines.anima import pipeline_anima as _m

    class _FakePipe:
        prompt_tokenizer = None
        execution_device = "cpu"
        text_encoder_dtype = torch.float32
        model_dtype = torch.float32
        use_module_cpu_offload = False
        text_encoder = torch.nn.Linear(1, 1)

    with pytest.raises(RuntimeError, match="prompt_tokenizer"):
        _m._prepare_prompt_embedding_inputs(
            _FakePipe(),  # type: ignore[arg-type]
            prompt=["hello"],
            negative_prompt=[""],
        )


def test_pipeline_module_imports_F_at_module_level() -> None:
    """torch.nn.functional must be imported at module level, not inside a function."""
    import diffusers_anima.pipelines.anima.pipeline_anima as pipeline_module
    import torch.nn.functional as F_ref

    assert getattr(pipeline_module, "F", None) is F_ref, (
        "torch.nn.functional (F) should be a module-level name in pipeline_anima.py"
    )


def test_from_pretrained_rejects_removed_runtime_feature_kwargs() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported `from_pretrained` arguments: enable_vae_slicing",
    ):
        AnimaPipeline.from_pretrained(
            "dummy.safetensors",
            enable_vae_slicing=True,
        )


# ---------------------------------------------------------------------------
# prompt_embeds / negative_prompt_embeds pairing validation
# ---------------------------------------------------------------------------


def _call_check_inputs_with_embeds(
    prompt_embeds: torch.Tensor | None,
    negative_prompt_embeds: torch.Tensor | None,
) -> None:
    """Call check_inputs via unbound method with a minimal stub."""
    import types

    stub = types.SimpleNamespace()
    AnimaPipeline.check_inputs(  # type: ignore[arg-type]
        stub,
        prompt="dummy",
        negative_prompt=None,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        image=None,
        mask_image=None,
        strength=1.0,
        width=1024,
        height=1024,
        num_inference_steps=1,
        num_images_per_prompt=1,
        generator=None,
        sampler="euler_a_rf",
        sigma_schedule="beta",
        cfg_batch_mode="split",
        output_type="pil",
    )


def test_check_inputs_rejects_prompt_embeds_without_negative_prompt_embeds() -> None:
    """Providing prompt_embeds without negative_prompt_embeds must raise ValueError."""
    with pytest.raises(ValueError, match="both be provided"):
        _call_check_inputs_with_embeds(
            prompt_embeds=torch.zeros(1, 512, 1024),
            negative_prompt_embeds=None,
        )


def test_check_inputs_rejects_negative_prompt_embeds_without_prompt_embeds() -> None:
    """Providing negative_prompt_embeds without prompt_embeds must raise ValueError."""
    with pytest.raises(ValueError, match="both be provided"):
        _call_check_inputs_with_embeds(
            prompt_embeds=None,
            negative_prompt_embeds=torch.zeros(1, 512, 1024),
        )


def test_call_signature_includes_prompt_embeds() -> None:
    """encode_prompt return values must be passable to __call__ via prompt_embeds."""
    sig = inspect.signature(AnimaPipeline.__call__)
    assert "prompt_embeds" in sig.parameters
    assert "negative_prompt_embeds" in sig.parameters
    # Both default to None so users who don't use encode_prompt are unaffected.
    assert sig.parameters["prompt_embeds"].default is None
    assert sig.parameters["negative_prompt_embeds"].default is None
