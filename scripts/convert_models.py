"""Convert a raw Anima checkpoint into a Diffusers-format pipeline directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from diffusers_anima import AnimaPipeline


def _parse_torch_dtype(value: str) -> torch.dtype:
    mapping: dict[str, torch.dtype] = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported --torch-dtype: {value}. Use one of: {', '.join(sorted(mapping))}"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load an original Anima single-file checkpoint and save a standard "
            "Diffusers-format pipeline directory."
        )
    )
    parser.add_argument(
        "source",
        help=(
            "Single-file source path (local .safetensors, HF file URL, or "
            "'repo_id::filename')."
        ),
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for the converted Diffusers pipeline.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype used while loading the raw checkpoint before save_pretrained().",
    )
    parser.add_argument(
        "--text-encoder-dtype",
        default="bfloat16",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Text encoder dtype for the single-file loader.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disable network access when resolving dependent assets.",
    )
    parser.add_argument(
        "--metadata-from",
        default=None,
        help=(
            "Optional path to an existing converted Anima Diffusers repository/snapshot. "
            "Copies README/LICENSE/licenses metadata into the output directory and applies "
            "small wording fixes."
        ),
    )
    return parser


def _patch_copied_metadata_text(output_dir: Path) -> None:
    replacements_by_file = {
        output_dir / "README.md": [
            ("T5 Components (Text Encoder)", "T5 Components (Tokenizer)"),
            (
                "# Optimizations\npipe.enable_model_cpu_offload()\npipe.enable_vae_slicing()\n",
                '# Standard Diffusers GPU path\npipe.to("cuda")\n'
                '# Lower-VRAM alternative (use instead of `pipe.to("cuda")`)\n'
                "# pipe.enable_model_cpu_offload()\n\n"
                "# VAE memory optimizations\n"
                "pipe.enable_vae_slicing()\n"
                "# pipe.enable_vae_tiling()\n",
            ),
        ],
        output_dir / "licenses" / "README.md": [
            ("Google Flan-T5-XXL Text Encoder", "Google Flan-T5-XL Tokenizer"),
        ],
    }

    for path, replacements in replacements_by_file.items():
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        updated = content
        for before, after in replacements:
            updated = updated.replace(before, after)
        if updated != content:
            path.write_text(updated, encoding="utf-8")

    top_level_license = output_dir / "LICENSE"
    if not top_level_license.exists() or top_level_license.stat().st_size == 0:
        top_level_license.write_text(
            (
                "This repository aggregates components under multiple licenses.\n\n"
                "See `README.md` and `licenses/` for component-level licensing details.\n"
                "Use and redistribution of the combined weights are governed by the most restrictive terms.\n"
            ),
            encoding="utf-8",
        )


def _copy_metadata_artifacts(*, metadata_from: str, output_dir: Path) -> None:
    source_dir = Path(metadata_from).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"--metadata-from must point to an existing directory, got: {source_dir}"
        )

    for name in (".gitattributes", "README.md", "LICENSE"):
        src = source_dir / name
        if src.is_file():
            shutil.copy2(src, output_dir / name)

    licenses_src = source_dir / "licenses"
    licenses_dst = output_dir / "licenses"
    if licenses_src.is_dir():
        shutil.copytree(licenses_src, licenses_dst, dirs_exist_ok=True)

    _patch_copied_metadata_text(output_dir)


def _patch_converted_model_index(output_dir: Path) -> None:
    """Remove single-file provenance fields from model_index.json.

    After Phase G (remove external model references from pipeline public API)
    these fields are no longer written by ``register_to_config``.
    This patch cleans up any residue from older cached conversions.
    """
    model_index_path = output_dir / "model_index.json"
    if not model_index_path.is_file():
        return

    data = json.loads(model_index_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return

    # Fields that were previously registered via register_to_config but are now
    # internal to loading.py and should not appear in the serialised config.
    _LEGACY_FIELDS = {
        "model_path",
        "text_encoder_weights",
        "text_encoder_config_repo",
        "qwen_tokenizer_repo",
        "t5_tokenizer_repo",
        "vae_repo",
    }
    changed = any(data.pop(field, None) is not None for field in _LEGACY_FIELDS)
    if not changed:
        return

    model_index_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _build_parser().parse_args()

    pipe = AnimaPipeline.from_single_file(
        args.source,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
        text_encoder_dtype=args.text_encoder_dtype,
        local_files_only=args.local_files_only,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(output_dir)
    _patch_converted_model_index(output_dir)
    if args.metadata_from is not None:
        _copy_metadata_artifacts(
            metadata_from=args.metadata_from, output_dir=output_dir
        )
    print(f"Saved Diffusers-format Anima pipeline to {output_dir}")


if __name__ == "__main__":
    main()
