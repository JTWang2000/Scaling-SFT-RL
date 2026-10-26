#!/usr/bin/env python3
"""Apply the GRPO fixes used by this repository's pinned Unsloth environment."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def _site_root(package: str) -> Path:
    try:
        return Path(distribution(package).locate_file("")).resolve()
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"{package} is not installed. Install submission/requirements.txt first."
        ) from error


def _replace(path: Path, old: str, new: str, label: str, check_only: bool) -> str:
    if not path.is_file():
        raise RuntimeError(f"Cannot apply {label}: missing {path}")

    source = path.read_text()
    if new in source:
        return "already applied"
    if old not in source:
        raise RuntimeError(
            f"Cannot apply {label}: the pinned-source pattern was not found in {path}. "
            "Check that unsloth==2025.12.8 and unsloth_zoo==2025.12.6 are installed."
        )
    if not check_only:
        path.write_text(source.replace(old, new, 1))
    return "needs patch" if check_only else "applied"


def _patch_source(check_only: bool) -> list[tuple[str, str]]:
    unsloth_root = _site_root("unsloth")
    zoo_root = _site_root("unsloth-zoo")
    results: list[tuple[str, str]] = []

    replacements = [
        (
            zoo_root / "unsloth_zoo/rl_replacements.py",
            "mask = (tensor != pad_id)",
            "mask = (tensor != pad_id).to(torch.int32)",
            "integer left-pack mask",
        ),
        (
            unsloth_root / "unsloth/models/rl_replacements.py",
            'batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size\n        try:',
            'batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size\n        has_images = images is not None\n        try:',
            "GRPO image-state initialization",
        ),
        (
            unsloth_root / "unsloth/models/rl_replacements.py",
            "grpo_compute_loss_slow(\n                    ref_hidden_states,\n                    per_token_logps,\n                    old_hidden_states,\n                    input_ids,",
            "grpo_compute_loss_slow(\n                    ref_hidden_states,\n                    per_token_logps,\n                    old_hidden_states,\n                    sampling_per_token_logps,\n                    input_ids,",
            "GRPO sampling log-probability argument",
        ),
        (
            unsloth_root / "unsloth/models/rl_replacements.py",
            "\n                    sampling_per_token_logps = sampling_per_token_logps,\n                )",
            "\n                )",
            "duplicate GRPO sampling keyword removal",
        ),
        (
            unsloth_root / "unsloth/models/rl.py",
            '''        if "CUDA_VISIBLE_DEVICES" in os.environ:
            lora_name = (
                trainer_file
                + "_lora_model_' + "
                + "(os.environ.get('CUDA_VISIBLE_DEVICES', '0').replace(',',''))"
            )
        else:
            lora_name = trainer_file + "_lora_model'"
''',
            '''        lora_name = (trainer_file + "_lora_model_' + __import__('socket').gethostname().replace('-', '_') + '_' + (os.environ.get('CUDA_VISIBLE_DEVICES', 'nogpu').replace(',',''))")
''',
            "host-specific vLLM LoRA cache name",
        ),
    ]

    for path, old, new, label in replacements:
        results.append((label, _replace(path, old, new, label, check_only)))
    return results


def _patch_existing_cache(check_only: bool) -> list[tuple[str, str]]:
    cache_file = Path.cwd() / "unsloth_compiled_cache/UnslothGRPOTrainer.py"
    if not cache_file.is_file():
        return [("compiled trainer cache", "not present; source patches will affect generation")]

    replacements = [
        (
            "mask = (tensor != pad_id)",
            "mask = (tensor != pad_id).to(torch.int32)",
            "compiled integer left-pack mask",
        ),
        (
            'batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size\n        try:',
            'batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size\n        has_images = images is not None\n        try:',
            "compiled GRPO image-state initialization",
        ),
        (
            "grpo_compute_loss_slow(\n                    ref_hidden_states,\n                    per_token_logps,\n                    old_hidden_states,\n                    input_ids,",
            "grpo_compute_loss_slow(\n                    ref_hidden_states,\n                    per_token_logps,\n                    old_hidden_states,\n                    sampling_per_token_logps,\n                    input_ids,",
            "compiled GRPO sampling log-probability argument",
        ),
        (
            "\n                    sampling_per_token_logps = sampling_per_token_logps,\n                )",
            "\n                )",
            "compiled duplicate GRPO sampling keyword removal",
        ),
        (
            "self.model.load_lora('grpo_trainer_lora_model_' + (os.environ.get('CUDA_VISIBLE_DEVICES', '0').replace(',','')), load_tensors = True)",
            "self.model.load_lora('grpo_trainer_lora_model_' + __import__('socket').gethostname().replace('-', '_') + '_' + (os.environ.get('CUDA_VISIBLE_DEVICES', 'nogpu').replace(',','')), load_tensors = True)",
            "compiled host-specific vLLM LoRA cache name",
        ),
    ]
    return [
        (label, _replace(cache_file, old, new, label, check_only))
        for old, new, label in replacements
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify patch state without changing installed files.",
    )
    args = parser.parse_args()

    results = _patch_source(args.check)
    results.extend(_patch_existing_cache(args.check))
    for label, status in results:
        print(f"{label}: {status}")

    if args.check and any(status == "needs patch" for _, status in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
