# Scaling Near-Optimal SFT-RL Annotation Budget Allocation from Small to Large LLMs [EMNLP 2026]

This repository contains the official implementation of the EMNLP 2026 paper
**[Scaling Near-Optimal SFT-RL Annotation Budget Allocation from Small to Large LLMs](https://arxiv.org/abs/2609.01573)**.

## Overview

We study how to allocate a fixed annotation budget between supervised fine-tuning (SFT) and reinforcement learning (RL) during LLM post-training. Instead of selecting a single optimal allocation, the code supports experiments that characterize a near-optimal region and test whether that region transfers from small proxy models to larger target models.

The released implementation includes:

- SFT baselines and SFT warm starts.
- Preference-based off-policy training with DPO and KTO.
- Reward-supervision on-policy training with GRPO/RLVR and colocated vLLM generation.
- GSM8K mathematical reasoning and TuluIF/IFEval instruction-following experiments.
- Llama 3 and Qwen 2.5 model configurations.

This implementation builds on the public codebase for [*Balancing the Budget: Understanding Trade-offs Between Supervised and Preference-Based Finetuning*](https://arxiv.org/abs/2502.11284).

## Environment setup

Create a Python 3.10 environment and install the pinned dependencies:

```bash
conda create --name balance-budget python=3.10 -y
conda activate balance-budget
python -m pip install -r requirements.txt
```

Before importing Unsloth or starting GRPO/RLVR, apply and verify the fixes required by the pinned Unsloth implementation:

```bash
python apply_unsloth_grpo_patches.py
python apply_unsloth_grpo_patches.py --check
```

Run these commands from the repository root. The patch command is idempotent. The verification command should report every source patch as `already applied`; if no compiled trainer has been generated yet, it may also report `compiled trainer cache: not present; source patches will affect generation`.

### IFEval dependency

Instruction-following evaluation uses the [`instruction_following_eval`](https://github.com/google-research/google-research/tree/master/instruction_following_eval) package from Google Research. Copy or symlink that directory into the repository root so the following path exists:

```text
./instruction_following_eval/evaluation_main.py
```

### Output location

Choose a writable output directory:

```bash
export TUNING_SAVE_DIR=/path/to/your/output
```

Models are saved under `$TUNING_SAVE_DIR/models`. Use the same environment variable for training, inference, and evaluation. The default `/PROJECT_ROOT` value in `tuning/config.py` is a placeholder and should be overridden.

## Data preparation

Prepare the SFT, preference, and RLVR datasets for GSM8K and TuluIF:

```bash
bash tuning/data_processing.sh
```

The processed datasets are written under `tuning/data/datasets/`.

## Quick start

### SFT and preference-based RL

Edit the `train_sizes` list in `tuning/run.sh`, then launch the interactive runner:

```bash
bash tuning/run.sh
```

The runner prompts for the task, dataset, SFT ratio, base model, preference-training method, and whether to run training, inference, and evaluation. Available preference methods are DPO and KTO.

Individual stages can also be invoked directly. For example:

```bash
python tuning/run_sft.py \
  --model llama3-8B \
  --dataset gsm8k \
  --train_size 1000 \
  --task math \
  --do_training \
  --do_inference \
  --do_evaluation
```

### GRPO/RLVR

Run reward-supervision training on GSM8K with:

```bash
python tuning/run_rlvr.py \
  --model llama3-8B \
  --dataset gsm8k \
  --train_size 1000 \
  --task math \
  --reward_toolkit gsm8k \
  --do_training \
  --do_inference \
  --do_evaluation
```

For TuluIF/IFEval:

```bash
python tuning/run_rlvr.py \
  --model llama3-8B \
  --dataset tuluif \
  --train_size 1000 \
  --task instruction \
  --num_train_epochs 2 \
  --reward_toolkit ifeval \
  --do_training \
  --do_inference \
  --do_evaluation
```

TuluIF runs default to two training epochs when `--num_train_epochs` is omitted, matching the paper configuration. An explicit value always overrides this default.

Use `--data_seed 41` or `--data_seed 42` to select the sampled training split. Add `--sft_train_size <size>` only when the corresponding SFT checkpoint already exists under the same `$TUNING_SAVE_DIR/models` directory.

## Code structure

- `tuning/data/`: dataset construction for SFT, preference training, and RLVR.
- `tuning/training/`: SFT, DPO, KTO, and GRPO/RLVR training implementations and configurations.
- `tuning/inference/`: model inference and vLLM utilities.
- `tuning/evaluation/`: GSM8K and IFEval evaluation.
- `tuning/run_sft.py`, `tuning/run_dpo.py`, and `tuning/run_rlvr.py`: direct experiment entry points.
- `tuning/run.sh`: interactive SFT/preference-allocation runner.
- `tuning/collections.yaml`: supported tasks, models, allocation ratios, and preference methods.
- `apply_unsloth_grpo_patches.py`: reproducible compatibility fixes for the pinned Unsloth GRPO/vLLM environment.
- `check_prompts.py`: utility for checking prompt alignment between preference and RLVR datasets.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{wang2026scaling,
  title     = {Scaling Near-Optimal SFT-RL Annotation Budget Allocation from Small to Large LLMs},
  author    = {Wang, Jingtan and Verma, Arun and Lin, Xiaoqiang and Liu, Zhengyuan and Chen, Nancy F. and Rus, Daniela and Low, Bryan Kian Hsiang},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```
