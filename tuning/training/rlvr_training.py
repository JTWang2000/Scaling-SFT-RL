import gc
import importlib
import json
import os
import re

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported

from instruction_following_eval.evaluation_lib import (
    InputExample,
    test_instruction_following_loose,
    test_instruction_following_strict,
)
from instruction_following_eval import instructions_registry
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import (
    LoraConfig,
    ModelLoadConfig,
    PTRunConfig,
    RLVRTrainingConfig,
    effective_batch_size,
    rlvr_batch_size,
)
from tuning.utils.utils import chat_template_func

try:
    from unsloth import PatchFastRL
except Exception:
    PatchFastRL = None

def _load_grpo_classes(enable_unsloth_patch: bool):
    if enable_unsloth_patch:
        if PatchFastRL is None:
            return None, None, ImportError("Unsloth PatchFastRL is unavailable.")
        PatchFastRL("GRPO", FastLanguageModel)
    try:
        trl_mod = importlib.import_module("trl")
        return trl_mod.GRPOConfig, trl_mod.GRPOTrainer, None
    except Exception as import_error:
        return None, None, import_error


def _coerce_instruction_kwargs(instruction_id_list, kwargs):
    if isinstance(instruction_id_list, str):
        instruction_id_list = [instruction_id_list]
    if not isinstance(instruction_id_list, list):
        instruction_id_list = []

    if isinstance(kwargs, dict):
        kwargs = [kwargs]
    if not isinstance(kwargs, list):
        kwargs = []

    cleaned_kwargs = []
    for item in kwargs:
        if not isinstance(item, dict):
            cleaned_kwargs.append({})
            continue
        cleaned_kwargs.append({k: v for k, v in item.items() if v is not None})
    kwargs = cleaned_kwargs

    if len(kwargs) < len(instruction_id_list):
        kwargs = kwargs + [{} for _ in range(len(instruction_id_list) - len(kwargs))]
    elif len(kwargs) > len(instruction_id_list):
        kwargs = kwargs[: len(instruction_id_list)]

    filtered_kwargs = []
    for instruction_id, item in zip(instruction_id_list, kwargs):
        if not isinstance(item, dict):
            filtered_kwargs.append({})
            continue
        instruction_cls = instructions_registry.INSTRUCTION_DICT.get(instruction_id)
        if instruction_cls is None:
            filtered_kwargs.append(item)
            continue
        try:
            allowed_keys = instruction_cls(instruction_id).get_instruction_args_keys() or []
        except Exception:
            filtered_kwargs.append(item)
            continue
        filtered_kwargs.append({k: v for k, v in item.items() if k in allowed_keys})
    kwargs = filtered_kwargs

    return instruction_id_list, kwargs


def _completion_to_text(completion):
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        return completion.get("content") or completion.get("text") or ""

    if isinstance(completion, list):
        texts = []
        for chunk in completion:
            if isinstance(chunk, dict):
                text = chunk.get("content") or chunk.get("text")
                if text:
                    texts.append(text)
            elif isinstance(chunk, str):
                texts.append(chunk)
        return "\n".join(texts)

    return str(completion)


def _format_prompt_with_template(tokenizer, dataset):
    tokenizer = chat_template_func(tokenizer)

    def _format(example):
        example["raw_prompt"] = example["prompt"]
        message = [
            {"role": "system", "content": example["system_message"]},
            {"role": "user", "content": example["prompt"]},
        ]
        example["prompt"] = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
        )
        return example

    return dataset.map(_format, batched=False), tokenizer


def _instruction_following_reward(prompt, completion, instruction_id_list, kwargs, use_loose=True):
    instruction_id_list, kwargs = _coerce_instruction_kwargs(instruction_id_list, kwargs)

    if not instruction_id_list:
        return 0.0

    example = InputExample(
        key=0,
        instruction_id_list=instruction_id_list,
        prompt=prompt,
        kwargs=kwargs,
    )

    prompt_to_response = {prompt: completion}
    if use_loose:
        result = test_instruction_following_loose(example, prompt_to_response)
    else:
        result = test_instruction_following_strict(example, prompt_to_response)

    return float(sum(result.follow_instruction_list) / len(result.follow_instruction_list))


def _extract_gsm8k_final_answer(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()

    boxed_matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    hash_matches = re.findall(r"####\s*([^\n]+)", text)
    if hash_matches:
        return hash_matches[-1].strip()

    number_matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if number_matches:
        return number_matches[-1].strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _normalize_answer(ans):
    if ans is None:
        return ""
    ans = str(ans).strip()
    ans = re.sub(r"\s+", " ", ans)
    ans = ans.replace(",", "")
    ans = ans.strip().lower()
    return ans


def _gsm8k_reward(completion, reference_answer):
    pred = _normalize_answer(_extract_gsm8k_final_answer(completion))
    ref = _normalize_answer(_extract_gsm8k_final_answer(reference_answer))
    return 1.0 if pred and ref and pred == ref else 0.0


def train_model_rlvr(
    run_config: PTRunConfig = None,
    lora_config: LoraConfig = None,
    model_load_config: ModelLoadConfig = None,
    training_args: RLVRTrainingConfig = None,
):
    # This repository's GRPO/vLLM path was run with Unsloth's patched trainer.
    # Set USE_UNSLOTH_GRPO_PATCH=0 only to debug the vanilla TRL implementation.
    use_unsloth_grpo_patch_env = os.getenv("USE_UNSLOTH_GRPO_PATCH", "1") == "1"
    enable_unsloth_grpo_patch = use_unsloth_grpo_patch_env

    GRPOConfig, GRPOTrainer, grpo_import_error = _load_grpo_classes(enable_unsloth_grpo_patch)
    if GRPOTrainer is None or GRPOConfig is None:
        raise ImportError(
            "GRPOTrainer is unavailable. Install a TRL version that supports GRPO (for example, >=0.12). "
            f"Original import error: {grpo_import_error}"
        )

    train_batch_size = max(rlvr_batch_size(run_config.dataset_config.train_size), training_args.num_generations)
    gradient_accumulation_steps = effective_batch_size(run_config.dataset_config.train_size) // train_batch_size
    print(f"Per device train batch size: {train_batch_size}")

    world_size = torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1
    global_eval_batch_size = training_args.per_device_eval_batch_size * world_size
    if global_eval_batch_size % training_args.num_generations != 0:
        adjusted_global_eval_batch_size = training_args.num_generations
        while adjusted_global_eval_batch_size % world_size != 0:
            adjusted_global_eval_batch_size += training_args.num_generations
        adjusted_per_device_eval_batch_size = adjusted_global_eval_batch_size // world_size
        print(
            "Adjusting per-device eval batch size from "
            f"{training_args.per_device_eval_batch_size} to {adjusted_per_device_eval_batch_size} "
            f"so global eval batch size is divisible by num_generations={training_args.num_generations}."
        )
        training_args.per_device_eval_batch_size = adjusted_per_device_eval_batch_size

    dataset = get_train_dataset(run_config)

    if run_config.sft_run_config:
        model_path = run_config.sft_run_config.output_dir
    else:
        model_path = run_config.model_name_hf

    effective_fast_inference = bool(training_args.fast_inference)
    effective_use_vllm = bool(training_args.use_vllm)

    print(f"Loading model from {model_path}")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=model_load_config.max_seq_length,
            dtype=model_load_config.dtype,
            load_in_4bit=model_load_config.load_in_4bit,
            fast_inference=effective_fast_inference,
            max_lora_rank=training_args.max_lora_rank,
            gpu_memory_utilization=training_args.gpu_memory_utilization,
        )
    except TypeError:
        # Backward compatibility with older Unsloth signatures.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=model_load_config.max_seq_length,
            dtype=model_load_config.dtype,
            load_in_4bit=model_load_config.load_in_4bit,
        )

    dataset, tokenizer = _format_prompt_with_template(tokenizer, dataset)
    print(dataset)
    print(dataset["train"][0])

    if run_config.model_name == "qwen2-7B":
        lora_config.target_modules.extend(["embed_tokens", "lm_head"])

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config.r,
        target_modules=lora_config.target_modules,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
        random_state=lora_config.random_state,
        use_rslora=lora_config.use_rslora,
        loftq_config=lora_config.loftq_config,
    )

    print(f"Model loaded - {type(model)}")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    torch.cuda.empty_cache()
    gc.collect()

    def reward_func(prompts, completions, instruction_id_list=None, kwargs=None, **extra_inputs):
        rewards = []
        ids_batch = instruction_id_list if instruction_id_list is not None else [None] * len(prompts)
        kwargs_batch = kwargs if kwargs is not None else [None] * len(prompts)
        refs_batch = extra_inputs.get("reference_answer")
        if refs_batch is None:
            refs_batch = [None] * len(prompts)
        raw_prompts_batch = extra_inputs.get("raw_prompt")
        if raw_prompts_batch is None:
            raw_prompts_batch = prompts

        reward_toolkit = training_args.reward_toolkit
        if reward_toolkit == "auto":
            dataset_name = (run_config.dataset_config.dataset or "").lower()
            task_name = (run_config.task_name or "").lower()
            if dataset_name == "gsm8k" or task_name == "math":
                reward_toolkit = "gsm8k"
            elif dataset_name == "tuluif" or task_name == "instruction":
                reward_toolkit = "ifeval"
            else:
                raise ValueError(f"No RLVR reward toolkit for dataset={dataset_name!r}, task={task_name!r}")

        for prompt, raw_prompt, completion, ids, kw, ref in zip(
            prompts,
            raw_prompts_batch,
            completions,
            ids_batch,
            kwargs_batch,
            refs_batch,
        ):
            completion_text = _completion_to_text(completion)
            if reward_toolkit == "gsm8k":
                reward = _gsm8k_reward(completion_text, ref)
            elif reward_toolkit == "ifeval":
                reward = _instruction_following_reward(
                    prompt=raw_prompt,
                    completion=completion_text,
                    instruction_id_list=ids,
                    kwargs=kw,
                    use_loose=training_args.use_loose_verifier,
                )
            else:
                raise ValueError(f"Unsupported RLVR reward toolkit: {reward_toolkit}")
            rewards.append(reward)

        return rewards

    grpo_config = GRPOConfig(
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        warmup_ratio=training_args.warmup_ratio,
        num_train_epochs=training_args.num_train_epochs,
        learning_rate=training_args.learning_rate,
        do_eval=training_args.do_eval,
        eval_strategy=training_args.eval_strategy,
        eval_steps=training_args.eval_steps,
        optim=training_args.optim,
        weight_decay=training_args.weight_decay,
        lr_scheduler_type=training_args.lr_scheduler_type,
        report_to=training_args.report_to,
        logging_steps=training_args.logging_steps,
        output_dir=run_config.output_dir,
        save_strategy="no",
        max_prompt_length=training_args.max_prompt_length,
        max_completion_length=training_args.max_completion_length,
        num_generations=training_args.num_generations,
        use_vllm=effective_use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=training_args.gpu_memory_utilization,
        temperature=training_args.temperature,
        top_p=training_args.top_p,
        top_k=training_args.top_k,
        min_p=training_args.min_p,
        repetition_penalty=training_args.repetition_penalty,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        seed=42,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        args=grpo_config,
    )

    trainer.train()

    model.save_pretrained_merged(run_config.output_dir, tokenizer, save_method="merged_16bit")

    args = trainer.args.to_dict()
    with open(f"{run_config.output_dir}/training_config.json", "w") as f:
        json.dump(args, f, indent=4)
