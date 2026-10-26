import json
import wandb
from unsloth import FastLanguageModel, is_bfloat16_supported, PatchDPOTrainer
from tuning.config import MODELS_DIR
from tuning.data.train_dataset import get_train_dataset
from tuning.training.config_training import PTRunConfig, LoraConfig, ModelLoadConfig, DatasetConfig, DPOTrainingConfig, dpo_batch_size, effective_batch_size
from tuning.utils.utils import apply_chat_template_pt, chat_template_func
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer
import pprint

PatchDPOTrainer()

import torch
import gc

def train_model_dpo(
        run_config: PTRunConfig = None,
        lora_config: LoraConfig = None,
        model_load_config: ModelLoadConfig = None,
        training_args: DPOTrainingConfig = None,
):

    train_batch_size = dpo_batch_size(run_config.dataset_config.train_size)
    gradient_accumulation_steps = effective_batch_size(run_config.dataset_config.train_size) // train_batch_size

    print(f"Per device train batch size: {train_batch_size}")

    dataset = get_train_dataset(run_config)
    
    if run_config.sft_run_config:
        # model_path = f"{MODELS_DIR}/{run_config.sft_run_config.run_name}"
        model_path = run_config.sft_run_config.output_dir
    else:
        model_path = run_config.model_name_hf

    print(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = model_load_config.max_seq_length,
        dtype = model_load_config.dtype,
        load_in_4bit = model_load_config.load_in_4bit,
    )

    dataset = apply_chat_template_pt(tokenizer, dataset)

    pprint.pprint(dataset["train"][0])
    pprint.pprint(dataset)

    if run_config.model_name == "qwen2-7B":
        lora_config.target_modules.extend(["embed_tokens", "lm_head"])

    print(f"Using LORA with config: {lora_config.target_modules}")
    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_config.r,
        target_modules = lora_config.target_modules,
        lora_alpha = lora_config.lora_alpha, 
        lora_dropout = lora_config.lora_dropout,
        bias = lora_config.bias,
        use_gradient_checkpointing = lora_config.use_gradient_checkpointing,
        random_state = lora_config.random_state, 
        use_rslora = lora_config.use_rslora,
        loftq_config = lora_config.loftq_config,
    )

    print(f"Model loaded - {type(model)}")

    # Print memory usage
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    torch.cuda.empty_cache()
    gc.collect()

    trainer = DPOTrainer(
        model = model,
        ref_model = None,
        tokenizer = chat_template_func(tokenizer),
        beta = training_args.beta,
        train_dataset = dataset["train"],
        eval_dataset = dataset["test"],
        max_length = model_load_config.max_seq_length,
        args = DPOConfig(
            per_device_train_batch_size = train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,

            per_device_eval_batch_size = training_args.per_device_eval_batch_size,
            warmup_ratio = training_args.warmup_ratio,
            num_train_epochs = training_args.num_train_epochs,
            learning_rate = training_args.learning_rate,
            do_eval = training_args.do_eval,
            eval_strategy = training_args.eval_strategy,
            eval_steps = training_args.eval_steps,
            optim = training_args.optim,
            weight_decay = training_args.weight_decay,
            lr_scheduler_type = training_args.lr_scheduler_type,
            report_to = training_args.report_to,
            logging_steps = training_args.logging_steps,

            output_dir = run_config.output_dir,
            save_strategy = "no",

            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            seed = 42,
        )
    )

    trainer_stats = trainer.train()
    model.save_pretrained_merged(run_config.output_dir, tokenizer, save_method = "merged_16bit")

    args = trainer.args.to_dict()
    with open(f"{run_config.output_dir}/training_config.json", "w") as f:
        json.dump(args, f, indent=4)    

if __name__ == "__main__":
    dataset_config = DatasetConfig(
        dataset = "tuluif",
        dataset_type = "pt",
        train_size = 10000
    )

    print(dataset_config)



    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    training_args = DPOTrainingConfig()

    run_config = PTRunConfig(
        dataset_config = dataset_config,
        model_name_hf = "unsloth/Meta-Llama-3.1-8B",
        model_name = "llama3-8B",
        do_training=True,
        do_inference=False,
        do_evaluation=False,
    )
    run = wandb.init(name=run_config.run_name, project="tuning", reinit=True)

    print(run_config)
    with run:
        train_model_dpo(
            run_config = run_config,
            lora_config = lora_config,
            model_load_config = model_load_config,
            training_args = training_args,
        )
        
