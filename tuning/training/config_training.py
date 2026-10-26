from enum import Enum
from pydantic import BaseModel, Field
from tuning.config import MODELS_DIR
from typing import Optional

BaseModel.model_config['protected_namespaces'] = ()
import torch

def get_gpu_memory_gb(device=0):
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / (1024 ** 3)


def is_large_gpu(mem_gb):
    return mem_gb >= 120   # e.g., H200 / 140GB+


def sft_batch_size(dataset_size: int):
    mem_gb = get_gpu_memory_gb()
    return 16 if is_large_gpu(mem_gb) else 8


def dpo_batch_size(dataset_size: int):
    mem_gb = get_gpu_memory_gb()
    return 16 if is_large_gpu(mem_gb) else 8

def rlvr_batch_size(dataset_size: int):
    mem_gb = get_gpu_memory_gb()
    return 16 if is_large_gpu(mem_gb) else 8

def effective_batch_size(dataset_size: int):
    return 16

class ModelLoadConfig(BaseModel):
    max_seq_length: int = 1024
    dtype: str = None 
    load_in_4bit: bool = False 

class LoraConfig(BaseModel):
    r: int = 32
    target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",]
    lora_alpha: int = 32
    lora_dropout: int = 0
    bias: str = "none"
    use_gradient_checkpointing: bool = False
    random_state: int = 42
    use_rslora: bool = False
    loftq_config: str = None

class TrainingArgumentsConfig(BaseModel):
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    eval_steps: float = 0.5
    logging_steps: int = 1
    do_eval: bool = True
    eval_strategy: str = "steps"
    gradient_accumulation_steps: int = 4
    warmup_ratio: int = 0.1
    num_train_epochs: int = 1
    learning_rate: float = 1e-5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    report_to: list[str] = ["wandb"]


class DPOTrainingConfig(TrainingArgumentsConfig):
    beta: float = 0.1
    learning_rate: float = 1e-6
    num_train_epochs: int = 1
    per_device_eval_batch_size: int = 2


class RLVRTrainingConfig(TrainingArgumentsConfig):
    learning_rate: float = 1e-6
    num_train_epochs: int = 1
    per_device_eval_batch_size: int = 2
    max_prompt_length: int = 1024
    max_completion_length: int = 512
    num_generations: int = 4
    use_loose_verifier: bool = True
    use_vllm: bool = True
    fast_inference: bool = True
    max_lora_rank: int = 32
    gpu_memory_utilization: float = 0.3
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    reward_toolkit: str = "auto"
    do_eval: bool = False
    eval_strategy: str = "no"


class DatasetConfig(BaseModel):
    dataset: str = "gsm8k"
    dataset_type: str = "sft"
    train_size: int = 100
    data_seed: int = 42

    @property
    def dataset_full_name(self):
        if not self.train_size:
            return f"{self.dataset_type}-{self.dataset}"
        name = f"{self.dataset_type}-{self.dataset}-{self.train_size}"
        if self.data_seed != 42:
            name = f"{name}-seed-{self.data_seed}"
        return name
    
    def __str__(self):
        return self.dataset_full_name

    
class SFTRunConfig(BaseModel):
    model_name_hf: str = "unsloth/Meta-Llama-3.1-8B"
    dataset_config: Optional[DatasetConfig] = None
    model_name: str = "llama3-8B"
    task_name: str = "math"
    run_type: str = "sft"
    do_training: bool = False
    do_inference: bool = False
    do_evaluation: bool = False
    output_dir_base: Optional[str] = Field(default=None, alias="output_dir")
    
    
    @property
    def run_name(self):
        if not self.dataset_config or not self.dataset_config.train_size:
            return self.model_name
        return f"{self.model_name}_{self.dataset_config.dataset_full_name}"
    
    @property
    def output_dir(self):
        if self.output_dir_base is not None:
            return f"{self.output_dir_base}/{self.run_name}"
        return f"{MODELS_DIR}/{self.run_name}"
    
    def __str__(self):
        return self.run_name

    model_config = {"populate_by_name": True}
    

class PTRunConfig(BaseModel):
    model_name_hf: str = "unsloth/Meta-Llama-3.1-8B"
    model_name: str = "llama3-8B"
    dataset_config: DatasetConfig = None
    sft_run_config: Optional[SFTRunConfig] = None
    run_type: str = "pt"
    task_name: str = "math"
    do_training: bool = False
    do_inference: bool = False
    do_evaluation: bool = False
    pft_method: str = "dpo"
    reward_toolkit: str = "auto"
    add_beta_run_name: bool = False
    beta: float = 0.1
    output_dir_base: Optional[str] = Field(default=None, alias="output_dir")

    @property
    def run_name(self):
        run_name = self.model_name
        if self.sft_run_config:
            run_name = self.sft_run_config.run_name
        if self.dataset_config:
            run_name = f"{run_name}_{self.dataset_config.dataset_full_name}"
        if self.pft_method == "kto":
            run_name = f"{run_name}_{self.pft_method}"
        if self.add_beta_run_name:
            run_name = f"{run_name}_beta-{self.beta}"

            run_name = run_name.replace(".", "-")   
        return run_name
    
    @property
    def output_dir(self):
        if self.output_dir_base is not None:
            return f"{self.output_dir_base}/{self.run_name}"
        return f"{MODELS_DIR}/{self.run_name}"
    
    def __str__(self):
        return self.run_name
    model_config = {"populate_by_name": True}
