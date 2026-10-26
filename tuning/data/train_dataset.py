from typing import Union
from tuning.training.config_training import DatasetConfig, PTRunConfig, SFTRunConfig
from tuning.data.utils import get_random_train_subset
from datasets import load_from_disk, DatasetDict
from tuning.config import DATASETS_DIR
from pathlib import Path

def get_train_dataset(run_config: Union[PTRunConfig, SFTRunConfig]) -> DatasetDict:

    print(f"Getting train dataset for run config: {run_config}")

    if run_config.run_type == "pt":

        # if run_config.sft_run_config is None:
        #     dataset_stub = f"pt-{run_config.dataset_config.dataset}"
        # else:
        #     sft_run_name = run_config.sft_run_config.run_name
        #     sft_run_name = sft_run_name.replace("_", "|").replace("-", "|")
        #     dataset_stub = f"pt-{sft_run_name}"

        dataset_stub = f"pt-{run_config.dataset_config.dataset}"
    elif run_config.run_type == "sft":
        dataset_stub = f"sft-{run_config.dataset_config.dataset}"
    elif run_config.run_type == "rlvr":
        dataset_stub = f"rlvr-{run_config.dataset_config.dataset}"
    else:
        raise ValueError(f"Unsupported run type: {run_config.run_type}")
    
    save_name = run_config.dataset_config.dataset_full_name
    check_path = f"{DATASETS_DIR}/{save_name}"

    print(f"Checking for dataset at {check_path}")
    if Path(check_path).exists():
        print(f"Dataset already exists at {check_path}")
        return load_from_disk(check_path)

    train_size = run_config.dataset_config.train_size
    
    full_dataset_path = f"{DATASETS_DIR}/{dataset_stub}"
    print(f"Loading dataset from {full_dataset_path}")
    full_dataset = load_from_disk(full_dataset_path)
    print(f"Full dataset: {full_dataset}")

    sampled_dataset = get_random_train_subset(
        full_dataset,
        train_size,
        seed=run_config.dataset_config.data_seed,
    )
    print(f"Sampled dataset: {sampled_dataset}")
    print(f"Example row: {sampled_dataset['train'][0]}")

    sampled_dataset.save_to_disk(f"{DATASETS_DIR}/{save_name}")

    return sampled_dataset

if __name__ == "__main__":

    run_config = SFTRunConfig(
            model_name="llama3-8B",
            task_name="math",
            dataset_config=DatasetConfig(
                dataset="gsm8k",
                dataset_type="sft",
                train_size=10000,
            ),
            do_training=True,
        )

    dataset = get_train_dataset(run_config=run_config)

    print(f"Dataset: {dataset}")
