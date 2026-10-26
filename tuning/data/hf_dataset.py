import logging
import math
import random
import os
import shutil
from abc import ABC, abstractmethod

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from tuning.config import DATASETS_DIR


random.seed(42)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HFDataset(ABC):
    def __init__(self, dataset_name: str):
        self._dataset_name = dataset_name

    def load_from_huggingface(self, hf_path: str, *args, **kwargs):
        self.hf_path = hf_path
        self._dataset = load_dataset(self.hf_path, *args, **kwargs)
        self._raw_dataset = self._dataset

    def load_from_huggingface_disk(self, local_path: str):
        full_path = f"{DATASETS_DIR}/{local_path}"
        self._hf_path = full_path
        self._dataset = load_from_disk(dataset_path=full_path)
        self._raw_dataset = self._dataset

    def load_from_disk(self, disk_path: str, filetype: str = "csv", delimiter: str = "\t"):
        self._disk_path = disk_path
        self._dataset = load_dataset(filetype, data_files=self._disk_path, delimiter=delimiter)

    def get_dataset(self):
        return self._dataset

    def filter_columns(self):
        raise NotImplementedError
    
    def clear_old_datasets(self, prefix: str):
        # List all items in the datasets directory
        try:
            items = os.listdir(DATASETS_DIR)

            # Iterate through the items and remove directories with the specified prefix
            for item in items:
                item_path = os.path.join(DATASETS_DIR, item)
                if os.path.isdir(item_path) and item.startswith(prefix):
                    shutil.rmtree(item_path)
                    print(f"Removed directory: {item_path}")
        except:
            print("No datasets to remove")


    @abstractmethod
    def format_dataset(self):
        raise NotImplementedError

    def save_dataset_to_disk(self, dataset: DatasetDict = None, save_name=None):

        if not dataset:
            dataset = self._dataset

        if not save_name:
            save_name = self._dataset_name

        path = f'{DATASETS_DIR}/{save_name}'

        logger.info(f"Saving dataset to {path}")

        print(dataset)

        dataset.save_to_disk(path)

    def sample_dataset(
        self, sample_column: str = None, n=1000, sampled_dataset_name: str = None
    ) -> DatasetDict:

        train_split = self._dataset["train"]
        test_split = self._dataset["test"]

        if not sampled_dataset_name:
            sampled_dataset_name = f"{self._dataset_name}{n}"

        if not sample_column:
            random_subset = random.sample(range(len(train_split)), n)
            train_split = train_split.select(random_subset)

        else:
            ds_df = train_split.to_pandas()

            logger.info("Before sampling")
            logger.info(ds_df.columns)
            logger.info(ds_df[sample_column].value_counts())

            n_categories = len(ds_df[sample_column].unique())
            sample_amount = math.ceil(n / n_categories)

            sampled_df = (
                ds_df.groupby(sample_column)
                .apply(lambda x: x.sample(n=sample_amount) if len(x) > sample_amount else x)
                .reset_index(drop=True)
            )

            logger.info("After sampling")
            logger.info(sampled_df.columns)
            logger.info(sampled_df[sample_column].value_counts())

            train_split = Dataset.from_pandas(sampled_df)

            train_split.remove_columns([sample_column])
            test_split.remove_columns([sample_column])

        sampled_dataset = DatasetDict()

        sampled_dataset["train"] = train_split
        sampled_dataset["test"] = test_split
        logger.info(sampled_dataset)

        self.save_dataset_to_disk(sampled_dataset, save_name=sampled_dataset_name)

        return sampled_dataset

    def combine_with_dataset(self, name_2: str, combined_name: str = None) -> DatasetDict:

        dataset_1 = self._dataset
        dataset_2 = load_from_disk(f'{DATASETS_DIR}/{name_2}')

        logger.info(dataset_1)
        logger.info(dataset_2)
        final_dataset = DatasetDict()
        final_dataset["train"] = concatenate_datasets([dataset_1["train"], dataset_2["train"]])
        final_dataset["test"] = concatenate_datasets([dataset_1["test"], dataset_2["test"]])

        logger.info(f"Final dataset - {final_dataset}")

        if not combined_name:
            combined_name = f"{self._dataset_name}-{name_2}"

        logger.info(f'Saving dataset to {DATASETS_DIR}/{combined_name}')
        final_dataset.save_to_disk(f'{DATASETS_DIR}/{combined_name}')

        return final_dataset
