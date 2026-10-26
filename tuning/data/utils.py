from datasets import DatasetDict, concatenate_datasets
import random

random.seed(42)


def get_random_train_subset(dataset: DatasetDict, train_size: int, seed: int = 42) -> DatasetDict:

    train_split = dataset["train"]
    test_split = dataset["test"]
    rng = random.Random(seed)

    n_train = len(train_split)
    if train_size <= n_train:
        random_subset = rng.sample(range(n_train), train_size)
        train_split = train_split.select(random_subset)
    else:
        full_repeats = train_size // n_train
        remainder = train_size % n_train

        repeated_splits = []
        for _ in range(full_repeats):
            full_subset = rng.sample(range(n_train), n_train)
            repeated_splits.append(train_split.select(full_subset))
        if remainder > 0:
            remainder_subset = rng.sample(range(n_train), remainder)
            repeated_splits.append(train_split.select(remainder_subset))

        train_split = concatenate_datasets(repeated_splits)

    sampled_dataset = DatasetDict()

    sampled_dataset["train"] = train_split
    sampled_dataset["test"] = test_split

    return sampled_dataset
