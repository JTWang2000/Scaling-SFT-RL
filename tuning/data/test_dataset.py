from datasets import Dataset
from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING
import random
import json

random.seed(42)
RESAMPLE = False

def random_subset(dataset, n=1000):
    random_subset = random.sample(range(len(dataset)), n)
    return dataset.select(random_subset)

def get_ifeval_test_dataset():
    with open("instruction_following_eval/data/input_data.jsonl", "r") as f:
        ifeval_prompts = [json.loads(line) for line in f]

    messages = [
        [
            {"role": "system", "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING},
            {"role": "user", "content": prompt["prompt"]},
        ]
        for prompt in ifeval_prompts
    ]

    prompts = [prompt["prompt"] for prompt in ifeval_prompts]
    dataset = Dataset.from_dict({"messages": messages, "prompt": prompts})
    return dataset
