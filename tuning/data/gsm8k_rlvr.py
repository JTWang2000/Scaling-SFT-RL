from datasets import Dataset, DatasetDict

from tuning.data.config import SYSTEM_MESSAGE_GSM8K
from tuning.data.hf_dataset import HFDataset


class GSM8KRLVR(HFDataset):
    def __init__(self):
        super().__init__(dataset_name="gsm8k")

    def _normalize_row(self, example):
        prompt = example.get("prompt") or example.get("question") or ""
        reference_answer = (
            example.get("reference_answer")
            or example.get("reference_solution")
            or example.get("answer")
            or ""
        )
        chosen = example.get("chosen") or ""
        rejected = example.get("rejected") or ""

        if not isinstance(prompt, str) or not prompt.strip():
            return None
        if not isinstance(reference_answer, str) or not reference_answer.strip():
            return None

        return {
            "system_message": SYSTEM_MESSAGE_GSM8K,
            "prompt": prompt,
            "reference_answer": reference_answer,
            "chosen": chosen,
            "rejected": rejected,
        }

    def format_dataset(self):
        if isinstance(self._dataset, DatasetDict):
            if "train" in self._dataset:
                source = self._dataset["train"]
            else:
                source = self._dataset[next(iter(self._dataset.keys()))]
        else:
            source = self._dataset

        rows = [self._normalize_row(example) for example in source]
        rows = [row for row in rows if row is not None]

        if not rows:
            raise ValueError("No GSM8K RLVR rows produced after normalization. Check source schema.")

        formatted = Dataset.from_list(rows)
        test_size = 200 if len(formatted) > 200 else max(1, len(formatted) // 10)
        formatted_dataset = formatted.train_test_split(test_size=test_size, shuffle=False)

        print(f"GSM8K rlvr dataset - {formatted_dataset}")
        if len(formatted_dataset["train"]):
            print(f'Example GSM8K rlvr row - {formatted_dataset["train"][0]}')
        print("***")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    gsm8k = GSM8KRLVR()
    gsm8k.load_from_huggingface("mohit-raghavendra/gsm8k-gpt4opreferences", split="train")
    gsm8k.format_dataset()
    gsm8k.clear_old_datasets(prefix="rlvr-gsm8k")
    gsm8k.save_dataset_to_disk(save_name="rlvr-gsm8k")
