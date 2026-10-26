from datasets import Dataset
from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_GSM8K

GSM8K_STRING = """Question: {question}\nAnswer:"""

class GSM8KSFT(HFDataset):
    def __init__(self):
        super().__init__("gsm8k")

    def _make_msg(self, example):

        prompt = GSM8K_STRING.format(question=example["prompt"])
        return {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE_GSM8K,
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": example["correct"][0]}
            ]
        }

    def _get_rows(seld, dataset):
        prompts = dataset["prompt"]
        corrects = dataset["correct"]
        reference_answers = dataset["reference_answer"]

        rows = []
        for prompt, correct_set, ref in zip(prompts, corrects, reference_answers):
            correct = correct_set
            prompt = GSM8K_STRING.format(question=prompt)
            for correct in correct_set:
                rows.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_MESSAGE_GSM8K,
                        },
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": correct}
                    ]
                })

        return rows
    
    def format_dataset(self):        
        rows = self._get_rows(self._dataset)
        formatted_dataset = Dataset.from_list(rows).train_test_split(test_size=200, shuffle=False)
        print(f"Dataset - {formatted_dataset}")
        print(f'Example row - {formatted_dataset["train"][57]}')
        print(f'Example row - {formatted_dataset["train"][5]}')
        print(f'Example row - {formatted_dataset["train"][27]}')
        self._dataset = formatted_dataset
        
if __name__ == "__main__":
    gsm8k = GSM8KSFT()
    gsm8k.load_from_huggingface("mohit-raghavendra/gsm8k-gpt4ogenerations", split="train")
    gsm8k.format_dataset()
    gsm8k.clear_old_datasets(prefix="sft-gsm8k")
    gsm8k.save_dataset_to_disk(save_name="sft-gsm8k")

    
