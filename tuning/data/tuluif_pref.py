from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING

class TuluIFPT(HFDataset):
    def __init__(self):
        super().__init__(dataset_name="conifer")

    def _get_messages(self, examples):
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        prompt = examples["prompt"]

        return {
            "system_message": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING,
            "prompt": prompt,
            "chosen": chosen[-1]["content"],
            "rejected": rejected[-1]["content"]
        } 
    
    def _filter_long(self, examples):
        keep_chosen = len(examples["prompt"].split(" ")) + len(examples["chosen"].split(" ")) + len(examples["system_message"].split(" ")) < 1024
        keep_rejected = len(examples["prompt"].split(" ")) + len(examples["rejected"].split(" ")) + len(examples["system_message"].split(" ")) < 1024

        return keep_chosen and keep_rejected

    def format_dataset(self):

        formatted_dataset = self._dataset["train"].map(self._get_messages)
        formatted_dataset = formatted_dataset.filter(self._filter_long)
        print(f"Tuluif sft dataset - {formatted_dataset}")
        print(f'Example Tuluif sft row - {formatted_dataset[0]}')
        print("***")
        self._dataset = formatted_dataset.train_test_split(test_size=200, shuffle=False)

        dataset = self._dataset["test"]
        longest = 0
        longest_prompt = ""
        longest_response = ""
        for row in dataset:
            total_len_c = len(row["prompt"].split(" ")) + len(row["chosen"].split(" ")) + len(row["system_message"].split(" "))
            total_len_r = len(row["prompt"].split(" ")) + len(row["rejected"].split(" ")) + len(row["system_message"].split(" "))
            
            if total_len_c > longest:
                longest = total_len_c
                longest_prompt = row["prompt"]
                longest_response = row["chosen"]
            if total_len_r > longest:
                longest = total_len_r
                longest_prompt = row["prompt"]
                longest_response = row["rejected"]

        print(f"Longest row: {longest}")
        print(f"Prompt: {longest_prompt}")
        print(f"Response: {longest_response}")

if __name__ == "__main__":

    tuluif = TuluIFPT()

    tuluif.load_from_huggingface("allenai/tulu-3-pref-personas-instruction-following")
    tuluif.format_dataset()
    tuluif.clear_old_datasets(prefix="pt-tuluif")
    tuluif.save_dataset_to_disk(save_name="pt-tuluif")