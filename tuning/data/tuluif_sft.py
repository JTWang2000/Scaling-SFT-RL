from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING
    
class TuluIFSFT(HFDataset):
    def __init__(self):
        super().__init__(dataset_name="tulu")
    
    def _get_messages(self, examples):
        messages = examples["messages"]
        responses = [m[-1]["content"] for m in messages]
        prompts = examples["prompt"]

        return {
            "messages": [[
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING,
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ] for prompt, response in zip(prompts, responses)]
        } 

    def format_dataset(self):

        formatted_dataset = self._dataset.map(self._get_messages, batched=True)

        print(f"Tuluif sft dataset - {formatted_dataset}")
        print(f'Example Tuluif sft row - {formatted_dataset[0]}')
        print("***")
        self._dataset = formatted_dataset.train_test_split(test_size=200, shuffle=False)


if __name__ == "__main__":

    tuluif = TuluIFSFT()

    tuluif.load_from_huggingface("allenai/tulu-3-sft-personas-instruction-following", split="train")
    tuluif.format_dataset()
    tuluif.clear_old_datasets(prefix="sft-tuluif")
    tuluif.save_dataset_to_disk(save_name="sft-tuluif")