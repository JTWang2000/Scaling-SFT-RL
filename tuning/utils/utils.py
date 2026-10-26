import re
from unsloth.chat_templates import get_chat_template
from datasets import DatasetDict, Dataset


def chat_template_func(tokenizer):
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "chatml", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
        map_eos_token = False, # Maps <|im_end|> to </s> instead
    )

    return tokenizer

def apply_chat_template(tokenizer, dataset):

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts }

    tokenizer = chat_template_func(tokenizer)
    dataset = dataset.map(formatting_prompts_func, batched = True,)

    return dataset

def apply_chat_template_pt(tokenizer, dataset):

    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)
    
    tokenizer = chat_template_func(tokenizer)

    def formatting_prompts_func(example):
        
        prompt = example["prompt"]

        if type(prompt) == str:
            message = [
                {"role": "system", "content": example["system_message"]},
                {"role": "user", "content": example["prompt"]},
            ]
        elif type(prompt) == list:
            message = prompt

        example["prompt"] = tokenizer.apply_chat_template(message, tokenize = False, add_generation_prompt = True)
        example["chosen"] = tokenizer.apply_chat_template([{"role": "assistant", "content": example["chosen"]}], tokenize = False, add_generation_prompt = False)
        example["rejected"] = tokenizer.apply_chat_template([{"role": "assistant", "content": example["rejected"]}], tokenize = False, add_generation_prompt = False)

        example["chosen"] = _strip_prefix(example["chosen"], "<|im_start|>assistant\n ")
        example["rejected"] = _strip_prefix(example["rejected"], "<|im_start|>assistant\n ")

        return example


    dataset = dataset.map(formatting_prompts_func, batched = False)
    return dataset


def get_kto_rows(dataset):
    def get_rows(dataset_split):
        rows = []
        for prompt, chosen, rejected in zip(dataset_split["prompt"], dataset_split["chosen"], dataset_split["rejected"]): 
            rows.extend([
                {
                    "prompt": prompt[100:],
                    "completion": chosen,
                    "label": True,
                },
                {
                    "prompt": prompt[100:],
                    "completion": rejected,
                    "label": False,
                }
            ])

        return rows
    
    dataset_kto = DatasetDict()
    dataset_kto["train"] = Dataset.from_list(get_rows(dataset["train"]))
    dataset_kto["test"] = Dataset.from_list(get_rows(dataset["test"]))

    print(dataset_kto["train"][0])
    return dataset_kto
        