from tuning.config import GSM8K_OUTPUTS_DIR, MODELS_DIR, RESPONSES_FILENAME
import subprocess
from tuning.inference.config_inference import VLLMSamplingParamsConfig
from tuning.data.config import SYSTEM_MESSAGE_GSM8K
from tuning.evaluation.config import CHATML_TEMPLATE
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

def _save_base_model_with_chat_template(model_name: str, source_name: str):
    target_path = f"{MODELS_DIR}/{model_name}"
    os.makedirs(target_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source_name)
    model = AutoModelForCausalLM.from_pretrained(source_name)
    tokenizer.chat_template = CHATML_TEMPLATE
    model.save_pretrained(target_path)
    tokenizer.save_pretrained(target_path)
    return target_path


def gsm8k_evaluate(model_name: str, model_source: str | None = None, no_path: bool = False):

    print(f"Running GSM8K evaluation for {model_name}")

    if no_path:
        source_name = model_source or model_name
        model_path = _save_base_model_with_chat_template(model_name, source_name)
    else:
        model_path = f"{MODELS_DIR}/{model_name}"

    output_path = f"{GSM8K_OUTPUTS_DIR}/{model_name}/{RESPONSES_FILENAME}"

    config_path = os.path.join(model_path, 'tokenizer_config.json')
    with open(config_path, 'r') as file:
        config_data = json.load(file)
    config_data['chat_template'] = CHATML_TEMPLATE
    
    # Write the updated JSON data back to the file
    with open(config_path, 'w') as file:
        json.dump(config_data, file, indent=4)

    print(f"Updated {config_path} with chat_template.")    


    # lm-eval --tasks gsm8k --model vllm \
    #     --model_args model_path \
    #     --output_path=output_path \
    #     --log_samples --num_fewshot 0 --batch_size auto:N \
    #     --gen_kwargs temperature=0.4,top_p=0.95,repetition_penalty=1.1 \
    
    sampling_params = VLLMSamplingParamsConfig().model_dump()
    print(model_path)
    model_args = ",".join([
    f"pretrained={model_path}",
    "gpu_memory_utilization=0.6",
    # "seed=42",
    # "enforce_eager=True",
    # "enable_prefix_caching=False",
    # "disable_custom_all_reduce=True"
    ])

    try:
        result = subprocess.run(
            [
                "lm-eval",
                "--tasks", "gsm8k",
                "--model", "vllm",
                "--model_args", model_args,
                f"--output_path={output_path}",
                "--log_samples",
                "--num_fewshot", "0",
                "--batch_size", "auto:N",
                "--gen_kwargs", f"temperature={sampling_params['temperature']},top_p={sampling_params['top_p']}",
                "--verbosity", "DEBUG",
                "--write_out",
                "--apply_chat_template", "true",
                "--system_instruction", SYSTEM_MESSAGE_GSM8K,
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        return
    

    print(model_name)
    print(output_path)
    
if __name__ == "__main__":
    import time

    print("Starting GSM8K evaluation")
    # gsm8k_evaluate("llama3-8B_pt-gsm8k-10000")
