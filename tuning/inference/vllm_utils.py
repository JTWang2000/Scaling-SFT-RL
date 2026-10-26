from multiprocessing import util

from tuning.utils.utils import chat_template_func
from vllm import LLM, SamplingParams
from tuning.config import MODELS_DIR
from tuning.inference.config_inference import VLLMSamplingParamsConfig

# ====================== helper function =====================
import torch

def get_gpu_memory_gb(device=0):
    props = torch.cuda.get_device_properties(device)
    total_mem_bytes = props.total_memory
    return total_mem_bytes / (1024 ** 3)  # convert to GB


def choose_gpu_utilization(mem_gb):
    if mem_gb <= 50:          # e.g., 40GB class (L40, A100-40G)
        return 0.6
    elif mem_gb >= 120:       # e.g., 140GB+ (H200)
        return 0.5
    else:                     # middle range (A100-80G etc.)
        return 0.5


def load_vlm_model(model_name: str, no_path=False, sampling_overrides: dict | None = None) -> LLM:
    model_path = f"{MODELS_DIR}/{model_name}"
    if no_path:
        model_path = model_name
    print(f"Loading model from {model_path}")
    mem_gb = get_gpu_memory_gb()
    util = choose_gpu_utilization(mem_gb)
    print(f"Detected GPU memory: {mem_gb:.1f} GB")
    print(f"Using gpu_memory_utilization={util}")
    llm = LLM(model=model_path, gpu_memory_utilization=util)
    sampling_config = VLLMSamplingParamsConfig().model_dump()
    if sampling_overrides:
        sampling_config.update(sampling_overrides)
    print(f"Sampling params: {sampling_config}")
    sampling_params = SamplingParams(
        **sampling_config
    )

    return llm, sampling_params

def make_vllm_call(llm: LLM, sampling_params: SamplingParams, prompts: list[str]) -> list[str]:
        
    tokenizer = llm.get_tokenizer()
    tokenizer = chat_template_func(tokenizer)
    chat_template = tokenizer.chat_template

    outputs = llm.chat(prompts, sampling_params, chat_template=chat_template)
    if sampling_params.n == 1:
        responses = [output.outputs[0].text for output in outputs]
    else:
        responses = [[response.text for response in output.outputs] for output in outputs]

    print(f"Generated {len(responses)} responses using vllm")

    return responses

def tokenize_test_dataset(llm, messages):

    tokenizer = llm.get_tokenizer()
    tokenizer = chat_template_func(tokenizer)
    tokenized_prompts = [
        tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]

    return tokenized_prompts

def generate_responses_vllm(llm: LLM, sampling_params: SamplingParams, prompts: list[str], dataset):

    responses = make_vllm_call(llm, sampling_params, dataset)

    results = [
        {
            "prompt": prompt,
            "response": response,
        }
        for prompt, response in zip(prompts, responses)
    ]
    return results
