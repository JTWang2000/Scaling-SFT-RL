import argparse

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from tuning.config import HF_MODEL_MAP, MODELS_DIR

def download_save_model(model_name, tokenizer_name, save_name):

    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    model.save_pretrained(f'{MODELS_DIR}/{save_name}')
    tokenizer.save_pretrained(f'{MODELS_DIR}/{save_name}')

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download and save model")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Model name",
    )

    args = parser.parse_args()
    model_name = args.model_name
    hf_model_name = HF_MODEL_MAP[model_name]

    download_save_model(
        model_name=hf_model_name, tokenizer_name=hf_model_name, save_name=model_name
    )
