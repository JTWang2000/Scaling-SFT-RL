from tuning.data.test_dataset import get_ifeval_test_dataset
from tuning.config import IFEVAL_OUTPUTS_DIR, RESPONSES_FILENAME
from tuning.inference.vllm_utils import generate_responses_vllm, load_vlm_model
from tuning.utils.gpt_utils import save_responses

def run_inference_ifeval(model_name: str, model_source: str | None = None, no_path=False):
    test_dataset = get_ifeval_test_dataset()

    load_name = model_source or model_name
    llm, sampling_params = load_vlm_model(load_name, no_path=no_path)
    responses = generate_responses_vllm(llm = llm,
                                        sampling_params = sampling_params,
                                        prompts = list(test_dataset["prompt"]),
                                        dataset = list(test_dataset["messages"])
                                        )
    
    save_path = f"{IFEVAL_OUTPUTS_DIR}/{model_name}/"
    save_responses(save_path, RESPONSES_FILENAME, responses)

if __name__ == "__main__":

    import time

    run_inference_ifeval("llama3-8B_pt-tuluif-10000")
