from tuning.inference.ifeval_inference import run_inference_ifeval
from tuning.training.config_training import SFTRunConfig, PTRunConfig
from typing import Union

def run_inference(run_config: Union[SFTRunConfig, PTRunConfig]):

    task_name = run_config.task_name
    model_name = run_config.run_name

    if task_name == "instruction":
        run_inference_ifeval(model_name)
    elif task_name == "math":
        print("GSM8K inference is run during evaluation")
    else:
        raise ValueError(f"Task {task_name} not supported")
