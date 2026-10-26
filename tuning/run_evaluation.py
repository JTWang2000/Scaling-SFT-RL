from tuning.evaluation.ifeval_evaluate import ifeval_evaluate
from tuning.evaluation.gsm8k_evaluate import gsm8k_evaluate
from tuning.training.config_training import SFTRunConfig, PTRunConfig
from typing import Union

def run_evaluation(run_config: Union[SFTRunConfig, PTRunConfig]):

    task_name = run_config.task_name
    model_name = run_config.run_name
    
    if task_name == "instruction":
        ifeval_evaluate(model_name)
    elif task_name == "math":
        gsm8k_evaluate(model_name)
    else:
        raise ValueError(f"Task {task_name} not supported")
