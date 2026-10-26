import json
from tuning.utils.gpt_utils import make_gpt4o_call, make_gpt4o_calls_batched, ANNOTATION_ERROR
import pathlib

def gpt_evaluate(prompts):
    annotations = make_gpt4o_call(prompts, result_label_list=None)
    # annotations = make_gpt4o_calls_batched(prompts=prompts, model_name="gpt-4o-2")
    rankings = []
    for ranking in annotations:
        if ranking != ANNOTATION_ERROR:
            ranking = ranking.replace("```python\n", "")
            ranking = ranking.replace("```json\n", "")
            ranking = ranking.replace("```", "")
            try:
                ranking = json.loads(ranking)
                rankings.append(ranking["model_2"])
            except:
                rankings.append(0)
        else:
            rankings.append(0)

    return rankings
