import re
from openai import AzureOpenAI
import tqdm
import os
import json
import pathlib
import time

ANNOTATION_ERROR = "Annotation Error"


def _get_azure_client(api_version: str) -> AzureOpenAI:
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not azure_endpoint:
        raise ValueError("Set AZURE_OPENAI_ENDPOINT before calling Azure OpenAI helpers.")

    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=api_version,
    )


def make_gpt4o_calls_batched(prompts: list[str], result_label_list=None, model_name="gpt-4o-2", num_responses=1, temperature=1.0):
    client = _get_azure_client("2024-10-21")
    
    tasks = []
    for idx, prompt in enumerate(prompts):
        tasks.append(
            {
                "custom_id": str(idx),
                "method": "POST",
                "url": "/chat/completions",
                "body": {
                    "model": model_name,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                    "n": num_responses,
                },                
            }
        )

    print(f"Annotating {len(tasks)} tasks in a batch call")
    temp_dir = "./data/tmp"

    pathlib.Path(temp_dir).mkdir(parents=True, exist_ok=True)

    file_name = f"{temp_dir}/gpt_annotation_calls.jsonl"

    with open(file_name, "w+") as file:
        for obj in tasks:
            file.write(json.dumps(obj) + "\n")

    print(f"Batch call file written to {file_name}")

    batch_file = client.files.create(file=open(file_name, "rb"), purpose="batch")

    print(batch_file.model_dump_json(indent=2))
    uploaded = False
    while not uploaded:
        batch_file = client.files.retrieve(batch_file.id)
        print(f"Waiting for batch file to upload. Current status: {batch_file.status}")
        if batch_file.status == "processed":
            uploaded = True
        time.sleep(10)

    batch_job = client.batches.create(
        input_file_id=batch_file.id, endpoint="/v1/chat/completions", completion_window="24h"
    )

    id = batch_job.id
    completed = False
    result_file_id = None

    while not completed:
        current_job = client.batches.retrieve(id)
        print(f"Waiting for batch job to complete. Current status: {current_job.status}")

        if current_job.status == "failed":
            print("Batch job failed")
            print(current_job)
            break

        if current_job.status == "completed":
            print(current_job)
            result_file_id = current_job.output_file_id
            print(f"Batch job completed. Result file id: {result_file_id}")
            if not result_file_id:
                print("Batch requests failed")
                result_file_id = current_job.error_file_id
                print(client.files.content(result_file_id).content)
            break

        time.sleep(10)

    result = client.files.content(result_file_id).content
    result_file_name = f"{temp_dir}/subqa_annotate_calls_results.jsonl"

    with open(result_file_name, "wb") as file:
        file.write(result)

    results = []
    with open(result_file_name, "r") as file:
        for line in file:
            json_object = json.loads(line.strip())
            results.append(json_object)

    print(f"Annotated {len(results)} tasks")

    if num_responses == 1:
        annotations = {
            int(res["custom_id"]): res["response"]["body"]["choices"][0]["message"]["content"]
            for res in results
        }
    else:
        annotations = {
            int(res["custom_id"]): [choice["message"]["content"] for choice in res["response"]["body"]["choices"]]
            for res in results
        }

    annotations = list(annotations.items())
    annotations.sort(key=lambda x: int(x[0]))
    annotations = [x[1] for x in annotations]

    assert len(annotations) == len(prompts)
    return annotations

    
def make_gpt4o_call(prompts: list[str], result_label_list=None, model_name="gpt-4o", num_responses=1, temperature=1.0):
    print(f"Annotating {len(prompts)} examples with GPT-4")

    client = _get_azure_client("2024-02-15-preview")
    annotations = []
    for prompt in tqdm.tqdm(prompts):
        try:
            chat_completion = client.chat.completions.create(
                model=model_name, 
                messages=[{"role": "user", "content": prompt}],
                n=num_responses,
                temperature=temperature
            )
            if num_responses == 1:
                response = chat_completion.choices[0].message.content
            else:
                response = [choice.message.content for choice in chat_completion.choices]
            annotations.append(response)
        except Exception as e:
            print(f"Error: {e}")
            response = ANNOTATION_ERROR
            annotations.append(ANNOTATION_ERROR)

    assert len(annotations) == len(prompts)
    return annotations

def save_responses(save_path: str, file_name: str, results: list[dict]):

    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(f"{save_path}/{file_name}", "w", encoding='utf-8') as f:
        for response in results:
            f.write(json.dumps(response, ensure_ascii=False) + "\n")

    print(f"Responses {len(results)} saved to {save_path}/{file_name}")


def read_responses(save_path: str, file_name: str) -> list[dict]:
    print(f"Reading responses from {save_path}/{file_name}")
    with open(f"{save_path}/{file_name}", "r") as f:
        generations = [json.loads(line) for line in f]

    return generations

if __name__ == "__main__":
    sample_prompts = ["What is the capital of France?", "What is the capital of Germany?"]
    annotations = make_gpt4o_calls_batched(
        prompts=sample_prompts,
        model_name="gpt-4o-2",
        num_responses=2)
    
    print(annotations)
