#!/bin/bash

set -e

train_sizes=( 100 1000 10000 )

tasks=($(python3 -c '
import yaml
with open("tuning/collections.yaml") as f:
    data = yaml.safe_load(f)
    print(" ".join(data["tasks"]))
'))

PS3="Select task:"
select task in "${tasks[@]}"
do
    echo "Selected task: $task"
    # Get datasets for selected task
    task_datasets=($(python3 -c "
import yaml
with open('tuning/collections.yaml') as f:
    data = yaml.safe_load(f)
    print(' '.join(data['dataset']['$task']))
"))
    break
done 

PS3="Select dataset:"
select dataset in "${task_datasets[@]}"
do
    echo "Selected dataset: $dataset"
    break
done

sft_ratios=($(python3 -c '
import yaml
with open("tuning/collections.yaml") as f:
    data = yaml.safe_load(f)
    print(" ".join(map(str, data["sft_ratio"])))
'))

PS3="Select sft ratio:"
select sft_ratio in "${sft_ratios[@]}"
do
    echo "Selected sft ratio: $sft_ratio"
    break
done

base_models=($(python3 -c '
import yaml
with open("tuning/collections.yaml") as f:
    data = yaml.safe_load(f)
    print(" ".join(map(str, data["base_models"])))
'))

PS3="Select base model:"
select base_model in "${base_models[@]}"
do
    echo "Selected base model: $base_model"
    break
done


pft_types=($(python3 -c '
import yaml
with open("tuning/collections.yaml") as f:
    data = yaml.safe_load(f)
    print(" ".join(map(str, data["pft_types"])))
'))

PS3="Select pft method:"
select pft_method in "${pft_types[@]}"
do
    echo "Selected pft method: $pft_method"
    break
done

read -p "Do you want to perform training? (y/N): " do_training
read -p "Do you want to perform inference? (y/N): " do_inference
read -p "Do you want to perform evaluation? (y/N): " do_evaluation

# Prepare the command arguments
training_flag=$([ "${do_training,,}" = "y" ] && echo "--do_training" || echo "")
inference_flag=$([ "${do_inference,,}" = "y" ] && echo "--do_inference" || echo "")
evaluation_flag=$([ "${do_evaluation,,}" = "y" ] && echo "--do_evaluation" || echo "")

# Only ask about SFT if training is enabled
do_sft_first="n"
if [ "${do_training,,}" = "y" ]; then
    read -p "Do you want to run SFT first before DPO? (y/N): " do_sft_first
fi    

for train_size in "${train_sizes[@]}"; do
    sft_train_size=$(awk "BEGIN {print $train_size * $sft_ratio}")
    dpo_train_size=$(awk "BEGIN {print $train_size * (1 - $sft_ratio)}")
    
    if [ "$sft_ratio" = "1.0" ]; then
        python tuning/run_sft.py \
            --train_size "$sft_train_size" \
            --model $base_model \
            --dataset $dataset \
            --task $task \
            $training_flag \
            $inference_flag \
            $evaluation_flag
    else

        if [ "${do_sft_first,,}" = "y" ]; then
            echo "Running SFT first (training only)..."
            python tuning/run_sft.py \
                --train_size "$sft_train_size" \
                --model $base_model \
                --dataset $dataset \
                --task $task \
                --do_training
        fi

        python tuning/run_dpo.py \
            --train_size "$dpo_train_size" \
            --model $base_model \
            --dataset $dataset \
            --sft_train_size "$sft_train_size" \
            --task $task \
            --pft_method $pft_method \
            $training_flag \
            $inference_flag \
            $evaluation_flag
    fi
done