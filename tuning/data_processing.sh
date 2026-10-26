#!/bin/bash

tasks=("gsm8k" "tuluif")
methods=("sft" "pref" "rlvr")

for task in "${tasks[@]}"; do
  for method in "${methods[@]}"; do
    script="${task}_${method}.py"
    echo "Processing ${task} for ${method}..."
    python "tuning/data/${script}"
  done
done
