import wandb
from tuning.training.config_training import ModelLoadConfig, LoraConfig, SFTRunConfig, TrainingArgumentsConfig, DatasetConfig
from tuning.config import HF_MODEL_MAP
from tuning.training.sft_training import train_model_sft
from tuning.run_inference import run_inference
from tuning.run_evaluation import run_evaluation
from tuning.inference.ifeval_inference import run_inference_ifeval
from tuning.evaluation.ifeval_evaluate import ifeval_evaluate
from tuning.evaluation.gsm8k_evaluate import gsm8k_evaluate

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SFT training")

    parser.add_argument("--model", type=str, help="llama3-8B", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset", required=True)
    parser.add_argument("--train_size", type=int, help="Train size", required=True)
    parser.add_argument("--data_seed", type=int, default=42, help="Data sampling seed")
    parser.add_argument("--task", type=str, help="Task name", required=True)
    parser.add_argument("--learning_rate", type=float, default=None, help="Override training learning rate")
    parser.add_argument("--num_train_epochs", type=float, default=None, help="Override number of training epochs")
    parser.add_argument("--zeroshot", action="store_true", help="Evaluate the original HF base model without training")
    parser.add_argument("--do_training", action="store_true", help="Do training")
    parser.add_argument("--do_inference", action="store_true", help="Do inference")
    parser.add_argument("--do_evaluation", action="store_true", help="Do evaluation")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    print(args.__dict__)

    if args.zeroshot:
        if args.do_training:
            raise ValueError("Zero-shot mode does not support --do_training")

        model_source = HF_MODEL_MAP[args.model]
        save_name = f"{args.model}_zeroshot-{args.dataset}"

        if args.do_inference:
            if args.task == "instruction":
                run_inference_ifeval(save_name, model_source=model_source, no_path=True)
            elif args.task == "math":
                print("GSM8K zero-shot inference is run during evaluation")
            else:
                raise ValueError(f"Task {args.task} not supported")

        if args.do_evaluation:
            if args.task == "instruction":
                ifeval_evaluate(save_name)
            elif args.task == "math":
                gsm8k_evaluate(save_name, model_source=model_source, no_path=True)
            else:
                raise ValueError(f"Task {args.task} not supported")

        print(f"********" * 20)
        raise SystemExit(0)

    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    training_args = TrainingArgumentsConfig()
    if args.learning_rate is not None:
        training_args.learning_rate = args.learning_rate
    if args.num_train_epochs is not None:
        training_args.num_train_epochs = args.num_train_epochs
    elif args.dataset.lower() == "tuluif":
        training_args.num_train_epochs = 2

    train_size = args.train_size 
    dataset_config = DatasetConfig(
        dataset = args.dataset,
        dataset_type = "sft",
        train_size = args.train_size,
        data_seed = args.data_seed,
    ) 

    print(dataset_config)

    run_config = SFTRunConfig(
        dataset_config = dataset_config,
        model_name_hf = HF_MODEL_MAP[args.model],
        model_name = args.model,
        task_name=args.task,
        do_training=args.do_training,
        do_inference=args.do_inference,
        do_evaluation=args.do_evaluation,
        output_dir=args.output_dir,
    )

    print(run_config)

    if run_config.do_training:
        run = wandb.init(name=run_config.run_name, project="tuning", reinit=True)

        with run:
            train_model_sft(
                run_config = run_config,
                lora_config = lora_config,
                model_load_config = model_load_config,
                training_args = training_args,
            )

    if run_config.do_inference:
        run_inference(run_config)

    if run_config.do_evaluation:
        run_evaluation(run_config)


    print(f"********" * 20) 
