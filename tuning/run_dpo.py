import wandb
import argparse
from tuning.training.config_training import ModelLoadConfig, LoraConfig, PTRunConfig, DPOTrainingConfig, DatasetConfig, SFTRunConfig
from tuning.config import HF_MODEL_MAP
from tuning.run_inference import run_inference
from tuning.run_evaluation import run_evaluation
from tuning.training.dpo_training import train_model_dpo
from tuning.training.kto_training import train_model_kto

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SFT training")

    parser.add_argument("--model", type=str, help="llama3-8B", required=True)
    parser.add_argument("--dataset", type=str, help="Dataset", required=True)
    parser.add_argument("--train_size", type=int, help="Train size", required=True)
    parser.add_argument("--sft_train_size", type=int, help="SFT run train size", required=False)
    parser.add_argument("--data_seed", type=int, default=42, help="Data sampling seed")
    parser.add_argument("--task", type=str, help="Task name", required=True)
    parser.add_argument("--pft_method", type=str, help="PFT method", required=True)
    parser.add_argument("--learning_rate", type=float, default=None, help="Override training learning rate")
    parser.add_argument("--num_train_epochs", type=float, default=None, help="Override number of training epochs")
    parser.add_argument("--do_training", action="store_true", help="Do training")
    parser.add_argument("--do_inference", action="store_true", help="Do inference")
    parser.add_argument("--do_evaluation", action="store_true", help="Do evaluation")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    print(args.__dict__)
    lora_config = LoraConfig()
    model_load_config = ModelLoadConfig()
    training_args = DPOTrainingConfig()
    if args.learning_rate is not None:
        training_args.learning_rate = args.learning_rate
    if args.num_train_epochs is not None:
        training_args.num_train_epochs = args.num_train_epochs
    elif args.dataset.lower() == "tuluif":
        training_args.num_train_epochs = 2

    dataset_config = DatasetConfig(
        dataset = args.dataset,
        dataset_type = "pt",
        train_size = args.train_size,
        data_seed = args.data_seed,
    )

    print(dataset_config)

    if args.sft_train_size:
        sft_run_config = SFTRunConfig(
            dataset_config = DatasetConfig(
                dataset = args.dataset,
                dataset_type = "sft",
                train_size = args.sft_train_size,
                data_seed = args.data_seed,
            ),
            model_name_hf = HF_MODEL_MAP[args.model],
            model_name = args.model,
            task_name=args.task,
            output_dir=args.output_dir,
        )

    run_config = PTRunConfig(
        dataset_config = dataset_config,
        model_name_hf = HF_MODEL_MAP[args.model],
        model_name = args.model,
        sft_run_config = sft_run_config if args.sft_train_size else None,
        task_name=args.task,
        pft_method=args.pft_method,
        do_training=args.do_training,
        do_inference=args.do_inference,
        do_evaluation=args.do_evaluation,
        output_dir=args.output_dir,
    )

    print(run_config)


    if run_config.do_training:
        run = wandb.init(name=run_config.run_name, project="tuning", reinit=True)

        with run:
            if run_config.pft_method == "dpo":
                train_model_dpo(
                    run_config = run_config,
                    lora_config = lora_config,
                    model_load_config = model_load_config,
                    training_args = training_args,
                )
            elif run_config.pft_method == "kto":
                train_model_kto(
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
