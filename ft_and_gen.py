import argparse
from termcolor import cprint
from pathlib import Path
import datasets

from dataset_utils import *
from model_utils import *

from unsloth import is_bfloat16_supported

cache_dir = ".cache"

if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--seed', type=str, default="1")

    # Model training
    parser.add_argument('--model-name', type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument('--save-merged', action="store_true", help='Save the merged model in addition to the adapters.')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr-scheduler', type=str, default="linear")
    parser.add_argument('--warmup-ratio', type=float, default=0.0)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--per-device-batch-size', type=int, default=16)
    parser.add_argument('--alpha', type=int, default=16)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--use-rslora', type=str, default="False", help='Rank Stabilized Lora')
    parser.add_argument('--quantization', type=str, default="True", help='Quantize model before Lora (Q-Lora)')

    # Generation
    parser.add_argument('--generate', action="store_true", help='generate after training')
    parser.add_argument('--temp', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--max-new-tokens', type=int, default=300, help='Max new tokens')
    parser.add_argument('--top-p', type=float, default=1.0, help='1.0 no effect')
    parser.add_argument('--top-k', type=int, default=0, help='0 deactivates it')
    parser.add_argument('--min-p', type=float, default=0, help='')
    parser.add_argument('--repetition-penalty', type=float, default=1.0, help='1.0 no penalty')
    parser.add_argument('--roof-prob', type=float, default=None, help='Max prob for instruction.')
    parser.add_argument('--gen-n', type=int, default=4000)
    parser.add_argument('--deduplicate', action="store_true", help='Deduplicate generated posts')
    parser.add_argument('--dataset-name', type=str, help='The dataset to use for instructions for generation.')
    parser.add_argument('--split', type=str, default="train")

    args = parser.parse_args()
    print("Args:", args)

    save_dir = Path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    data_logs = {}

    # parse string_seed to int
    args.seed = int(hashlib.md5(args.seed.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)

    if args.quantization == "True":
        args.quantization = True
    elif args.quantization == "False":
        args.quantization = False
    else:
        raise ValueError(f"Unrecognized value for {args.quantization} for quantization.")

    if args.use_rslora == "True":
        args.use_rslora = True
    elif args.use_rslora == "False":
        args.use_rslora = False
    else:
        raise ValueError(f"Unrecognized value for {args.quantization} for quantization.")

    input_dataset_path = str(save_dir / "input_dataset")
    print(f"Loading the input dataset from : {input_dataset_path}")
    train_dataset = datasets.load_from_disk(input_dataset_path)

    data_logs["dataset_size"] = len(train_dataset)
    print(f"Dataset size: ", data_logs["dataset_size"])

    # Train the model
    train_logs = {}

    print("Loading the model")
    model, tokenizer = load_model(
        args.model_name, args.seed,
        r=args.rank, alpha=args.alpha, load_in_4bit=args.quantization,
        use_rslora=args.use_rslora,
        max_model_len=1024
    )

    # Linear
    train_args = dict(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=1,
        optim="adamw_8bit",
        save_steps=500,
        logging_steps=50,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler,
        seed=args.seed,
        save_strategy="steps",
        group_by_length=True,
    )

    print("Training")
    # Train model
    model_save_dir = save_dir / "model"
    train_logs = train_model_unsloth(
        tokenizer=tokenizer,
        model=model,
        dataset=train_dataset,
        save_dir=model_save_dir,
        save_merged=args.save_merged,
        train_args=train_args
    )
    hours, minutes, seconds = secs_2_hms(train_logs['train_result'].metrics['train_runtime'])
    cprint("Training Time: %d:%02d:%02d" % (hours, minutes, seconds), "blue")

    if args.generate:

        all_instructions = get_instructions(dataset_name=args.dataset_name, cache_dir=cache_dir, split=args.split)

        if args.roof_prob:
            unique_instructions, capped_probs = get_capped_probs(all_instructions, roof_prob=args.roof_prob)
            instructions = random.choices(unique_instructions, weights=capped_probs, k=args.gen_n)

        else:
            instructions = random.sample(all_instructions, args.gen_n)

        print("Generating")
        output_dataset, gen_logs = generate_data(
            tokenizer=tokenizer,
            model=model,
            instructions=instructions,
            generation_arguments=dict(
                max_new_tokens=args.max_new_tokens,  # 300
                temperature=args.temp,
                min_p=args.min_p,
                top_k=args.top_k,  # 20
                top_p=args.top_p,  # 0.9
                repetition_penalty=args.repetition_penalty,  # 1.15
                do_sample=True,
            ),
            seed=args.seed
        )

        if args.deduplicate:
            duplicated_dataset_path = save_dir / "output_dataset_with_duplicates"
            output_dataset.save_to_disk(duplicated_dataset_path)
            print(f"Dataset with duplicates saved to: {duplicated_dataset_path}")

            output_dataset = output_dataset.select(np.unique(output_dataset['text'], return_index=True)[1])
            print(f"N posts after deduplication: {len(output_dataset)}")

        output_dataset_path = save_dir / "output_dataset"
        output_dataset.save_to_disk(output_dataset_path)
        print(f"Output dataset saved to {output_dataset_path}")

        # save generations to csv
        save_texts_to_csv(output_dataset['text'], save_dir / "generations.csv")
        hours, minutes, seconds = secs_2_hms(gen_logs["generation_time"])
        cprint("Generation Time: %d:%02d:%02d" % (hours, minutes, seconds), "blue")

    else:
        gen_logs = None

    end_time = time.time()
    total_time = end_time - start_time
    cprint(f"Total time (ft_and_gen): {total_time}", "blue")

    logs = {
        "args": vars(args),
        "data": data_logs,
        "training": train_logs,
        "generation": gen_logs,
        "total_time": total_time,
    }

    log_json_path = save_dir / "log.json"
    with open(log_json_path, "w") as f:
        json.dump(logs, f)

    print(f"Log saved to {log_json_path}.")

    del model
    torch.cuda.empty_cache()

