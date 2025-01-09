import argparse
from termcolor import cprint
from pathlib import Path
import datasets
from termcolor import cprint

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
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--max-steps', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr-scheduler', type=str, default="linear")
    parser.add_argument('--warmup-ratio', type=float, default=0.0)
    parser.add_argument('--warmup-steps', type=int, default=0)
    parser.add_argument('--per-device-batch-size', type=int, default=16)
    parser.add_argument('--alpha', type=int, default=16)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--quantize', action="store_true", help='Quantize model before Lora (Q-Lora)')
    parser.add_argument('--rslora', action="store_true", help='Quantize model before Lora (Q-Lora)')

    # Generation
    parser.add_argument('--generate', action="store_true", help='generate after training')
    parser.add_argument('--temp', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--max-new-tokens', type=int, default=300, help='Max new tokens')
    parser.add_argument('--top-p', type=float, default=1.0, help='1.0 no effect')
    parser.add_argument('--top-k', type=int, default=0, help='0 deactivates it')
    parser.add_argument('--min-p', type=float, default=0, help='')
    parser.add_argument('--repetition-penalty', type=float, default=1.0, help='1.0 no penalty')
    parser.add_argument('--gen-n', type=int, default=4000)
    parser.add_argument('--dataset-name', type=str, help='The dataset to use for instructions for generation.')
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--dataset-type', type=str, default="standard", help='Use only with reddit (ld, hq, standard)')

    args = parser.parse_args()
    print("Args:", args)

    save_dir = Path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    data_logs = {}

    # parse string_seed to int
    args.seed = int(hashlib.md5(args.seed.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)

    input_dataset_path = str(save_dir / "input_dataset")
    print(f"Loading the input dataset from : {input_dataset_path}")
    train_dataset = datasets.load_from_disk(input_dataset_path)

    data_logs["dataset_size"] = len(train_dataset)
    print(f"Training dataset size: ", data_logs["dataset_size"])

    # Train
    #######
    train_logs = {}

    print("Loading the model")
    model, tokenizer = load_model(
        args.model_name, args.seed,
        r=args.rank, alpha=args.alpha, load_in_4bit=args.quantize, use_rslora=args.rslora,
        max_model_len=1024
    )

    # Linear
    train_args = dict(
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=1,
        optim="adamw_8bit",
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

    cprint(f"Training (train set size = {len(train_dataset)})", "green")
    # Train model
    model_save_dir = save_dir / "model"
    start_time_train = time.time()
    train_logs = train_model_unsloth(
        tokenizer=tokenizer,
        model=model,
        dataset=train_dataset,
        save_dir=model_save_dir,
        train_args=train_args
    )
    train_secs = time.time() - start_time_train
    print(f"Train time comp - time: {train_secs} unsloth: {train_logs['train_result'].metrics['train_runtime']}")
    hours, minutes, seconds = secs_2_hms(train_secs)
    cprint("Training time: %d:%02d:%02d" % (hours, minutes, seconds) + f" ({train_secs} secs)", "blue")

    # Generate
    ###########
    all_instructions = get_instructions(dataset_name=args.dataset_name, n=args.gen_n)

    start_time_generate = time.time()

    cprint(f"Generating ({args.gen_n} posts)", "green")
    output_dataset, gen_logs = generate_data(
        tokenizer=tokenizer,
        model=model,
        batch_size=500,
        all_instructions=all_instructions,
        generate_n_posts=args.gen_n,
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

    output_dataset_path = save_dir / "full_output_dataset"
    output_dataset.save_to_disk(output_dataset_path)
    print(f"Output dataset saved to {output_dataset_path}")

    print("Samples of generated texts: ")
    for i in range(3):
        print(f"Sample {i}->{output_dataset['text'][i]}")

    # Save
    ######
    # save generations to csv
    save_texts_to_csv(output_dataset['text'], save_dir / "generations.csv")
    gen_secs = time.time() - start_time_generate
    print(f"Gen time comp - time: {gen_secs} unsloth: {gen_logs['generation_time']}")
    hours, minutes, seconds = secs_2_hms(gen_secs)
    cprint("Generation time: %d:%02d:%02d" % (hours, minutes, seconds) + f" ({gen_secs} secs)", "blue")

    end_time = time.time()
    total_time = end_time - start_time
    hours, minutes, seconds = secs_2_hms(total_time)
    cprint("Total time (ft_and_gen): %d:%02d:%02d" % (hours, minutes, seconds) + f" ({total_time} secs)", "blue")

    logs = {
        "args": vars(args),
        "data": data_logs,
        "training": train_logs,
        "generation": gen_logs,
        "gen_time": gen_secs,
        "train_time": train_secs,
        "total_time": total_time,
    }

    log_json_path = save_dir / "log.json"
    with open(log_json_path, "w") as f:
        json.dump(logs, f)

    print(f"Log saved to {log_json_path}.")

    del model
    torch.cuda.empty_cache()

