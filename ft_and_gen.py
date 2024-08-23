import json
import os
import argparse
import datetime
from pathlib import Path
import hashlib

from dataset_utils import *
from model_utils import *
from eval_utils import *

from unsloth import is_bfloat16_supported

# Set the huggingface cache path

hostname = os.uname()[1]
if hostname == "PTB-09003439":
    hf_cache_dir = "/home/flowers-user/.cache/huggingface"
else:
    hf_cache_dir = "/gpfsscratch/rech/imi/utu57ed/.cache/huggingface"
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--generation', "-g", type=int, default=0, help='generation of the model')
    parser.add_argument('--participant-id', "-i", type=int, default=0, help='participant id (e.g. index)')
    parser.add_argument('--exp-path', type=str, default=None)
    parser.add_argument('--seed', type=str, default="1")
    parser.add_argument('--dataset-seed', type=int, default="1")

    # Model
    parser.add_argument('--model-name', type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument('--lora-r', type=int, default=256, help='lora rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora-dropout', type=int, default=0.1, help='lora dropout')

    # Generation
    parser.add_argument('--temp', type=float, default=0.7, help='Temperature for generation')

    # Dataset
    parser.add_argument('--dataset', type=str, default="twitter")
    parser.add_argument('--load-n', type=int, default=4000)
    parser.add_argument('--gen-n', type=int, default=4000)
    parser.add_argument('--lean', type=str, default=None, choices=["Liberal", "Conservative"])
    parser.add_argument('--deduplicate', action="store_true", help='Deduplicate generated posts')

    args = parser.parse_args()
    print(f"Gen: {args.generation} Part: {args.participant_id}")

    if args.exp_path is None:
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        args.exp_path = f"./results/test_model_{timestamp}"

    curr_generation_save_dir = Path(args.exp_path) / f"gen_{args.generation}" / f"part_{args.participant_id}"
    os.makedirs(curr_generation_save_dir, exist_ok=True)

    data_logs = {}

    # parse string_seed to int
    args.seed = int(hashlib.md5(args.seed.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)

    # Load train data
    if args.generation == 0:
        print(f"Loading a human dataset, with seed {args.dataset_seed}")

        if args.dataset == "twitter":
            train_dataset, _, _ = load_twitter_dataset(
                cache_dir=hf_cache_dir, load_n=args.load_n, lean=args.lean, seed=args.dataset_seed)

            d_ = train_dataset.map(remove_links, batched=True)

        elif args.dataset == "reddit":
            train_dataset, _, _ = load_reddit_dataset(
                cache_dir=hf_cache_dir, load_n=args.load_n, lean=args.lean, seed=args.dataset_seed)

        else:
            raise NotImplementedError(f"Undefined dataset {args.dataset}.")

    else:
        input_dataset_path = str(curr_generation_save_dir / "input_dataset.csv")
        print(f"Loading the input dataset from : {input_dataset_path}")
        train_dataset = load_dataset_from_csv(input_dataset_path)

    data_logs["dataset_size"] = len(train_dataset)
    print(f"Dataset size: ", data_logs["dataset_size"])

    # instructions
    instructions = get_instructions()

    # Train the model
    train_logs = {}

    assert args.model_name.startswith("unsloth")

    print("Loading the model")
    assert args.model_name == "unsloth/llama-3-8b-bnb-4bit"  # for now
    model_path = os.path.join(hf_cache_dir, args.model_name)

    model, tokenizer = load_model(model_path, args.seed)

    # Linear
    train_args = dict(
        num_train_epochs=1,
        max_steps=-1,
        # max_steps=5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        optim="adamw_8bit",
        save_steps=500,
        logging_steps=50,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        warmup_steps=5,
        weight_decay=0.01,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        seed=args.seed,
        save_strategy="steps",
        group_by_length=True,
    )

    print("Training")
    # Train model
    model_save_dir = curr_generation_save_dir / "model"
    train_logs = train_model_unsloth(
        tokenizer=tokenizer,
        model=model,
        dataset=train_dataset,
        save_dir=model_save_dir,
        instructions=instructions,
        train_args=train_args
    )
    hours, minutes, seconds = secs_2_hms(train_logs['train_result'].metrics['train_runtime'])
    print("Training Time: %d:%02d:%02d" % (hours, minutes, seconds))

    print("Generating")

    generated_texts, gen_logs = generate_data(
        tokenizer=tokenizer,
        model=model,
        save_dir=curr_generation_save_dir,
        instructions=instructions,
        n_posts_to_generate=args.gen_n,
        verbose=False,
        deduplicate=args.deduplicate,
        generation_arguments=dict(
            max_new_tokens=300,
            temperature=args.temp,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
        )
    )

    # save generations to csv
    save_texts_to_csv(generated_texts, curr_generation_save_dir / "generations.csv")

    hours, minutes, seconds = secs_2_hms(gen_logs["generation_time"])
    print("Generation Time: %d:%02d:%02d" % (hours, minutes, seconds))

    metrics = evaluate_generations(generated_texts, verbose=True)

    print("Mean TTR: ", metrics['TTR'])
    print("Min TTR: ", np.min(metrics['per_generation_metrics']["TTR"]))
    print("Joint TTR: ", metrics['JointTTR'])

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time}")

    logs = {
        "args": vars(args),
        "data": data_logs,
        "training": train_logs,
        "generation": gen_logs,
        "evaluation": metrics,
        "total_time": total_time,
    }

    log_json_path = curr_generation_save_dir / "log.json"
    with open(log_json_path, "w") as f:
        json.dump(logs, f)

    print(f"Log saved to {log_json_path}.")

    del model
    torch.cuda.empty_cache()






