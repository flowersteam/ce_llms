try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except:
    pass


import argparse
from termcolor import cprint
from pathlib import Path

from dataset_utils import *
from model_utils import *

cache_dir = ".cache"

if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model-name', type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--seed', type=str, default="1")

    # Model loading
    parser.add_argument('--alpha', type=int, default=16)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--quantized', action="store_true", help='Quantize model before Lora (Q-Lora)')

    # Generation
    parser.add_argument('--vllm', action="store_true", help='use vllm')
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

    args = parser.parse_args()
    print("Args:", args)

    save_dir = Path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    data_logs = {}

    # parse string_seed to int
    args.seed = int(hashlib.md5(args.seed.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
    # args.dataset_seed = int(hashlib.md5(args.dataset_seed.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
    all_instructions = get_instructions(dataset_name=args.dataset_name, cache_dir=cache_dir)

    if args.roof_prob:
        unique_instructions, capped_probs = get_capped_probs(all_instructions, roof_prob=args.roof_prob)
        instructions = random.choices(unique_instructions, weights=capped_probs, k=args.gen_n)

    else:
        instructions = random.sample(all_instructions, args.gen_n)

    max_model_len = 1024

    print("Loading the model")
    if args.vllm:
        s_load = time.time()

        model = LLM(
            model=args.model_name,
            max_model_len=max_model_len,
            seed=args.seed
        )

        loading_time = time.time() - s_load
        cprint(f"Model loading time: {loading_time}.", "blue")

        print("Generating")
        output_dataset, gen_logs = generate_data(
            vllm=True,
            model=model,
            instructions=instructions,
            generation_arguments=dict(
                max_tokens=args.max_new_tokens,  # 300
                temperature=args.temp,
                min_p=args.min_p,
                top_k=args.top_k,  # 20
                top_p=args.top_p,  # 0.9
                repetition_penalty=args.repetition_penalty,  # 1.15
            )
        )

    else:
        # unsloth
        print("Loading the model")
        model, tokenizer = load_model(
            args.model_name, args.seed,
            r=args.rank, alpha=args.alpha, load_in_4bit=args.quantized,
            max_model_len=1024
        )

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
    print("Generation Time: %d:%02d:%02d" % (hours, minutes, seconds))

    end_time = time.time()
    total_time = end_time - start_time
    cprint(f"Total time (gen): {total_time}", "blue")

    logs = {
        "args": vars(args),
        "data": data_logs,
        "generation": gen_logs,
        "total_time": total_time,
    }

    log_json_path = save_dir / "log_gen.json"
    with open(log_json_path, "w") as f:
        json.dump(logs, f)

    print(f"Log saved to {log_json_path}.")

    avg_len = np.mean([len(t) for t in output_dataset['text']])
    print(f"Avg gen text len: {avg_len}.")

    del model
    torch.cuda.empty_cache()

