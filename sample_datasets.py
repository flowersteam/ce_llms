import json
import argparse
import time
from pathlib import Path
import hashlib

import numpy as np

from dataset_utils import *

from datasets import concatenate_datasets


# Set the huggingface cache path

hostname = os.uname()[1]
if hostname == "PTB-09003439":
    hf_cache_dir = "/home/flowers-user/.cache/huggingface"
else:
    hf_cache_dir = os.environ["HF_HOME"]
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--exp-path', type=str, default=None)
    parser.add_argument('--generation', "-g", type=int, default=0, help='generation of the model')
    parser.add_argument('--n-participants', "-p", type=int, default=1)
    parser.add_argument('--human-dataset-size', "-hd", type=int, default=0)
    parser.add_argument('--human-dataset', type=str, default="twitter")
    parser.add_argument('--human-dataset-lean', type=str, default=None, choices=["Liberal", "Conservative"])
    parser.add_argument('--human-dataset-seed', type=str, default="1")
    parser.add_argument('--seed', type=str, default="1")
    parser.add_argument('--deduplicate', action="store_true", help='Deduplicate generated posts')

    args = parser.parse_args()
    print(f"Gen: {args.generation}")
    args.seed = int(hashlib.md5(args.seed.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
    args.human_dataset_seed = int(hashlib.md5(args.human_dataset_seed.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)

    if args.generation == 0:
        raise ValueError("Should not be called for generation 0.")

    all_datasets = []

    if args.human_dataset_size > 0:
        print(f"Loading a human dataset, with seed {args.human_dataset_seed}")

        if args.human_dataset == "twitter":
            human_dataset, _, _ = load_twitter_dataset(
                cache_dir=hf_cache_dir, load_n=args.human_dataset_size, lean=args.human_dataset_lean, seed=args.human_dataset_seed)

            human_dataset = human_dataset.map(remove_links, batched=True)

        elif args.human_dataset == "reddit":
            human_dataset, _, _ = load_reddit_dataset(
                cache_dir=hf_cache_dir, load_n=args.human_dataset_size, lean=args.human_dataset_lean, seed=args.human_dataset_seed)

        else:
            raise NotImplementedError(f"Undefined dataset {args.dataset}.")

        all_datasets.append(human_dataset)

    prev_generation_dir = Path(args.exp_path) / f"gen_{args.generation - 1}"
    assert args.n_participants == len(list(prev_generation_dir.glob("*")))

    # load all datasets
    for participant_id in range(args.n_participants):
        prev_generation_part_dir = prev_generation_dir / f"part_{participant_id}"
        prev_generation_part_generations_path = str(prev_generation_part_dir / "generations.csv")

        participant_generations = load_dataset_from_csv(prev_generation_part_generations_path)
        all_datasets.append(participant_generations)

    assert args.n_participants + int(args.human_dataset_size > 0) == len(all_datasets)

    # resample datasets
    full_dataset = concatenate_datasets(all_datasets).shuffle(seed=args.seed)
    print(f"Full dataset size: {full_dataset}")
    all_generations = full_dataset['text']

    new_datasets = [full_dataset.shard(num_shards=args.n_participants, index=i) for i in range(args.n_participants)]

    # separate the dataset
    curr_generation_dir = Path(args.exp_path) / f"gen_{args.generation}"
    for participant_id, input_dataset in enumerate(new_datasets):
        curr_generation_part_dir = curr_generation_dir / f"part_{participant_id}"
        os.makedirs(curr_generation_part_dir, exist_ok=True)
        # save generations to csv
        input_dataset_path = curr_generation_part_dir / "input_dataset.csv"
        save_texts_to_csv(input_dataset['text'], input_dataset_path)
        print(f"Input dataset (size={len(input_dataset)} saved to : {input_dataset_path}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total for resampling datasets: {total_time}")






