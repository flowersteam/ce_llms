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
    hf_cache_dir = "/gpfsscratch/rech/imi/utu57ed/.cache/huggingface"
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--exp-path', type=str, default=None)
    parser.add_argument('--generation', "-g", type=int, default=0, help='generation of the model')
    parser.add_argument('--n-participants', "-p", type=int, default=1)
    parser.add_argument('--seed', type=str, default="1")

    parser.add_argument('--deduplicate', action="store_true", help='Deduplicate generated posts')

    args = parser.parse_args()
    print(f"Gen: {args.generation}")

    if args.generation == 0:
        raise ValueError("Should not be called for generation 0.")

    prev_generation_dir = Path(args.exp_path) / f"gen_{args.generation - 1}"
    assert args.n_participants == len(list(prev_generation_dir.glob("*")))

    # load all datasets
    all_datasets = []
    for participant_id in range(args.n_participants):
        prev_generation_part_dir = prev_generation_dir / f"part_{participant_id}"
        prev_generation_part_generations_path = str(prev_generation_part_dir / "generations.csv")

        participant_generations = load_dataset_from_csv(prev_generation_part_generations_path)
        all_datasets.append(participant_generations)

    assert args.n_participants == len(all_datasets)

    args.seed = int(hashlib.md5(args.seed.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)

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






