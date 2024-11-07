import json
import argparse
import time
from pathlib import Path
import hashlib

import numpy as np

from dataset_utils import *

from datasets import concatenate_datasets

cache_dir = ".cache"


# Set the huggingface cache path


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--exp-path', type=str, default=None)
    parser.add_argument('--generation', "-g", type=int, default=0, help='generation of the model')
    parser.add_argument('--n-participants', "-p", type=int, default=1)
    parser.add_argument('--per-participant-human-dataset-size', "-hd", type=int, default=0)
    parser.add_argument('--roof-prob', type=float, default=None, help='Max prob for each class of data (e.g. subreddit)')
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--human-dataset', type=str, default="twitter")
    parser.add_argument('--human-dataset-lean', type=str, default=None, choices=["Liberal", "Conservative"])
    parser.add_argument('--seed', type=str, default="1")

    args = parser.parse_args()
    print(f"Sample datasets -> gen: {args.generation}")
    args.seed = int(hashlib.md5(args.seed.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)

    all_datasets = []

    if args.per_participant_human_dataset_size > 0:
            print(f"Loading a human dataset of size {args.per_participant_human_dataset_size} * {args.n_participants}, with seed {args.seed}")

            human_dataset = load_human_dataset(
                dataset_name=args.human_dataset,
                roof_prob=args.roof_prob,
                load_n=args.per_participant_human_dataset_size*args.n_participants,
                lean=args.human_dataset_lean,
                seed=args.seed,
                split=args.split
            )
            source_column = ["human"] * len(human_dataset)
            human_dataset = human_dataset.add_column("source", source_column)
            all_datasets.append(human_dataset)

    if args.generation > 0:

        prev_generation_dir = Path(args.exp_path) / f"gen_{args.generation - 1}"
        assert args.n_participants == len([p for p in prev_generation_dir.glob("*") if p.is_dir()])

        # load all datasets
        for participant_id in range(args.n_participants):
            prev_generation_part_dir = prev_generation_dir / f"part_{participant_id}"
            prev_generation_part_output_dataset_path = str(prev_generation_part_dir / "output_dataset")

            participant_output_dataset = Dataset.load_from_disk(prev_generation_part_output_dataset_path)

            source_column = [f"AI_part_{participant_id}"] * len(participant_output_dataset)
            participant_output_dataset = participant_output_dataset.add_column("source", source_column)
            all_datasets.append(participant_output_dataset)

        assert args.n_participants + int(args.per_participant_human_dataset_size > 0) == len(all_datasets)

    # resample datasets
    full_dataset = concatenate_datasets(all_datasets).shuffle(seed=args.seed)
    full_dataset_size = len(full_dataset)
    print(f"Full dataset size: {full_dataset_size}")

    new_datasets = [full_dataset.shard(num_shards=args.n_participants, index=i) for i in range(args.n_participants)]

    # separate the dataset
    curr_generation_dir = Path(args.exp_path) / f"gen_{args.generation}"
    for participant_id, input_dataset in enumerate(new_datasets):
        curr_generation_part_dir = curr_generation_dir / f"part_{participant_id}"
        os.makedirs(curr_generation_part_dir, exist_ok=True)

        input_dataset_path = curr_generation_part_dir / "input_dataset"
        input_dataset.save_to_disk(input_dataset_path)
        print(f"Input dataset (size={len(input_dataset)}) saved to : {input_dataset_path}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total for resampling datasets: {total_time}")

    logs = {
        "args": vars(args),
        "full_dataset_size": full_dataset_size,
        "total_time": total_time,
    }

    log_json_path = curr_generation_dir / "log_sample_datasets.json"
    with open(log_json_path, "w") as f:
        json.dump(logs, f)

    print(f"Log saved to {log_json_path}.")






