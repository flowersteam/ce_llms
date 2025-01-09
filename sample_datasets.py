import json
import argparse
import time
from pathlib import Path
import hashlib

import numpy as np

from dataset_utils import *
from termcolor import cprint
from datasets import concatenate_datasets

cache_dir = ".cache"

def secs_2_hms(s):
    minutes, seconds = divmod(s, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--exp-path', type=str, default=None)
    parser.add_argument('--generation', "-g", type=int, default=0, help='generation of the model')
    parser.add_argument('--n-participants', "-p", type=int, default=1)
    parser.add_argument('--per-participant-human-dataset-size', "-hd", type=int, default=0)
    parser.add_argument('--per-participant-ai-dataset-size', "-aid", type=int, default=0)
    parser.add_argument('--gen-train-dataset-size-ratio', type=float, default=1.0)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--human-dataset', type=str, default="twitter")
    parser.add_argument('--human-dataset-lean', type=str, default=None, choices=["Liberal", "Conservative"])
    parser.add_argument('--keep-ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=str, default="1")
    parser.add_argument('--load-presampled-human-dataset', action="store_true", help='Deduplicate generated posts')
    parser.add_argument('--deduplicate', action="store_true", help='Deduplicate generated posts')
    parser.add_argument('--accumulate', type=int, default=0, help='Accumulate data (1 - True; 0 - False)')
    parser.add_argument('--dataset-type', type=str, default="standard", help='Use only with reddit (ld, hq, standard)')

    args = parser.parse_args()
    args.accumulate = bool(args.accumulate)

    print(f"Sample datasets -> gen: {args.generation}")
    args.seed = int(hashlib.md5(args.seed.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)

    if args.accumulate and args.keep_ratio != 0.5:
        raise NotImplementedError("Accumulation not implemented for data sharing.")

    # load human dataset for all participants
    if args.per_participant_human_dataset_size > 0:
            print(f"Loading a human dataset of size {args.per_participant_human_dataset_size} * {args.n_participants}, with seed {args.seed}")

            curr_generation_presampled_human_input_dataset_dir = Path(args.exp_path) / f"gen_{args.generation}" / "presampled_human_input_dataset"
            if args.load_presampled_human_dataset:
                assert curr_generation_presampled_human_input_dataset_dir.exists()
                print(f"Loading presampled human dataset from: {curr_generation_presampled_human_input_dataset_dir}")
                human_dataset = load_from_disk(curr_generation_presampled_human_input_dataset_dir)

            else:
                raise ValueError("Use presampled human datasets.")
                human_dataset = load_human_dataset(
                    dataset_name=args.human_dataset,
                    load_n=args.per_participant_human_dataset_size * args.n_participants,
                    lean=args.human_dataset_lean,
                    seed=args.seed,
                    split=args.split,
                    dataset_type=args.dataset_type
                )

            if args.gen_train_dataset_size_ratio != 1.0 and args.generation > 0:
                full_human_dataset_size = args.per_participant_human_dataset_size * args.n_participants
                human_dataset_size = int(np.ceil(full_human_dataset_size * args.gen_train_dataset_size_ratio))
                print(f"Human dataset size: {human_dataset_size} (gen_train ratio {args.gen_train_dataset_size_ratio}, {args.n_participants} participants)")
                human_dataset = human_dataset.select(range(human_dataset_size))
            else:
                assert len(human_dataset) == args.per_participant_human_dataset_size * args.n_participants

            source_column = [f"human_gen_{args.generation}"] * len(human_dataset)
            human_dataset = human_dataset.add_column("source", source_column)
    else:
        human_dataset = None

    ai_datasets = []

    # load ai datasets (generated in gen - 1 ) for all participants
    if args.generation > 0 and args.per_participant_ai_dataset_size > 0:

        prev_generation_dir = Path(args.exp_path) / f"gen_{args.generation - 1}"
        assert args.n_participants == len([p for p in prev_generation_dir.glob("part_*") if p.is_dir()])

        # load all datasets
        for participant_id in range(args.n_participants):
            prev_generation_part_dir = prev_generation_dir / f"part_{participant_id}"
            prev_generation_part_output_dataset_path = str(prev_generation_part_dir / "full_output_dataset")

            participant_output_dataset = Dataset.load_from_disk(prev_generation_part_output_dataset_path)

            if args.gen_train_dataset_size_ratio != 1.0:
                ai_dataset_size = int(np.ceil(args.per_participant_ai_dataset_size * args.gen_train_dataset_size_ratio))
                print(f"AI dataset size: {ai_dataset_size} (gen_train ratio {args.gen_train_dataset_size_ratio}, per participant)")
                participant_output_dataset = participant_output_dataset.select(range(ai_dataset_size))

            else:
                assert len(participant_output_dataset) == args.per_participant_ai_dataset_size

            if args.deduplicate:
                unique_indices = np.unique(participant_output_dataset['text'], return_index=True)[1]
                participant_output_dataset = participant_output_dataset.select(unique_indices)

            source_column = [f"AI_gen_{args.generation-1}_part_{participant_id}"] * len(participant_output_dataset)
            participant_output_dataset = participant_output_dataset.add_column("source", source_column)
            ai_datasets.append(participant_output_dataset)

    if args.generation == 0:
        assert len(ai_datasets) == 0

    # Merge human and ai datasets
    if human_dataset is not None:
        ai_datasets.append(human_dataset)
    full_dataset = concatenate_datasets(ai_datasets).shuffle(seed=args.seed)

    if args.accumulate:
        # save the new dataset (current full_dataset) - data generated by models in gen-1 + new human data
        curr_generation_new_dataset = Path(args.exp_path) / f"gen_{args.generation}" / "new_dataset"
        full_dataset.save_to_disk(curr_generation_new_dataset)

        print(f"Accumulating datasets from previous {args.generation} generations and {args.n_participants} participants.")


        # load previous datasets, this collects input datasets (new datasets)
        # gen=0
        previous_new_datasets = []
        for g_i in range(1, args.generation):  # skip first generation to maintain constant ai/human ratio
            prev_inp_dataset_path = str(Path(args.exp_path) / f"gen_{g_i}" / "new_dataset")
            previous_new_datasets.append(load_from_disk(prev_inp_dataset_path))

        # add previous datasets to the full_dataset
        datasets_to_merge = [full_dataset, *previous_new_datasets]

        assert len(set(map(len, datasets_to_merge))) == 1
        # sample the new full_dataset, size should be n_participants * per_part_ft_dataset_size
        full_dataset = concatenate_datasets(datasets_to_merge).shuffle(seed=args.seed)

    # Split the merged dataset into new training datasets
    per_participant_ft_dataset_size = args.per_participant_ai_dataset_size + args.per_participant_human_dataset_size

    if args.gen_train_dataset_size_ratio != 1.0:
        print("Generation / Training ratio is not 1 -> sampling the per participant training datasets")
        # sample a dataset for each participant
        n = min(per_participant_ft_dataset_size, len(full_dataset))  # if not enough data, shuffle and take all
        new_datasets = [full_dataset.shuffle().select(range(n)) for i in range(args.n_participants)]

    else:
        # split into exclusive datasets for each participant
        full_dataset = full_dataset.select(range(args.n_participants * per_participant_ft_dataset_size))
        new_datasets = [full_dataset.shard(num_shards=args.n_participants, index=i) for i in range(args.n_participants)]

    full_dataset_size = sum([len(new_d) for new_d in new_datasets])
    if args.gen_train_dataset_size_ratio == 1.0:
        assert full_dataset_size == args.n_participants*(args.per_participant_ai_dataset_size + args.per_participant_human_dataset_size)

    print(f"Full (fresh) dataset size: {full_dataset_size}")

    # save the new training datasets
    curr_generation_dir = Path(args.exp_path) / f"gen_{args.generation}"
    for participant_id, input_dataset in enumerate(new_datasets):
        curr_generation_part_dir = curr_generation_dir / f"part_{participant_id}"
        os.makedirs(curr_generation_part_dir, exist_ok=True)

        input_dataset_path = curr_generation_part_dir / "input_dataset"
        input_dataset.save_to_disk(input_dataset_path)
        print(f"Input dataset (size={len(input_dataset)}) saved to : {input_dataset_path}")

    end_time = time.time()
    total_time = end_time - start_time
    hours, minutes, seconds = secs_2_hms(total_time)
    cprint("Total time (sample_datasets): %d:%02d:%02d" % (hours, minutes, seconds) + f" ({total_time} secs)", "blue")

    logs = {
        "args": vars(args),
        "full_dataset_size": full_dataset_size,
        "total_time": total_time,
    }

    log_json_path = curr_generation_dir / "log_sample_datasets.json"
    with open(log_json_path, "w") as f:
        json.dump(logs, f)

    print(f"Log saved to {log_json_path}.")





