import json
import argparse
import time
from pathlib import Path

from termcolor import cprint

from dataset_utils import *

def secs_2_hms(s):
    minutes, seconds = divmod(s, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds

cache_dir = ".cache"


if __name__ == "__main__":

    start_time = time.time()

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--exp-path', type=str, default=None)
    parser.add_argument('--generations', "-g", type=int, default=0, help='generation of the model')
    parser.add_argument('--n-participants', "-p", type=int, default=1)
    parser.add_argument('--per-participant-human-dataset-size-gen-0', type=int, default=0)
    parser.add_argument('--per-participant-human-dataset-size', type=int, default=0)
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--human-dataset', type=str, default="twitter")
    parser.add_argument('--human-dataset-lean', type=str, default=None, choices=["Liberal", "Conservative"])
    parser.add_argument('--seed', type=str, default="1")
    parser.add_argument('--dataset-type', type=str, default="standard", help='Use only with reddit (ld, hq, standard)')

    args = parser.parse_args()

    if args.per_participant_human_dataset_size > 0:
        generations = args.generations
    else:
        generations = 1  # only first generation needs human data

    print(f"Presample human datasets")
    args.seed = int(hashlib.md5(args.seed.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)

    total_n = args.n_participants*(
            (generations - 1) * args.per_participant_human_dataset_size + args.per_participant_human_dataset_size_gen_0
    )

    human_dataset = load_human_dataset(
        dataset_name=args.human_dataset,
        load_n=total_n,
        lean=args.human_dataset_lean,
        seed=args.seed,
        split=args.split,
        dataset_type=args.dataset_type
    )
    human_dataset = human_dataset.shuffle(seed=args.seed)

    human_input_indices = []
    start_index = 0
    for gen_i in range(generations):
        if gen_i == 0:
            end_index = start_index + (args.per_participant_human_dataset_size_gen_0 * args.n_participants)
        else:
            end_index = start_index + (args.per_participant_human_dataset_size * args.n_participants)

        gen_part_input_dataset = human_dataset.select(range(start_index, end_index))

        gen_dir = Path(args.exp_path) / f"gen_{gen_i}"
        os.makedirs(gen_dir, exist_ok=True)

        input_dataset_path = gen_dir / "presampled_human_input_dataset"
        gen_part_input_dataset.save_to_disk(input_dataset_path)
        print(f"Presampled human input dataset (Gen:{gen_i};Indices:{(start_index, end_index)}; Size:{len(gen_part_input_dataset)}) saved to : {input_dataset_path}")

        start_index = end_index

    end_time = time.time()
    total_time = end_time - start_time
    hours, minutes, seconds = secs_2_hms(total_time)
    cprint("Total time (presample_human_datasets): %d:%02d:%02d" % (hours, minutes, seconds) + f" ({total_time} secs)", "blue")

    logs = {
        "args": vars(args),
        "full_dataset_size": total_n,
        "total_time": total_time,
    }

    log_json_path = Path(args.exp_path) / "log_sample_human_datasets.json"
    with open(log_json_path, "w") as f:
        json.dump(logs, f)

    print(f"Log saved to {log_json_path}.")






