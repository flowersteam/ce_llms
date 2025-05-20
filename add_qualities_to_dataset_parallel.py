import argparse
import numpy as np
import datasets
from eval_utils import llama_quality_scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add qualities to part of dataset")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset file")
    parser.add_argument("dataset_save_path", type=str, help="Path to the saved dataset file prefix")
    parser.add_argument("part_id", type=int, help="Part ID to process (0 to 19)")
    args = parser.parse_args()

    if args.dataset_path.endswith("/"):
        raise ValueError("dataset_path should not end with /")

    if args.dataset_save_path.endswith("/"):
        raise ValueError("dataset_save_path should not end with /")
    if not (0 <= args.part_id < 20):
        raise ValueError("part_id must be between 0 and 19")

    print(f"Loading dataset from {args.dataset_path}...")
    dataset = datasets.load_from_disk(args.dataset_path)
    print(f"Dataset loaded: {dataset}")

    # Split into 20 parts

    chunk_size = int(np.ceil(len(dataset) / 20))
    start_indx = chunk_size*args.part_id
    end_indx = min(start_indx+chunk_size, len(dataset))

    print(f"Dataset part: {start_indx} to {end_indx} (total: {len(dataset)})")

    dataset_part = dataset.select(range(start_indx, end_indx))

    dataset_part = dataset_part.map(
        lambda examples: {
            "llama_quality_scale": llama_quality_scale(examples["text"])
        },
        batched=True,
        desc=f"Adding qualities to part {args.part_id}",
        batch_size=10,
        num_proc=30
    )

    # Save the processed part
    save_path = f"{args.dataset_save_path}_part_{args.part_id}"
    print(f"Saving part {args.part_id} to {save_path}...")
    dataset_part.save_to_disk(save_path)
    print(f"Part {args.part_id} saved.")
