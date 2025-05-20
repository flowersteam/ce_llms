import argparse
import datasets
from eval_utils import llama_quality_scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add qualities to dataset")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset file")
    parser.add_argument("dataset_save_path", type=str, help="Path to the saved dataset file")
    args = parser.parse_args()

    if args.dataset_path.endswith("/"):
        raise ValueError("dataset_path should not end with /")

    if args.dataset_save_path.endswith("/"):
        raise ValueError("dataset_save_path should not end with /")

    dataset = datasets.load_from_disk(args.dataset_path)
    print(f"Dataset loaded: {dataset}")

    dataset = dataset.map(
        lambda examples: {
            "llama_quality_scale": llama_quality_scale(examples["text"])
        }, batched=True, desc="Adding qualities", batch_size=10, num_proc=30
    )
    print("Dataset:", dataset)

    # overwrite_to_disk(dataset, args.dataset_path)

    dataset.save_to_disk(args.dataset_save_path)
