import argparse
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge dataset parts")
    parser.add_argument("dataset_save_path", type=str, help="Path to the saved dataset file, also the prefix of parts")
    args = parser.parse_args()

    if args.dataset_save_path.endswith("/"):
        raise ValueError("dataset_save_path should not end with /")

    ds = []
    for part_id in range(20):
        part_load_path = f"{args.dataset_save_path}_part_{part_id}"
        ds.append(datasets.load_from_disk(part_load_path))

    dataset = datasets.concatenate_datasets(ds)

    print(f"Dataset loaded: {dataset}")
    d_ = dataset.load_from_disk("./data/wikipedia/wikipedia_dataset")
    assert d_['text'] == dataset['text']

    dataset.save_to_disk(args.dataset_save_path)
    print(f"Dataset saved to: {args.dataset_save_path}")
