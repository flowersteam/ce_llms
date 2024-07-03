import argparse
import json
import os

from visualization_utils import *

def load_results(directories):
    results = {}
    for directory in directories:
        results_file = os.path.join(directory, "results.json")

        if not os.path.isfile(results_file):
            print(f"Error: {results_file} not found.")
            return None

        with open(results_file, "r") as f:
            results[directory] = json.load(f)

    return results


# example run:
# python visualize.py  eval_results/Testing_iterative_learning_* --metric var_diversities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a metric from multiple directories.")
    parser.add_argument("directories", nargs="+", help="List of directories to search for results.json")
    parser.add_argument("--metric", type=str, help="Name of the metric to load", default="cos_diversities", choices=[
        'var_diversities', 'cos_diversities',
        'logreg_loss', 'logreg_accuracy',
        'mean_ttrs', 'mean_n_words', 'dataset_lens',
        'ppls'
    ])
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    # load data
    all_results = load_results(args.directories)

    all_dataset_labels = [res['dataset_labels'] for res in all_results.values()]

    # take the shortest
    dataset_labels = min(all_dataset_labels, key=len)
    n_datasets = len(dataset_labels)

    dir_labels, metric_values = [], []

    for dir, res in all_results.items():

        dir_labels.append(dir)

        if args.metric not in res:
            raise ValueError(f"Metric {args.metric} not found in {dir}.")

        metric_values.append(res[args.metric][:n_datasets])

    plot_and_save(
        x=dataset_labels,   # shape (n_datasets)
        ys=metric_values,  # shape (n_dirs, n_datasets)
        labels=dir_labels,  # shape (n_dirs)
        ylabel=args.metric,
        save_path=args.save_path,
        no_show=args.no_show,
    )
