import argparse
import json
import os

from visualization_utils import *
import pickle

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
    parser.add_argument("--directories", nargs="+", help="List of directories to search for results.json")
    parser.add_argument("--metric", nargs="+", help="List of the metrics to load")
    parser.add_argument("--visualize-datasets", action="store_true")
    parser.add_argument("--plot-2D", action="store_true")
   
    # , choices=[
    #     'all',
    #     'var_diversities', 'cos_diversities',
    #     'logreg_loss', 'logreg_accuracy',
    #     'mean_ttrs', 'mean_n_words', 'dataset_lens',
    #     'ppls', 'positivity', 'political_bias', 'toxicity'
    # ])
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    # load data
    all_results = load_results(args.directories)

    all_dataset_labels = [res['dataset_labels'] for res in all_results.values()]

    # take the shortest
    dataset_labels = min(all_dataset_labels, key=len)
    n_datasets = len(dataset_labels)

    # Iterate over metrics passed as arguments
    for metric in args.metric:
        dir_labels, metric_values = [], []
        for dir, res in all_results.items():

            dir_labels.append(dir)
            
            if metric not in res:
                raise ValueError(f"Metric {metric} not found in {dir}.")

            metric_values.append(res[metric][:n_datasets])
            save_path = args.save_path + f"_{metric}" if args.save_path else None

            plot_and_save(
                x=dataset_labels,   # shape (n_datasets)
                ys=metric_values,  # shape (n_dirs, n_datasets)
                labels=dir_labels,  # shape (n_dirs)
                ylabel=metric,
                save_path=save_path,
                no_show=args.no_show,
            )


    # By default the 2D plot is toxicity x political bias
    if args.plot_2D:
        for dir, res in all_results.items():

            y1 = res["toxicity"][:n_datasets]
            y2 = res["political_bias"][:n_datasets]
            save_path = args.save_path

            plot_2D(y1, y2, dir, save_path, args.no_show)
            