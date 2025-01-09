import argparse
import warnings
from collections import defaultdict
import json
import itertools
from pathlib import Path
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from visualization_utils import *
import numpy as np

from scipy.stats import spearmanr, pearsonr


# different label -> different curves

def metric_name_parser(metric_name):
    metric_name_parser_dict = {"dataset_lens": "unique_posts_generated"}
    if metric_name in metric_name_parser_dict:
        return metric_name_parser_dict[metric_name]

    metric_name = metric_name.replace("ppls", "perplexity_")
    metric_name = metric_name.replace("mistralai/", "")
    return metric_name


def load_results(directories):
    results = {}
    for directory in directories:
        results_file = os.path.join(directory, "results.json")

        if not os.path.isfile(results_file):
            print(f"Error: {results_file} not found.")
            continue

        with open(results_file, "r") as f:
            res = json.load(f)
            # parse str generations to ints
            for metr_, vals_ in res.items():
                if isinstance(vals_, dict):
                    res[metr_] = {int(gen): v for gen, v in vals_.items()}

        if "political_lean_score" in res:
            res["political_lean_score_std"] = [np.std(r) for r in res['political_lean_score']]

        results[directory] = res

    return results


# example run:
# python visualize.py  eval_results/Testing_iterative_learning_* --metric var_diversities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a metric from multiple directories.")
    parser.add_argument("--directories", nargs="+", help="List of directories to search for results.json")
    parser.add_argument("--part", type=str, default="part_0", help="Wildcard defining which participants to show. (Default all)")
    parser.add_argument("--metrics", nargs="+", type=str)
    parser.add_argument("--generations", nargs="+", type=int, default=[19])
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")

    args = parser.parse_args()

    print("N_dirs:", len(args.directories))
    args.directories = sorted(args.directories, key=lambda d: int(d.split("generated_")[1].split("_human")[0]))

    all_participant_jsons = list(itertools.chain(*[Path(d).glob(f"**/{args.part}/results.json") for d in args.directories]))

    all_participant_directories = sorted([j.parent for j in all_participant_jsons])

    print(f"Results found: {len(all_participant_directories)}")
    # load data
    all_results = load_results(all_participant_directories)


    datapoints = defaultdict(list)
    for dir, res in all_results.items():
        for generation in args.generations:
            for metric in args.metrics:

                try:
                    scores = res[metric][generation]
                except:
                    warnings.warn(f"Metric {metric} generation {generation} not found in {dir}")
                    continue

                if type(scores) in [float, int]:
                    scores = [scores]

                if len(args.metrics) == 2:
                    assert len(args.generations) == 1
                    datapoints[metric].append(np.mean(list(map(float, scores))))

                elif len(args.generations) == 2:
                    assert len(args.metrics) == 1
                    datapoints[generation].append(np.mean(list(map(float, scores))))

                else:
                    raise ValueError("Generations of metrics must have 2 elements")

    all_colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys())

    # this will be used to merge runs into seeds later

    m1, m2 = datapoints.keys()
    data_m1, data_m2 = datapoints[m1], datapoints[m2]
    assert len(data_m1) == len(data_m2)
    print("Number of datapoints found: ", len(data_m1))

    spearman_corr, _ = spearmanr(data_m1, data_m2)
    print(f"Spearman: {spearman_corr}")
    pearson_corr, _ = pearsonr(data_m1, data_m2)
    print(f"Pearson: {pearson_corr}")

    # Compute linear regression (for the line)
    slope, intercept = np.polyfit(data_m1, data_m2, 1)  # Linear regression (degree=1)
    line = np.polyval([slope, intercept], data_m1)  # Evaluate the line

    # Create the plot
    plt.scatter(data_m1, data_m2, color='blue', label="Data points")
    plt.plot(data_m1, line, color='red', label=f"Regression line\ny = {slope:.2f}x + {intercept:.2f}")
    plt.xlabel(m1)
    plt.ylabel(m2)

    plt.title(f"Correlations {args.metrics}")
    plt.legend(title=f"Spearman r = {spearman_corr:.2f}\nPearson r = {pearson_corr:.2f}")
    plt.grid(True)

    if not args.no_show:
        plt.show()

    if args.save_path:
        plt.savefig(args.save_path+".png", dpi=300)
        print(f"Saved to: {args.save_path}")

