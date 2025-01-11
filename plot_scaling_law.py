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

from scipy.stats import spearmanr, pearsonr, wasserstein_distance, ttest_ind


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

        results[directory] = res

    return results

def cohend(d1, d2):
    n1, n2 = len(d1), len(d2)
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    u1, u2 = np.mean(d1), np.mean(d2)
    return (u1 - u2) / s


def gaussian_kl(samples_p, samples_q):
    # p - true
    # q - approx
    # Estimate Gaussian parameters
    mu_p, std_p = np.mean(samples_p), np.std(samples_p)
    mu_q, std_q = np.mean(samples_q), np.std(samples_q)

    # KL divergence formula
    kl = (np.log(std_q / std_p) +
          (std_p ** 2 + (mu_p - mu_q) ** 2) / (2 * std_q ** 2) - 0.5)
    return kl


# example run:
# python visualize.py  eval_results/Testing_iterative_learning_* --metric var_diversities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a metric from multiple directories.")
    parser.add_argument("--directories", nargs="+", help="List of directories to search for results.json", required=True)
    parser.add_argument("--part", type=str, default="all_parts", help="Wildcard defining which participants to show. (Default all)")
    parser.add_argument("--metric", type=str, default="llama_quality_cap_100")
    parser.add_argument("--generation", type=int, default=4)
    parser.add_argument("--smooth-n", type=int, default=1)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--normalize", "-n", action="store_true", help="Normalize with respect to the human input dataset from generation 0")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--kl", action="store_true")
    parser.add_argument("--assert-n-datapoints", type=int, default=None, help="Assert number of datapoints to shade by (e.g. seeds).")

    args = parser.parse_args()

    print("N_dirs:", len(args.directories))
    args.directories = sorted(args.directories, key=lambda d: int(d.split("generated_")[1].split("_human")[0]))

    all_participant_jsons = list(itertools.chain(*[Path(d).glob(f"**/{args.part}/results.json") for d in args.directories]))

    all_participant_directories = sorted([j.parent for j in all_participant_jsons])

    print(f"Results found: {len(all_participant_directories)}")
    # load data
    print("Loading data")
    all_results = load_results(all_participant_directories)
    print("Data loaded")

    # extract data
    datapoints = defaultdict(lambda: defaultdict(list))
    for dir, res in all_results.items():
        try:
            n_parts = res['n_participants']
        except:
            # backwards compatibility
            n_parts = int(str(dir).split("participants_")[1].split("/")[0])
            # n_parts = json.loads((dir.parent / "gen_0" / "log_sample_datasets.json").read_text(encoding="UTF-8"))

        # n_parts = {4: 4.5, 5: 4.5}.get(n_parts, n_parts)
        ai_ratio = res['ai_ratio']

        # extract number of participants
        try:
            score = res[args.metric][args.generation]
        except:
            warnings.warn(f"Metric {args.metric} generation {args.generation} not found in {dir}")
            continue

        if type(score) == list:
            try:
                score = np.mean(score)
            except:
                scores_no_nan = [s for s in score if s is not None]
                n_nans = len(score) - len(scores_no_nan)
                warnings.warn(f"Scores cannot be averaged ({n_nans} Nones found) for Metric {args.metric} generation {args.generation} in {dir}. ")
                continue

                score = np.mean(scores_no_nan)
                # warnings.warn(f"Scores cannot be averaged (skipping {n_nans} Nones) for Metric {args.metric} generation {args.generation} in {dir}. ")

        if args.smooth_n > 1:
            end_ind = args.generation + 1
            start_ind = end_ind - args.smooth_n
            score = float(np.mean([res[args.metric][g] for g in range(start_ind, end_ind)]))

        if args.normalize:
            # score of input dataset at generation 0
            human_input_dataset_score = np.mean(res["input_" + args.metric][0])
            score = score / human_input_dataset_score

        if args.assert_n_datapoints:
            # max 3 seeds
            if len(datapoints[ai_ratio][n_parts]) < args.assert_n_datapoints:
                datapoints[ai_ratio][n_parts].append(score)
        else:
            datapoints[ai_ratio][n_parts].append(score)

    # plot data
    all_colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys())
    datapoints = dict(sorted(datapoints.items()))
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # this will be used to merge runs into seeds later

    all_xs=[]
    for ratio, ratio_data in datapoints.items():

        color = color_cycle[0]
        color_cycle = color_cycle[1:]

        xs = list(ratio_data.keys())
        xs = sorted(xs)

        xs_sc, ys_sc = zip(*itertools.chain(*[[(x, y_) for y_ in ratio_data[x]] for x in xs]))
        all_xs.extend(xs_sc)

        ys = np.array([np.mean(ratio_data[x]) for x in xs])
        plt.plot(xs, ys, label=f"ratio: {ratio}", color=color)

        if args.assert_n_datapoints:
            assert len(xs_sc) == args.assert_n_datapoints * len(xs), f" {len(xs_sc)} is not {args.assert_n_datapoints} * {len(xs)}"

        plt.scatter(xs_sc, ys_sc, color=color, s=5, marker='x', linewidths=0.5)

        try:
            shade_ys = np.array([float(sem(ratio_data[x])) for x in xs])
            # shade_ys = np.array([float(np.std(ratio_data[x])) for x in xs])
            plt.fill_between(xs, ys - shade_ys, ys + shade_ys, alpha=0.2, color=color)
        except:
            pass
        plt.scatter(xs, ys, color=color, s=8)

        plt.ylabel(args.metric)

    uniq_xs = sorted(set(all_xs))

    plt.xlabel("N participants")

    plt.legend()

    if not args.no_show:
        plt.show()

    if args.save_path:
        plt.savefig(args.save_path+".png", dpi=300)
        print(f"Saved to: {args.save_path}")

