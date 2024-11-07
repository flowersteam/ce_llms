import argparse
import warnings
from collections import defaultdict
import json
import itertools
from pathlib import Path
import re

from visualization_utils import *

# different label -> different curves
def create_label(dir, part_dir, interaction_metric):  # different labels -> different curves
    # we merge different seeds into same curve

    # if "seed" in str(dir):
    #     seed_str = f" (seed={dir.name[:36][-9:]})"
    # else:
    #     seed_str = ""
    #
    # if "split" in dir:
    #     split_str = f" {dir.split('_split_')[1].split('_')[0]} split "
    # else:
    #     split_str = " train split "
    #
    # return str(dir).split("reddit_")[1].split("_roof_prob_")[0] + \
    #     split_str + seed_str + f" (AI ratio={interaction_metric}, {str(part_dir.name).split('_undeduplicated')[0]})"

    return str(dir).replace("eval_results/dev_results/", "") + f"({part_dir.name})"


def create_label_interaction_plot(dir, part_dir):  # different label -> different curves
    # we merge different ratios and seeds into same curve

    # if "split" in dir:
    #     split_str = f" {dir.split('_split_')[1].split('_')[0]} split "
    # else:
    #     split_str = " train split "
    #
    # return str(dir).split("reddit_")[1].split("_roof_prob_")[0] + split_str + f"({str(part_dir.name).split('_undeduplicated')[0]})"
    return str(dir).replace("eval_results/dev_results/", "").split("generated_")[0] + f"({part_dir.name})"


def label_to_color_id(label):  # label with different color_id -> different colors
    # we remove all after generated_ -> different ratios and seed have same color

    # return label.split("AI ratio")[0]
    # return label
    return label.split("generated_")[0]


# this can be used to fix some experiment_ids to colors
def parse_colors_dict(colors_dict):
    # parse color dict
    # for d, c in colors_dict.items():
    #     if "ft_size_8000" in d:
    #         colors_dict[d] = "green"
    #     elif "participants_2" in d:
    #         colors_dict[d] = "blue"
    #     else:
    #         colors_dict[d] = "red"
    return colors_dict


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

        if "political_lean_score" in res:
            res["political_lean_score_std"] = [np.std(r) for r in res['political_lean_score']]

        results[directory] = res

    return results


# example run:
# python visualize.py  eval_results/Testing_iterative_learning_* --metric var_diversities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a metric from multiple directories.")
    parser.add_argument("--directories", nargs="+", help="List of directories to search for results.json")
    parser.add_argument("--metric", type=str, default="ces_Qwen/Qwen2.5-72B")
    parser.add_argument("--interaction-metric", type=str, default="ai_ratio")
    parser.add_argument("--interaction-plots", "-ip", action="store_true", help="Show interaction plots.")
    parser.add_argument("--visualize-datasets", action="store_true")
    parser.add_argument("--assert-n-datapoints", type=int, default=None, help="Assert number of datapoints to shade by (e.g. seeds).")
    parser.add_argument("--plot-2D", nargs="+", help="Plot 2D graph of the metrics", default=None)

    parser.add_argument("--violin", action="store_true")
    parser.add_argument("--per-seed", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--log", action="store_true", help="Log scale y axis")

    args = parser.parse_args()

    if args.metric == "div":
        args.metric = "cos_diversity_stella"
    elif args.metric == "ppl":
        args.metric = "ppl_Qwen/Qwen2.5-72B"
    elif args.metric == "ce":
        args.metric = "ce_Qwen/Qwen2.5-72B"
    elif args.metric == "uniq":
        args.metric = "n_unique_posts"
    elif args.metric == "tox":
        args.metric = "toxicity"

    args.directories = sorted(args.directories, key=lambda d: int(d.split("generated_")[1].split("_human")[0]))

    all_participant_jsons = list(itertools.chain(*[Path(d).glob("**/part_*/results.json") for d in args.directories]))

    all_participant_directories = sorted([j.parent for j in all_participant_jsons])

    print(f"Results found: {len(all_participant_directories)}")
    # load data
    all_results = load_results(all_participant_directories)

    # extract the number of generations
    all_dataset_labels = [res['dataset_labels'] for res in all_results.values()]
    generation_labels = min(all_dataset_labels, key=len)
    n_generations = len(generation_labels)

    all_colors = ["red", "blue", "green", "black", "brown"] + list(mcolors.CSS4_COLORS.keys())[::-1]


    # this will be used to merge runs into seeds later
    dir_seed_part_dict = {}
    # dir -> seed -> part -> data
    for directory in args.directories:

        seed_part_dict = {}
        for seed_dir in list(Path(directory).glob("*")):
            part_dirs = list(seed_dir.glob("*"))
            # part -> data
            parts_dict = {part_dir: all_results[part_dir] for part_dir in part_dirs if part_dir in all_results} # part_dir could be empty
            # seed -> part
            seed_part_dict[seed_dir] = parts_dict

        # dir -> seed
        dir_seed_part_dict[directory] = seed_part_dict



    # label_data_dict (dir+part) -> (n_generations, n_datapoints)
    # n_datapoints - datapoints to shade by or draw violin plots by (e.g. n_seeds, or n_posts)
    # n_generations - x_axis
    label_metric_dict = defaultdict(lambda :defaultdict(list))
    label_linewith_dict = {}
    for dir, seed_part_dict in dir_seed_part_dict.items():
        for seed, parts_dict in seed_part_dict.items():
            for part, data in parts_dict.items():

                # (n_generations, n_posts) or (n_generations)
                try:
                    scores = data[args.metric]
                except:
                    raise ValueError(f"Metric {args.metric} not found. Available metrics: {data.keys()}.")

                interaction_metric = data[args.interaction_metric]  # e.g. AI ratio or ratio L-R ratio
                if args.interaction_plots:
                    # dir-wo_ratio -> (ratio, n_seeds)  -> different ratios in the same plot
                    label = create_label_interaction_plot(dir, part)
                    last_gen_score = np.mean(scores[-1])
                    label_metric_dict[label][interaction_metric].append(last_gen_score)

                elif args.per_seed:
                    # dir+seed -> (n_generations, n_posts)
                    label = create_label(seed, part, interaction_metric)

                    for gen_i, s in enumerate(scores):
                        assert len(label_metric_dict[label][gen_i]) == 0, "each seed should be processed only once"
                        label_metric_dict[label][gen_i] = s

                else:
                    # dir -> (n_generations, n_seeds)
                    label = create_label(dir, part, interaction_metric)

                    # some metrics are computer on a per-post basis (e.g. ce) so we average over posts
                    # for other metrics (e.g. diversity) this has no effect
                    scores = [np.mean(s) for s in scores]
                    for gen_i, s in enumerate(scores):
                        label_metric_dict[label][gen_i].append(s)

                if args.interaction_plots:
                    label_linewith_dict[label] = 1
                else:
                    if label in label_linewith_dict:
                        assert label_linewith_dict[label] == interaction_metric, "things with the same label must have the same ratio to define linewidth"
                    else:
                        label_linewith_dict[label] = interaction_metric

    labels = list(label_metric_dict.keys())
    ys = [label_metric_dict[l] for l in labels]
    linewidths = [label_linewith_dict[l] * 10 for l in labels]
    linewidths = [label_linewith_dict[l] * 5 for l in labels]
    # linewidths = [1.0 for l in labels]

    # label with the same color_id will have the same colors
    color_ids = [label_to_color_id(l) for l in labels]
    color_id_to_color_dict = dict(zip(set(color_ids), all_colors))
    colod_id_to_color_dict = parse_colors_dict(color_id_to_color_dict)
    colors = [color_id_to_color_dict[color_id] for color_id in color_ids]

    ylabel = metric_name_parser(args.metric)

    plot_and_save(
        # xs are ys.keys() and will be sorted
        ys=ys,  # shape (n_labels=n_curves, x_axis=n_generations/ratios, data_points=n_seed/n_posts)
        colors=colors,
        labels=labels,
        linewidths=linewidths,
        violin=args.violin,
        ylabel=ylabel,
        xlabel=args.interaction_metric if args.interaction_plots else "Generation",
        save_path=args.save_path,
        no_show=args.no_show,
        log=args.log,
        assert_n_datapoints=args.assert_n_datapoints,
        fontsize=10,
    )

    exit()

    # if metric != 'political_bias':
    for i in range(len(metric_values)):

        if args.metric == 'political_bias':
            print("Political bias")
            print(metric_values[i])

        plot_metric_distributions(metric_values[i], args.metric, dir_labels[i], args.no_show)

    # By default the 2D plot is toxicity x political bias
    if args.plot_2D:
        assert len(args.plot_2D) == 2, "Please provide exactly 2 metrics to plot in 2D."

        y1s, y2s = [], []
        for dir, res in all_results.items():

            y1 = res[args.plot_2D[0]][:n_generations]
            y2 = res[args.plot_2D[1]][:n_generations]
            save_path = args.save_path
            y1s.append(y1)
            y2s.append(y2)


        plot_2D(y1s, y2s, dir_labels, save_path, args.no_show, ylabel=args.plot_2D[1], xlabel=args.plot_2D[0])
            