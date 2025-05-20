import argparse
import os
import warnings
from collections import defaultdict
import json
import itertools
from pathlib import Path
import re
import matplotlib.colors as mcolors
from termcolor import cprint

from visualization_utils import *

# different label -> different curves
def create_label(dir, part_dir, interaction_metric=None, metric=None, legend_conf=None):  # different labels -> different curves
    if "merged" in str(dir):
        dataset = "merged"
        if "_partition_" in str(dir):
            dataset += f" ({str(dir).split('_partition_')[1].split('/')[0]})"

    elif "senator" in str(dir):
        dataset = "senators_tweets"
    elif "100m" in str(dir):
        dataset = "100M_twitter"
    elif "webis" in str(dir) and "rstrip" in str(dir):
        dataset = "webis_reddit(rstrip)"
    elif "webis" in str(dir):
        dataset = "webis_reddit"
    elif "reddit_submissions" in str(dir):
        dataset = "reddit_submissions"
    elif "wikipedia" in str(dir):
        dataset = "wikipedia paragraphs"
    else:
        dataset = "Unknown dataset"

    # we merge different seeds into same curve

    if "seed" in str(dir):
        seed_str = f" (seed={dir.name.split('_202')[0]}_{dir.name[:36][-9:]})"
    else:
        seed_str = ""

    n_part = str(dir).split("_participants_")[1].replace("/", "_").split("_")[0]
    # pop_str = f" (pop size: {n_part})"
    pop_str = ""

    if interaction_metric is not None:
        # interaction plot
        interaction_metric_str = f" ratio : {interaction_metric}"
    else:
        interaction_metric_str = ""

    if "_type_" in str(dir):
        type = str(dir).split("_type_")[1].split("_part")[0]
        # type = str(dir).split("_type_")[1].split("_")[0]
    else:
        type = "standard"

    if "hq_1" in str(dir):
        type = "hq"

    if "scale_v3" in str(dir):
        model_str = " bigger models"
    else:
        model_str = ""

    dataset_type_str = f" type: "+{"hq": "high quality", "mq": "mid quality", "lq": "low quality"}.get(type, type) if type != "standard" else ""

    return f"{dataset}{model_str}{pop_str}{dataset_type_str}{interaction_metric_str}{seed_str}"

def label_to_color_id(label):  # label with different color_id -> different colors
    # we remove all after generated_ -> different ratios and seed have same color
    # return label.split("pop size: ")[1].split(")")[0]
    # return label
    try:
        return label.split("seed")[-1] # different colors
        # return label.split("seed")[0]  # per seed colors
    except:
        return label.split("ratio")[0]

    # return label.split("generated_")[0]


# this can be used to fix some experiment_ids to colors
def parse_colors_dict(colors_dict, legend_conf):

    if legend_conf == "qd":
        for d, c in colors_dict.items():
            if "Q20" in d:
                colors_dict[d] = "tab:red"
            elif "Q40" in d:
                colors_dict[d] = "tab:orange"
            elif "Q51" in d:
                colors_dict[d] = "tab:purple"
            elif "Q60" in d:
                colors_dict[d] = "tab:blue"
            elif "Q80" in d:
                colors_dict[d] = "tab:green"
            elif "high quality" in d:
                colors_dict[d] = "tab:green"
            elif "mid quality" in d:
                colors_dict[d] = "tab:blue"
            elif "low quality" in d:
                colors_dict[d] = "tab:red"
            else:
                colors_dict[d] = "black"

    elif legend_conf == "datasets":
        for d, c in colors_dict.items():
            if "webis" in d:
                colors_dict[d] = "tab:orange"
            elif "reddit_submissions" in d.lower():
                colors_dict[d] = "tab:red"
            elif "100m_tw" in d.lower():
                colors_dict[d] = "tab:blue"
            elif "senator" in d.lower():
                colors_dict[d] = "tab:green"
            elif "wikipedia" in d:
                colors_dict[d] = "tab:gray"
            else:
                colors_dict[d] = "black"

    else:
        return colors_dict


    return colors_dict


def metric_name_parser(metric_name):
    metric_name_parser_dict = {
        "llama_quality_scale_cap_250": "Quality",
        "cos_diversity_stella_cap_250": "Semantic Diversity"
    }
    return metric_name_parser_dict.get(metric_name, metric_name)

def xlabel_parser(label):
    parsedict = {"ai_ratio": "Synthetic data ratio"}
    return parsedict.get(label, label)


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

        if "gibberish_score_cap_250" in res:
            res["gibberish_quality_cap_250"] = {g: list(3-np.array(r)) for g, r in res['gibberish_score_cap_250'].items()}

        if "input_gibberish_score_cap_250" in res:
            res["input_gibberish_quality_cap_250"] = {g: list(3-np.array(r)) for g, r in res['input_gibberish_score_cap_250'].items()}

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
    parser.add_argument("--interaction-generation", "-ig", type=int, default=19)
    parser.add_argument("--interaction-generation-smooth", "-igs", type=int, default=1)
    parser.add_argument("---estimate-full-human-ratio-from", "-fhr", type=float, default=None, help="Extract gen 0 from this ratio as an estimate of full human data ratio.")
    parser.add_argument("--visualize-datasets", action="store_true")
    parser.add_argument("--assert-n-datapoints", type=int, default=None, help="Assert number of datapoints to shade by (e.g. seeds).")
    parser.add_argument("--plot-2D", nargs="+", help="Plot 2D graph of the metrics", default=None)
    parser.add_argument("--legend-conf", type=str, help="Plot 2D graph of the metrics", default=None)
    parser.add_argument("--no-legend", action="store_true", help="Do not show legend")

    parser.add_argument("--violin", action="store_true")
    parser.add_argument("--per-seed", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--part", type=str, default="all_parts", help="Wildcard defining which participants to show. (Default all)")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--scatter", action="store_true")
    parser.add_argument("--normalize", "-n", action="store_true", help="Normalize with respect to the human input dataset from generation 0")
    parser.add_argument("--shift", "-s", action="store_true", help="Shift with respect to the human input dataset from generation 0")
    parser.add_argument("--log", action="store_true", help="Log scale y axis")

    args = parser.parse_args()
    # args.assert_n_datapoints = False
    # print("NOOO ASSERTING datapoints")
    # time.sleep(1)

    print("N_dirs:", len(args.directories))
    args.directories = sorted(args.directories, key=lambda d: (d.split("/")[0], int(d.split("generated_")[1].split("_human")[0])))

    all_participant_jsons = list(itertools.chain(*[Path(d).glob(f"**/{args.part}/results.json") for d in args.directories]))

    all_participant_directories = sorted([j.parent for j in all_participant_jsons])

    print(f"Results found: {len(all_participant_directories)}")
    # load data
    all_results = load_results(all_participant_directories)
    all_colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys()) * 10

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
    label_metric_dict = defaultdict(lambda: defaultdict(list))
    label_linewith_dict = {}
    for dir, seed_part_dict in dir_seed_part_dict.items():
        for seed, parts_dict in seed_part_dict.items():
            for part, data in parts_dict.items():

                if 'n_participants' not in data:
                    data['n_participants'] = int(str(seed).split("participants_")[1].split("/generate")[0])

                # (n_generations, n_posts) or (n_generations)
                try:
                    scores = data[args.metric]
                except:
                    warnings.warn(f"Metric {args.metric} not found for {seed}. Available metrics: {data.keys()}.")
                    continue

                # if type(list(scores.keys())[0]) != int: scores = {int(k): v for k, v in scores.items()}
                assert type(list(scores.keys())[0]) == int

                # parse llm-judge metrics from string to int
                if args.metric.startswith("gpt4o-mini_quality") or args.metric.startswith("llama_quality"):
                    try:
                        scores = {g: list(map(int, s)) for g, s in scores.items()}

                    except:
                        warnings.warn(f"{args.metric} scores could not be converted to floats for {seed}.")
                        n_nans = np.array([list([s_ is None for s_ in s]) for g, s in scores.items()]).sum()
                        gens_with_nans = [g for g, s in scores.items() if any([s_ is None for s_ in s])]
                        print(f"Skipping {n_nans} nans found in the followings dir: {part} (generation {gens_with_nans})")
                        scores = {g: list([int(s_) for s_ in s if s_ is not None]) for g, s in scores.items()}

                if args.normalize:
                    # score of input dataset at generation 0
                    normalizing_score = np.mean(data[args.metric][0])
                    scores = {g: np.array(s) / normalizing_score for g, s in scores.items()}

                elif args.shift:
                    # score of input dataset at generation 0
                    normalizing_score = np.mean(data[args.metric][0])
                    scores = {g: np.array(s) - normalizing_score for g, s in scores.items()}

                interaction_metric_value = data[args.interaction_metric]  # e.g. AI ratio or ratio L-R ratio

                if args.interaction_plots:
                    # dir-wo_ratio -> (ratio, n_seeds)  -> different ratios in the same plot
                    label = create_label(dir, part, metric=args.metric, legend_conf=args.legend_conf)
                    if args.interaction_generation in scores:

                        if args.interaction_generation_smooth > 1:

                            end_ind = args.interaction_generation+1
                            start_ind = end_ind - args.interaction_generation_smooth

                            # average inside generations
                            selected_scores = [np.mean(scores[i]) for i in range(start_ind, end_ind)]

                            assert len(selected_scores) == args.interaction_generation_smooth, "Not enough datapoints to smooth"
                            # smooth over generations
                            last_gen_score = np.mean(selected_scores)
                        else:
                            last_gen_score = np.mean(scores[args.interaction_generation])

                        label_metric_dict[label][interaction_metric_value].append(last_gen_score)

                        if args.estimate_full_human_ratio_from == interaction_metric_value and args.interaction_metric == "ai_ratio":
                            gen_zero_score = np.mean(scores[0])
                            label_metric_dict[label][0.0].append(gen_zero_score)

                    else:
                        warnings.warn(f"No generation {args.interaction_generation} found in {part}")

                elif args.per_seed:
                    # dir+seed -> (n_generations, n_posts)
                    label = create_label(seed, part, interaction_metric=interaction_metric_value, metric=args.metric, legend_conf=args.legend_conf)

                    for gen_i, s in scores.items():
                        assert len(label_metric_dict[label][gen_i]) == 0, "each seed should be processed only once"
                        label_metric_dict[label][gen_i] = s

                else:
                    # dir -> (n_generations, n_seeds)
                    label = create_label(dir, part, interaction_metric=interaction_metric_value, metric=args.metric, legend_conf=args.legend_conf)

                    # some metrics are computer on a per-post basis (e.g. ce) so we average over posts
                    # for other metrics (e.g. diversity) this has no effect
                    scores = {g: np.mean(s) for g, s in scores.items()}
                    for gen_i, s in scores.items():
                        label_metric_dict[label][gen_i].append(s)

                if args.interaction_plots:
                    # label_linewith_dict[label] = interaction_metric_value  # ratio
                    label_linewith_dict[label] = np.log(data['n_participants']/5 + 1)
                else:
                    lw = interaction_metric_value
                    # lw = np.log(n_parts/5 + 1)
                    if label in label_linewith_dict:
                        assert label_linewith_dict[label] == lw, "things with the same label must have the same ratio to define linewidth"
                    else:
                        label_linewith_dict[label] = lw

    labels = list(label_metric_dict.keys())
    ys = [label_metric_dict[l] for l in labels]
    linewidths = [label_linewith_dict[l] * 20 for l in labels]
    linestyles = ["--" if "input" in l else "-" for l in labels]

    # label with the same color_id will have the same colors
    color_ids = [label_to_color_id(l) for l in labels]
    color_id_to_color_dict = dict(zip(set(color_ids), all_colors))
    color_id_to_color_dict = parse_colors_dict(color_id_to_color_dict, legend_conf=args.legend_conf)

    # load colors from cache (to ensure same colors in same figures)
    if os.path.isfile(".cache/colors.json"):
        with open('.cache/colors.json', 'r') as file:
            loaded_colors_dict = json.load(file)
        if loaded_colors_dict.keys() == color_id_to_color_dict.keys():
            cprint("Loading colors from cache", "red")
            color_id_to_color_dict = loaded_colors_dict
    with open('.cache/colors.json', 'w') as file:
        json.dump(color_id_to_color_dict, file, indent=4)

    colors = [color_id_to_color_dict[color_id] for color_id in color_ids]

    # ylabel = f"{'Relative ' if args.normalize else 'Absolute '}"+metric_name_parser(args.metric)
    ylabel = metric_name_parser(args.metric)
    xlabel = xlabel_parser(args.interaction_metric if args.interaction_plots else "Generation")

    label_dict = {
        "webis_reddit": "Webis Reddit",
        "100M_twitter": "100M Twitter",
        "reddit_submissions": "Reddit Submissions",
        "senators_tweets": "Senators Tweets",
        "wikipedia paragraphs": "Wikipedia",
    }
    labels = [label_dict.get(label, label) for label in labels]



    plot_and_save(
        # xs are ys.keys() and will be sorted
        ys=ys,  # shape (n_labels=n_curves, x_axis=n_generations/ratios, data_points=n_seed/n_posts)
        colors=colors,
        labels=labels,
        linewidths=linewidths,
        linestyles=linestyles,
        violin=args.violin,
        ylabel=ylabel,
        xlabel=xlabel,
        save_path=args.save_path,
        no_show=args.no_show,
        log=args.log,
        # ylim=(0, 1.2) if args.normalize else None,
        assert_n_datapoints=args.assert_n_datapoints,
        scatter=args.scatter,
        fontsize=40,
        label_fontsize=50,
        no_legend=args.no_legend,
        subplot_adjust_args={"left": 0.14, "right": 0.99, "top": 0.99, "bottom": 0.15},
    )