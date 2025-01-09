import argparse
import warnings
from collections import defaultdict
import json
import itertools
from pathlib import Path
import re
import matplotlib.colors as mcolors

from visualization_utils import *

# different label -> different curves
def create_label(dir, part_dir, interaction_metric=None, metric=None):  # different labels -> different curves
    if "senator" in str(dir):
        dataset = "senators_tweets"
    elif "100m" in str(dir):
        dataset = "100M_twitter"
    elif "webis" in str(dir):
        dataset = "reddit"
    elif "reddit_submissions" in str(dir):
        dataset = "reddit_submissions"
    else:
        dataset = "Unknown dataset"

    # we merge different seeds into same curve

    if "seed" in str(dir):
        seed_str = f" (seed={dir.name.split('_202')[0]}_{dir.name[:36][-9:]})"
    else:
        seed_str = ""

    n_part = str(dir).split("_participants_")[1].replace("/", "_").split("_")[0]
    part_str = f"(pop size: {n_part})"

    if interaction_metric is not None:
        # interaction plot
        interaction_metric_str = f"ratio : {interaction_metric}"
    else:
        interaction_metric_str = ""

    if "_type_" in str(dir):
        type = str(dir).split("_type_")[1].split("_")[0]
    else:
        type = "standard"

    if "hq_1" in str(dir):
        type = "hq"

    dataset_type_str = f"type: "+{"hq": "high quality", "mq": "mid quality", "ld": "low diversity"}[type] if type != "standard" else ""

    return f"{dataset} {part_str} {dataset_type_str} {interaction_metric_str} {seed_str}"

def label_to_color_id(label):  # label with different color_id -> different colors
    # we remove all after generated_ -> different ratios and seed have same color
    # return label.split("pop size: ")[1].split(")")[0]

    # return label.split("ratio : ")[1]
    try:
        # return label.split("(seed=")[1] # different colors
        return label.split("seed")[0]  # per seed colors
    except:
        return label.split("ratio")[0]

    # return label.split("generated_")[0]


# this can be used to fix some experiment_ids to colors
def parse_colors_dict(colors_dict):
    # return colors_dict

    for d, c in colors_dict.items():
        # colors_dict[d] = plt.cm.gray(0.3)
        if "high quality" in d:
            colors_dict[d] = "tab:green"
        elif "mid quality" in d:
            colors_dict[d] = "tab:red"
        else:
            colors_dict[d] = "tab:blue"

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

    parser.add_argument("--violin", action="store_true")
    parser.add_argument("--per-seed", action="store_true")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--part", type=str, default="all_parts", help="Wildcard defining which participants to show. (Default all)")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--scatter", action="store_true")
    parser.add_argument("--normalize", "-n", action="store_true", help="Normalize with respect to the human input dataset from generation 0")
    parser.add_argument("--log", action="store_true", help="Log scale y axis")

    args = parser.parse_args()

    print("N_dirs:", len(args.directories))
    args.directories = sorted(args.directories, key=lambda d: (d.split("/")[0], int(d.split("generated_")[1].split("_human")[0])))

    all_participant_jsons = list(itertools.chain(*[Path(d).glob(f"**/{args.part}/results.json") for d in args.directories]))

    all_participant_directories = sorted([j.parent for j in all_participant_jsons])

    print(f"Results found: {len(all_participant_directories)}")
    # load data
    all_results = load_results(all_participant_directories)
    all_colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys())

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
                    # human_input_dataset_score = np.mean(data["input_"+args.metric][0]) #input
                    human_input_dataset_score = np.mean(data[args.metric][0])
                    scores = {g: np.array(s)/human_input_dataset_score for g, s in scores.items()}

                interaction_metric_value = data[args.interaction_metric]  # e.g. AI ratio or ratio L-R ratio

                if args.interaction_plots:
                    # dir-wo_ratio -> (ratio, n_seeds)  -> different ratios in the same plot
                    label = create_label(dir, part, metric=args.metric)
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
                    label = create_label(seed, part, interaction_metric=interaction_metric_value, metric=args.metric)

                    for gen_i, s in scores.items():
                        assert len(label_metric_dict[label][gen_i]) == 0, "each seed should be processed only once"
                        label_metric_dict[label][gen_i] = s

                else:
                    # dir -> (n_generations, n_seeds)
                    label = create_label(dir, part, interaction_metric=interaction_metric_value, metric=args.metric)

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
    linewidths = [label_linewith_dict[l] * 5 for l in labels]
    linestyles = ["--" if "input" in l else "-" for l in labels]

    # label with the same color_id will have the same colors
    color_ids = [label_to_color_id(l) for l in labels]
    color_id_to_color_dict = dict(zip(set(color_ids), all_colors))
    color_id_to_color_dict = parse_colors_dict(color_id_to_color_dict)

    # load colors from cache
    with open('.cache/colors.json', 'r') as file: loaded_colors_dict = json.load(file)
    if loaded_colors_dict.keys() == color_id_to_color_dict.keys(): color_id_to_color_dict = loaded_colors_dict
    with open('.cache/colors.json', 'w') as file: json.dump(color_id_to_color_dict, file, indent=4)

    colors = [color_id_to_color_dict[color_id] for color_id in color_ids]

    ylabel = metric_name_parser(args.metric)

    plot_and_save(
        # xs are ys.keys() and will be sorted
        ys=ys,  # shape (n_labels=n_curves, x_axis=n_generations/ratios, data_points=n_seed/n_posts)
        colors=colors,
        labels=labels,
        linewidths=linewidths,
        linestyles=linestyles,
        violin=args.violin,
        ylabel=ylabel,
        xlabel=args.interaction_metric if args.interaction_plots else "Generation",
        save_path=args.save_path,
        no_show=args.no_show,
        log=args.log,
        assert_n_datapoints=args.assert_n_datapoints,
        scatter=args.scatter,
        fontsize=30,
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
            