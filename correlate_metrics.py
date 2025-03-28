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
import pickle


def load_cluster_data(cl_index):
    with open("./data/webis/selected_clusters_indices_to_path.json", 'r') as f:
        cluster_index_to_path_dict = json.load(f)

    cluster_path = cluster_index_to_path_dict[str(cl_index)]
    results_path = "./aggregated_clustering_results/" + cluster_path.replace("./", "").replace("/", "_") + ".pkl"
    with open(results_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics


# different label -> different curves

def parse_scores(scores):
    if type(scores) in [float, int]:
        score = np.mean(scores)
    else:
        score = np.mean(list(map(float, scores)))
    return score


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
    parser.add_argument("--part", type=str, default="all_parts", help="Wildcard defining which participants to show. (Default all)")
    parser.add_argument("--metrics", nargs="+", type=str)
    parser.add_argument("--generations", nargs="+", type=int, default=[0, 19])
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--human-dataset", "-hm", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--arrows", action="store_true")
    parser.add_argument("--normalize", "-n", action="store_true", help="Normalize with respect to the human input dataset from generation 0")

    args = parser.parse_args()

    print("N_dirs:", len(args.directories))
    args.directories = sorted(args.directories, key=lambda d: int(d.split("generated_")[1].split("_human")[0]))

    all_participant_jsons = list(itertools.chain(*[Path(d).glob(f"**/{args.part}/results.json") for d in args.directories]))

    all_participant_directories = sorted([j.parent for j in all_participant_jsons])

    print(f"Results found: {len(all_participant_directories)}")
    # load data
    all_results = load_results(all_participant_directories)


    datapoints = defaultdict(list)
    human_dataset_metrics = defaultdict(list)
    human_clusters = defaultdict(list)
    colors = defaultdict(list)
    markers = defaultdict(list)
    for dir, res in all_results.items():
        if "llama_quality_scale_cap_250" not in res:
            continue
        for generation in args.generations:
            for metric in args.metrics:

                if metric == "text_len_cap_250":
                    score = parse_scores([len(t) for t in res["text_cap_250"][generation]])
                    assert not args.normalize
                else:
                    score = parse_scores(res[metric][generation])

                if args.normalize:
                    normalize = parse_scores(res[metric][0])
                    score = score / normalize

                if args.human_dataset:
                    if "cluster" in str(dir):
                        cluster_id = int(str(dir).split("cluster_")[1].split("_part")[0])
                        cluster_data = load_cluster_data(cluster_id)
                        cluster_metric_mapper = {
                            "llama_quality_scale_cap_250": "llama_quality_scale",
                            "cos_diversity_stella_cap_250": "cos_diversity_cap_250",
                        }
                        human_dataset_metrics[metric].append(cluster_data[cluster_metric_mapper[metric]])
                        human_clusters[metric].append((cluster_id, cluster_data))
                    else:
                        assert "type_standard" in str(dir)
                        if "webis_reddit" in str(dir):
                            met_dict = {
                                "llama_quality_scale_cap_250":  67.896,
                                "cos_diversity_stella_cap_250": 0.6915946910987972,
                            }
                        elif "reddit_submissions" in str(dir):
                            met_dict = {
                                "llama_quality_scale_cap_250": 57.412,
                                "cos_diversity_stella_cap_250": 0.6842812096121778,
                            }
                        elif "100m_tweets" in str(dir):
                            met_dict = {
                                "llama_quality_scale_cap_250": 39.392,
                                "cos_diversity_stella_cap_250": 0.6508998586914287,
                            }
                        elif "senator_tweets" in str(dir):
                            met_dict = {
                                "llama_quality_scale_cap_250": 66.196,
                                "cos_diversity_stella_cap_250": 0.629831525210795,
                            }

                        human_dataset_metrics[metric].append(met_dict[metric])


                datapoints[metric].append(score)
                colors[metric].append(res['human_dataset'])
                markers[metric].append(f"Gen {generation}")

    all_colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.CSS4_COLORS.keys())

    # this will be used to merge runs into seeds later

    m1, m2 = datapoints.keys()
    data_m1, data_m2 = np.array(datapoints[m1]), np.array(datapoints[m2])
    assert len(data_m1) == len(data_m2)

    color_ids = list(set(colors[m1]))
    id_2_col = dict(zip(sorted(color_ids), all_colors))
    cs = [id_2_col[c] for c in colors[m1]]
    assert colors[m1] == colors[m2]

    marker_ids = list(set(markers[m1]))
    id_2_mar = dict(zip(sorted(marker_ids), ["o", "x", "*"]))
    ms = [id_2_mar[c] for c in marker_ids]

    print("Number of datapoints found: ", len(data_m1))
    print(f"Metrics: {args.metrics}")

    spearman_corr, _ = spearmanr(data_m1, data_m2)
    print(f"Spearman: {spearman_corr}")
    pearson_corr, _ = pearsonr(data_m1, data_m2)
    print(f"Pearson: {pearson_corr}")

    # fit regression
    slope, intercept = np.polyfit(data_m1, data_m2, 1)  # Linear regression (degree=1)

    if args.arrows:
        # plot arrow
        inds = np.where(np.array(markers[m1]) == "Gen 0")[0]
        assert all(inds == np.where(np.array(markers[m2]) == "Gen 0")[0])
        X, Y = data_m1[inds], data_m2[inds]
        cs_ = np.array(cs)[inds]

        inds = np.where(np.array(markers[m1]) == "Gen 19")[0]
        assert all(inds == np.where(np.array(markers[m2]) == "Gen 19")[0])
        X_, Y_ = data_m1[inds], data_m2[inds]
        U, V = X_-X, Y_-Y
        plt.quiver(
            X, Y, U, V, color = cs_, angles = 'xy', scale_units = 'xy', scale = 1, headwidth = 3, headlength = 5, width = 0.001
        )
        # dataset metrics
        if args.human_dataset:
            DX, DY = np.array(human_dataset_metrics[m1])[inds], np.array(human_dataset_metrics[m2])[inds]
            plt.quiver(
                DX, DY, X-DX, Y-DY, color = cs_, angles = 'xy', scale_units = 'xy', scale = 1,
                headwidth = 10, headlength = 10,
                width = 0.001
            )

        ### avg starting points
        for color in np.unique(cs_):
            color_inds = np.where(cs_ == color)[0]

            avg_start_x = np.mean(X[color_inds])
            avg_start_y = np.mean(Y[color_inds])
            avg_end_x = np.mean(X_[color_inds])
            avg_end_y = np.mean(Y_[color_inds])

            plt.plot([avg_start_x, avg_end_x], [avg_start_y, avg_end_y], '--', color='black')
            distance = np.sqrt((avg_end_x - avg_start_x) ** 2 + (avg_end_y - avg_start_y) ** 2)
            plt.text(((avg_start_x + avg_end_x) / 2)*1.02, ((avg_start_y + avg_end_y) / 2)*0.98, f'd={distance:.2f}', color='black', fontweight="bold", fontsize=15)

            plt.plot(avg_start_x, avg_start_y, 'o', color=color, markersize=15, markeredgecolor='black', markeredgewidth=1.5)
            plt.plot(avg_end_x, avg_end_y, 'x', color="black", markersize=20, markeredgewidth=2.0)
            plt.plot(avg_end_x, avg_end_y, 'x', color=color, markersize=15, markeredgewidth=1.5)

            if args.human_dataset:
                avg_start_dx = np.mean(DX[color_inds])
                avg_start_dy = np.mean(DY[color_inds])
                print(f"avg_start_dx: {avg_start_dx}, avg_start_dy: {avg_start_dy}")
                plt.plot(avg_start_dx, avg_start_dy, 's', color=color, markersize=15, markeredgecolor='black', markeredgewidth=1.5)

                for i, (tx, ty, clst) in enumerate(zip(DX, DY, np.array(human_clusters[m1])[color_inds])):
                    print(f"{clst[0]}")
                    plt.text(tx, ty, f"{clst[0]}", color='black', fontsize=10)




        if args.human_dataset:
            x_min = np.min([X, X_, DX])
            y_min = np.min([Y, Y_, DY])
            x_max = np.max([X, X_, DX])
            y_max = np.max([Y, Y_, DY])
        else:
            x_min = np.min([X, X_])
            y_min = np.min([Y, Y_])
            x_max = np.max([X, X_])
            y_max = np.max([Y, Y_])
        print(x_min, x_max)
        print(y_min, y_max)

        plt.xlim(x_min*0.95, x_max*1.05)
        plt.ylim(y_min*0.95, y_max*1.05)
        # plt.gca().set_aspect('equal')

        # Create a custom legend that includes both the regression line and color meanings
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color=id_2_col[c_id], label=c_id, linestyle="None") for c_id in color_ids
        ]
        legend_title = f""

    else:
        # Create the plot
        # Create separate scatter plots for each marker type
        for m in marker_ids:
            inds = np.where(np.array(markers[m1]) == m)[0]
            plt.scatter(data_m1[inds], data_m2[inds], c=np.array(cs)[inds], marker=id_2_mar[m])

        # Plot regression line
        line = np.polyval([slope, intercept], data_m1)  # Evaluate the line
        plt.plot(data_m1, line, color='red', label=f"Regression line\ny = {slope:.2f}x + {intercept:.2f}")

        # Create a custom legend that includes both the regression line and color meanings
        legend_elements = [
            plt.Line2D([0], [0], color='red', label=f"Regression line\ny = {slope:.2f}x + {intercept:.2f}")
        ] + [
            plt.Line2D([0], [0], marker="o", color=id_2_col[c_id], label=c_id, linestyle="None") for c_id in color_ids
        ] + [
            plt.Line2D([0], [0], marker=id_2_mar[m_id], color="black", label=m_id, linestyle="None") for m_id in marker_ids
        ]
        legend_title = f"Spearman r = {spearman_corr:.2f}\nPearson r = {pearson_corr:.2f}"

    plt.xlabel(m1)
    plt.ylabel(m2)

    # plt.title(f"Correlations {args.metrics}")
    plt.legend(handles=legend_elements, title=legend_title)
    plt.grid(True)

    if not args.no_show:
        plt.show()

    if args.save_path:
        plt.savefig(args.save_path+".png", dpi=300)
        print(f"Saved to: {args.save_path}")

