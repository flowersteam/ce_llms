import os
import glob
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import random


def load_per_cluster_results(files):
    all_results = {}
    for filepath in files:
        with open(filepath, 'rb') as f:
            metrics = pickle.load(f)

        c_path = metrics['cluster_path']
        all_results[c_path] = metrics
    return all_results


def extract_cluster_metrics(cluster_results):
    metric_values = defaultdict(list)

    for cluster, cluster_metrics in cluster_results.items():
        for metric in cluster_metrics:
            metric_values[metric].append(cluster_metrics[metric])

    return dict(metric_values)


def remove_outlier_clusters_iqr(cluster_results, metric_values, selection_metrics=None):
    # calculate metric safe intervals (lower and upper 10%)
    metric_intervals = {}
    if selection_metrics is None:
        selection_metrics = metric_values.keys()
    print("Selecting based on {} metrics".format(len(selection_metrics)))

    # # interval selection
    # edge_pc = 1
    # for metric in selection_metrics:
    #     values = metric_values[metric]
    #     lower = np.percentile(values, edge_pc)
    #     upper = np.percentile(values, 100-edge_pc)
    #     metric_intervals[metric] = (lower, upper)

    # IQR outlier removal
    # for metric in selection_metrics:
    #     values = metric_values[metric]
    #     q1 = np.percentile(values, 25)  # First quartile (25th percentile)
    #     q3 = np.percentile(values, 75)  # Third quartile (75th percentile)
    #     iqr = q3 - q1  # Interquartile range
    #     lower = q1 - (1.5 * iqr)
    #     upper = q3 + (1.5 * iqr)
    #     metric_intervals[metric] = (lower, upper)

    outlier_clusters = set()
    for cluster, c_metrics in cluster_results.items():
        for m, vs in c_metrics.items():
            if m not in metric_intervals:
                continue
            if vs < metric_intervals[m][0] or vs > metric_intervals[m][1]:
                outlier_clusters.add(cluster)

    # remove outlier dataset
    cluster_results = {k: v for k, v in cluster_results.items() if k not in outlier_clusters}
    return cluster_results



def remove_center_clusters(cluster_results, metric_values, selection_metrics=None, max_remove=None):
    # calculate metric safe intervals (lower and upper 10%)
    metric_intervals = {}
    if selection_metrics is None:
        selection_metrics = metric_values.keys()
    print("Selecting based on {} metrics".format(len(selection_metrics)))

    # # interval selection
    # edge_pc = 1
    # for metric in selection_metrics:
    #     values = metric_values[metric]
    #     lower = np.percentile(values, edge_pc)
    #     upper = np.percentile(values, 100-edge_pc)
    #     metric_intervals[metric] = (lower, upper)

    # IQR outlier removal
    for metric in selection_metrics:
        values = metric_values[metric]
        q1 = np.percentile(values, 25)  # First quartile (25th percentile)
        q3 = np.percentile(values, 75)  # Third quartile (75th percentile)
        iqr = q3 - q1  # Interquartile range
        lower = q1 - (0.5 * iqr)
        upper = q3 + (0.5 * iqr)
        metric_intervals[metric] = (lower, upper)

    cluster_in_centers = {}  # cluster -> metric -> list of bools indicating if it's in the center
    for cluster, c_metrics in cluster_results.items():
        cluster_in_centers[cluster] = []
        for m, vs in c_metrics.items():
            if m not in metric_intervals:
                continue

            if vs > metric_intervals[m][0] and vs < metric_intervals[m][1]:
                cluster_in_centers[cluster].append(True)
            else:
                cluster_in_centers[cluster].append(False)


    to_remove = set()
    for cluster, in_centers in cluster_in_centers.items():
        if np.mean(in_centers) > 0.5:
            to_remove.add(cluster)

    if max_remove is not None:
        to_remove = random.sample(to_remove, int(max_remove*len(to_remove)))  # subsample
    print("To remove {} center clusters".format(len(to_remove)))


    # remove outlier dataset
    cluster_results = {k: v for k, v in cluster_results.items() if k not in to_remove}
    return cluster_results

def plot_metric_distributions_histogram(metric_values_dict, save_path=None, title="", selection_metrics=[],
                                        show_metrics=None):
    if show_metrics:
        metric_values_dict = {
            l: {
                m: v for m, v in c.items() if m in show_metrics
            } for l, c in metric_values_dict.items()
        }

    n_metrics = len(list(metric_values_dict.values())[0])
    f, a = plt.subplots(n_metrics)
    f.set_size_inches(15, 14)
    a = a.ravel()

    for i, (label, metric_values) in enumerate(metric_values_dict.items()):
        metrics = list(metric_values.keys())

        for idx, ax in enumerate(a):
            metric = metrics[idx]
            values = metric_values[metric]
            values = [v for v in values if not np.isinf(v)]

            # Calculate histogram
            counts, bins, _ = ax.hist(
                values, bins=200, alpha=0.3, label=label, density=True,
            )

            ax.set_yticks([])

            # Set x-axis limits and ticks
            # m_min = np.min(values)
            # m_max = np.max(values)
            # ticks = list(ax.get_xticks()) + [m_min, m_max]
            # ticks = [round(t, 2) for t in ticks]
            # ax.set_xticks(ticks)
            # ax.set_ylim(-0.1, 1.1)

            # Color the metric label if it's a selection metric
            c = 'red' if metric in selection_metrics else 'black'
            ax.set_ylabel(metric, rotation=0, labelpad=10, ha='right', color=c)

            # Remove left, right, and top borders
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    plt.suptitle(title)
    plt.legend(loc="lower right")
    # plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.03, top=0.99, left=0.15)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print("Plot saved to:", save_path)

    plt.show()


def plot_metric_distributions(metric_values_dict, save_path=None, title="", selection_metrics=[], show_metrics=None):
    if show_metrics:
        metric_values_dict = {
            l: {
                m: v for m, v in c.items() if m in show_metrics
            } for l, c in metric_values_dict.items()
        }

    n_metrics = len(list(metric_values_dict.values())[0])
    print(n_metrics)
    f, a = plt.subplots(n_metrics)
    f.set_size_inches(15, 14)
    a = a.ravel()

    # Reverse the items list before plotting
    items = list(metric_values_dict.items())[::-1]

    for i, (label, metric_values) in enumerate(items):
        metrics = list(metric_values.keys())

        for idx, ax in enumerate(a):
            metric = metrics[idx]
            # y = (len(items)-1-i)/len(metric_values_dict)*np.ones_like(metric_values[metric])
            y = (i)/(len(metric_values_dict)-1)*np.ones_like(metric_values[metric])
            ax.scatter(metric_values[metric], y, alpha=0.5, label=label, s=8)
            # ax.scatter(np.median(metric_values[metric]), y[0], marker="x", color="black", s=70)
            # ax.scatter(np.percentile(metric_values[metric], 25), y[0], marker="|", color="black", s=70)
            # ax.scatter(np.percentile(metric_values[metric], 75), y[0], marker="|", color="black", s=70)

            ax.set_yticks([])

            metric_values[metric] = [v for v in metric_values[metric] if not np.isinf(v)]
            # append min and max to xticks
            m_min = np.min(metric_values[metric])
            m_max = np.max(metric_values[metric])
            m_var = np.var(metric_values[metric])
            ticks = list(ax.get_xticks()) + [m_min, m_max]
            ticks = [round(t, 2) for t in ticks]
            ax.set_xticks(ticks)
            ax.set_ylim(-0.1, 1.1)

            c = 'red' if metric in selection_metrics else 'black'
            ax.set_ylabel(metric, rotation=0, labelpad=10, ha='right', color=c)

            # Remove left, right, and top borders
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    plt.suptitle(title)
    plt.legend(loc="lower right")
    # plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print("Plot saved to:", save_path)
    plt.show()


def plot_metric_correlations(metric_values_dict, save_path=None, show_metrics=None):
    import seaborn as sns

    if show_metrics is None:
        metrics = [m for m in metric_values_dict.keys() if m != "cluster_path" and not m.startswith("text_cap_")]
    else:
        metrics = show_metrics

    # Create correlation matrix
    corr_matrix = np.zeros((len(metrics), len(metrics)))

    # add QD by force

    # all
    # metric_values_dict["text_len_cap_80000"].extend([340.11505, 376.70835, 425.7857, 501.64195, 290.4489, 529.7946, 751.51875])
    # metric_values_dict["llama_quality_scale"].extend([20.0, 40.0, 60.0, 80.0, 58.07095, 68.93925, 73.7238])

    # no Q20
    # metric_values_dict["text_len_cap_80000"].extend([376.70835, 425.7857, 501.64195, 290.4489, 529.7946, 751.51875])
    # metric_values_dict["llama_quality_scale"].extend([40.0, 60.0, 80.0, 58.07095, 68.93925, 73.7238])

    # only Q
    # metric_values_dict["text_len_cap_80000"].extend([376.70835, 425.7857, 501.64195, 290.4489, 529.7946, 751.51875])
    # metric_values_dict["llama_quality_scale"].extend([40.0, 60.0, 80.0, 58.07095, 68.93925, 73.7238])

    # Compute correlations between each pair of metrics
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            corr = np.corrcoef(metric_values_dict[metric1], metric_values_dict[metric2])[0, 1]
            corr_matrix[i, j] = corr

    # compute variance inflection index (VIF)
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = np.array([metric_values_dict[m] for m in metrics]).T  # shape=(n_clusters, n_metrics)==(n_samples, n_features)
    vif = np.array([variance_inflation_factor(X, i) for i in range(len(metrics))])
    vif_dict = dict(zip(metrics, vif))

    # Create figure
    plt.figure(figsize=(15, 12))

    # Create heatmap
    sns.heatmap(
        np.abs(corr_matrix), # absolute
        xticklabels=metrics,
        yticklabels=[m+f"\n({vif_dict[m]})" for m in metrics],
        # annot=True,  # Show correlation values
        annot=corr_matrix,  # Show correlation values
        fmt='.2f',  # Format to 2 decimal places
        cmap='RdBu_r',  # Red-Blue diverging colormap
        vmin=0,
        # vmin=-1,
        vmax=1,
        center=0
    )

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.title('Absolute Metric Correlations')
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print("Correlation plot saved to:", save_path)

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze clustering results')
    # parser.add_argument('files', nargs='+')
    parser.add_argument('-o', '--save-path', default='viz_results/webis_selected_clusters/')
    # parser.add_argument('-sem', '--selection-metrics', nargs='+', default=[
    #     'text_len_cap_80000',
    #     "gaussian_aic_cap_80000",
    #     'word_entropy_cap_250',
    #     'kl_entropy_cap_80000_k_25',
    #     'kl_entropy_cap_250_k_25',
    #     'llama_quality_scale',
    #     'diversity_selfbleu_cap_250',
    #     'cos_diversity_cap_250',
    # ])
    parser.add_argument('-sem', '--selection-metrics', nargs='+', default=[
        # 'text_len_cap_250',
        # 'post_ttr_truncated_len_cap_250',
        # 'ttr_cap_250',
        # 'n_unique_words_total_cap_250',
        # 'n_unique_posts_cap_250',
        # 'pc_unique_posts_cap_250',
        # 'text_len_cap_10000',
        # 'post_ttr_truncated_len_cap_10000',
        # 'ttr_cap_10000',
        # 'n_unique_words_total_cap_10000',
        'llama_quality_scale',
        # 'llama_quality_scale_cap_250',
        # 'llama_quality_scale_cap_10000',
        # 'llama_quality_scale_cap_80000',
        # 'word_entropy_cap_250',
        # 'word_entropy_cap_10000',
        # 'diversity_selfbleu_cap_250',
        # 'diversity_selfbleu_cap_10000',
        # 'cos_diversity_cap_250',
        # 'cos_diversity_cap_10000',
        # 'knn_5_cos_diversity_cap_250',
        # 'knn_10_cos_diversity_cap_250',
        # 'knn_25_cos_diversity_cap_250',
        # 'knn_50_cos_diversity_cap_250',
        # 'knn_5_cos_diversity_cap_10000',
        # 'knn_10_cos_diversity_cap_10000',
        # 'knn_25_cos_diversity_cap_10000',
        # 'knn_50_cos_diversity_cap_10000',
        # 'knn_100_cos_diversity_cap_10000',
        # 'knn_1000_cos_diversity_cap_10000',
        # 'kl_entropy_cap_250_k_5',
        # # 'kl_entropy_cap_250_k_10',
        # # 'kl_entropy_cap_250_k_25',
        # 'kl_entropy_cap_250_k_50',
        # 'kl_entropy_cap_10000_k_5',
        # # 'kl_entropy_cap_10000_k_10',
        # # 'kl_entropy_cap_10000_k_25',
        # 'kl_entropy_cap_10000_k_50',
        # 'kl_entropy_cap_10000_k_100',
        # 'kl_entropy_cap_10000_k_1000',
        # 'kl_entropy_cap_10000_k_2000',
        'gaussian_aic_cap_10000',
        # 'toxicity_cap_250', 'toxicity_cap_10000',
        # 'positivity_cap_250', 'positivity_cap_10000'
        # 'aggregate_reading_level_cap_250',
        # 'aggregate_reading_level_cap_10000',
    ])

    parser.add_argument('-shm', '--show-metrics', nargs='+', default=[
        'text_len_cap_250',
        'post_ttr_truncated_len_cap_250',
        'ttr_cap_250',
        'n_unique_words_total_cap_250',
        'text_len_cap_10000',
        'post_ttr_truncated_len_cap_10000',
        'ttr_cap_10000',
        'n_unique_words_total_cap_10000',
        'llama_quality_scale',
        'llama_quality_scale_cap_250',
        'word_entropy_cap_250',
        'word_entropy_cap_10000',
        'diversity_selfbleu_cap_250',
        'diversity_selfbleu_cap_10000',
        'cos_diversity_cap_250',
        'cos_diversity_cap_10000',
        'knn_5_cos_diversity_cap_250',
        'knn_50_cos_diversity_cap_250',
        'knn_5_cos_diversity_cap_10000',
        'knn_50_cos_diversity_cap_10000',
        'knn_100_cos_diversity_cap_10000',
        'knn_1000_cos_diversity_cap_10000',
        'kl_entropy_cap_250_k_5',
        'kl_entropy_cap_250_k_50',
        'kl_entropy_cap_10000_k_5',
        'kl_entropy_cap_10000_k_50',
        'kl_entropy_cap_10000_k_100',
        'kl_entropy_cap_10000_k_1000',
        'kl_entropy_cap_10000_k_2000',
        'gaussian_aic_cap_10000',
        # 'toxicity_cap_250', 'toxicity_cap_10000',
        # 'positivity_cap_250', 'positivity_cap_10000'
        # 'aggregate_reading_level_cap_250',
        # 'aggregate_reading_level_cap_10000',
    ])

    np.random.seed(42)
    random.seed(42)


    args = parser.parse_args()

    args.show_metrics = [
        'text_len_cap_10000',
        # 'post_ttr_truncated_len_cap_10000',
        'ttr_cap_10000',
        # 'n_unique_words_total_cap_10000',
        'llama_quality_scale',
        'word_entropy_cap_10000',
        'diversity_selfbleu_cap_10000',  # acctualy 1000
        'cos_diversity_cap_10000',
        # 'knn_5_cos_diversity_cap_250',
        # 'knn_50_cos_diversity_cap_250',
        # 'knn_5_cos_diversity_cap_10000',
        'knn_50_cos_diversity_cap_10000',
        # 'knn_100_cos_diversity_cap_10000',
        'knn_1000_cos_diversity_cap_10000',
        # 'kl_entropy_cap_250_k_5',
        # 'kl_entropy_cap_250_k_50',
        'kl_entropy_cap_10000_k_5',
        # 'kl_entropy_cap_10000_k_50',
        # 'kl_entropy_cap_10000_k_100',
        'kl_entropy_cap_10000_k_1000',
        # 'kl_entropy_cap_10000_k_2000',
        'gaussian_aic_cap_10000',
        # 'toxicity_cap_250', 'toxicity_cap_10000',
        # 'positivity_cap_250', 'positivity_cap_10000'
        # 'aggregate_reading_level_cap_250',
        # 'aggregate_reading_level_cap_10000',
    ]

    all_results = {}
    metric_values_dict = {}
    title = ""

    files = glob.glob("aggregated_clustering_results/*.pkl")
    cluster_results = load_per_cluster_results(files)
    per_metric_values = extract_cluster_metrics(cluster_results)
    metric_values_dict[f"full ({len(cluster_results)})"] = per_metric_values
    # plot_metric_correlations(per_metric_values, save_path=args.save_path + f'/all_clusters_correlations.png', show_metrics=args.show_metrics)

    # select clusters
    # remove outliers
    # selected_cluster_results = remove_outlier_clusters_iqr(
    #     cluster_results, per_metric_values, selection_metrics=args.selection_metrics
    # )
    # selected_cluster_results = remove_center_clusters(
    #     selected_cluster_results, per_metric_values, selection_metrics=args.selection_metrics, max_remove=0.5
    # )
    selected_cluster_results = dict(random.sample(list(cluster_results.items()), 200))
    selected_per_metric_values = extract_cluster_metrics(selected_cluster_results)
    metric_values_dict[f"selected ({len(selected_cluster_results)})"] = selected_per_metric_values

    plot_metric_correlations(selected_per_metric_values, save_path=args.save_path + f'/selected_clusters_correlations.png', show_metrics=args.show_metrics)
    print("PLOTTED correlations and extiting")
    exit()

    plot_metric_distributions(
        metric_values_dict, title=title,
        save_path=args.save_path + f'/metrics_distributions.png',
        selection_metrics=args.selection_metrics, show_metrics=args.show_metrics
    )

    plot_metric_distributions_histogram(
        metric_values_dict, title=title,
        save_path=args.save_path + f'/metrics_distributions_histogram.png',
        selection_metrics=args.selection_metrics, show_metrics=args.show_metrics
    )

    # save
    save_path = args.save_path + f'/clustering_metrics_distributions.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(metric_values_dict, f)
        print(f"Metrics distributions saved to: {save_path}")

    print(list(selected_cluster_results)[:10])

    # plot correlations between metrics
    # cluster_index_to_path_dict_path = "./data/webis/cluster_index_to_path.json" # old
    # cluster_index_to_path_dict_path = "./data/webis/cluster_index_to_path_temp.json" # temp
    cluster_index_to_path_dict_path = "./data/webis/selected_clusters_indices_to_path.json"
    cluster_index_to_path_dict = {cluster_index: cluster for cluster_index, cluster in enumerate(selected_cluster_results)}
    with open(cluster_index_to_path_dict_path, 'w') as f:
        json.dump(cluster_index_to_path_dict, f, indent=4)
        print("Cluster index to path dict saved to:", cluster_index_to_path_dict_path)



