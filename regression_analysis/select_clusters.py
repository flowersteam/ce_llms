import os
from sklearn.preprocessing import StandardScaler
import argparse
import pickle
import numpy as np
import json
from collections import defaultdict
import random

import matplotlib.pyplot as plt

from regression_analysis.regression_analysis_utils import load_per_cluster_results, parse_metrics

# for correlations
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor




def extract_metrics_distributions(cluster_results):
    metric_distributions = defaultdict(list)

    for cluster, cluster_metrics in cluster_results.items():
        for metric, value in cluster_metrics.items():
            metric_distributions[metric].append(value)

    # Convert lists to numpy arrays
    metric_distributions = {metric: np.array(values) for metric, values in metric_distributions.items()}

    return dict(metric_distributions)


def plot_metric_distributions_histogram(metric_values_dict, save_path=None, title="", show_metrics=None):
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
            counts, bins, _ = ax.hist(values, bins=200, alpha=0.3, label=label+f" ({len(values)})", density=True)

            ax.set_yticks([])
            ax.set_ylabel(metric, rotation=0, labelpad=10, ha='right', color="black")

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


def plot_metric_distributions(metric_values_dict, save_path=None, title="", show_metrics=None):
    if show_metrics:
        metric_values_dict = {
            l: {
                m: v for m, v in c.items() if m in show_metrics
            } for l, c in metric_values_dict.items()
        }

    n_metrics = len(list(metric_values_dict.values())[0])
    print("Number of metrics:", n_metrics)
    f, a = plt.subplots(n_metrics)
    f.set_size_inches(15, 14)
    a = a.ravel()

    # Reverse the items list before plotting
    items = list(metric_values_dict.items())[::-1]

    for i, (label, metric_values) in enumerate(items):
        metrics = list(metric_values.keys())

        for idx, ax in enumerate(a):
            metric = metrics[idx]
            y = (len(items)-1-i)/len(metric_values_dict)*np.ones_like(metric_values[metric])
            ax.scatter(metric_values[metric], y, alpha=0.5, label=label+f"({len(y)})", s=8)
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

            ax.set_ylabel(metric, rotation=0, labelpad=10, ha='right', color="black")

            # Remove left, right, and top borders
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    # Adjust layout: tight top/bottom margins and increased vertical spacing between subplots
    f.subplots_adjust(top=0.95, bottom=0.05, hspace=0.6)

    plt.suptitle(title)
    plt.legend(loc="lower right")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print("Plot saved to:", save_path)
    plt.show()


def plot_metric_correlations(metric_values_dict, save_path=None, show_metrics=None):

    if show_metrics is None:
        metrics = [m for m in metric_values_dict.keys() if m != "cluster_path" and not m.startswith("text_cap_")]
    else:
        metrics = show_metrics

    # Create correlation matrix
    corr_matrix = np.zeros((len(metrics), len(metrics)))

    # Compute correlations between each pair of metrics
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            corr = np.corrcoef(metric_values_dict[metric1], metric_values_dict[metric2])[0, 1]
            corr_matrix[i, j] = corr

    #  compute VIF
    X = np.array([metric_values_dict[m] for m in metrics]).T  # shape=(n_clusters, n_metrics)==(n_samples, n_features)

    # print({m: bool(np.isinf(metric_values_dict[m]).any()) for m in metrics})
    X_scaled = StandardScaler().fit_transform(X)
    vif = np.array([variance_inflation_factor(X_scaled, i) for i in range(len(metrics))])

    vif_dict = dict(zip(metrics, vif))

    # Create figure
    plt.figure(figsize=(15, 12))

    # Create heatmap
    sns.heatmap(
        np.abs(corr_matrix),  # absolute
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
    parser.add_argument('--cluster-evaluation-files', required=True,  nargs='+', default=["data/webis/webis_dataset_clusters/evaluation/umap_(5,0.0001)_gmm_(5)_cluster_3.pkl"])
    parser.add_argument('--visualization-dir', required=True, default='viz_results/webis/clusters')
    parser.add_argument('--selection-save-path', required=True, default="./data/webis/selected_clusters_indices_to_path.json")
    parser.add_argument('--show-metrics', nargs='+', default=[
        'text_len_cap_10000',
        'ttr_cap_10000',
        'n_unique_words_total_cap_10000',
        'llama_quality_scale',
        'word_entropy_cap_10000',
        'diversity_selfbleu_cap_500',
        'cos_diversity_cap_10000',
        'knn_50_cos_diversity_cap_10000',
        'knn_1000_cos_diversity_cap_10000',
        # 'kl_entropy_cap_10000_k_5',
        # 'kl_entropy_cap_10000_k_50',
        'kl_entropy_cap_10000_k_1000',
        'gaussian_aic_cap_10000',
        'toxicity_cap_10000',
        'positivity_cap_10000',
    ])

    np.random.seed(42)
    random.seed(42)

    args = parser.parse_args()

    if os.path.isfile(args.selection_save_path):
        print(f"Selection save path already exists: {args.selection_save_path}")
        exit(1)

    all_results = {}
    metric_values_dict = {}
    title = ""

    cluster_results = load_per_cluster_results(args.cluster_evaluation_files)
    print(f"Loaded {len(cluster_results)} cluster results")

    cluster_results = parse_metrics(cluster_results, args.show_metrics)

    # plot all clusters correlations
    per_metric_values = extract_metrics_distributions(cluster_results)
    metric_values_dict[f"full"] = per_metric_values

    plot_metric_correlations(
        per_metric_values, show_metrics=args.show_metrics,
        save_path=os.path.join(args.visualization_dir, f'all_clusters_correlations.png'),
    )

    # Select clusters
    ################
    selected_cluster_results = dict(random.sample(list(cluster_results.items()), 200))
    selected_per_metric_values = extract_metrics_distributions(selected_cluster_results)
    metric_values_dict[f"selected"] = selected_per_metric_values

    # plot selected clusters correlations
    plot_metric_correlations(
        selected_per_metric_values, show_metrics=args.show_metrics,
        save_path=os.path.join(args.visualization_dir, f'selected_clusters_correlations.png'),
    )

    # show all and selected clusters distributions
    plot_metric_distributions(
        metric_values_dict, title=title, show_metrics=args.show_metrics,
        save_path=os.path.join(args.visualization_dir, f'metrics_distributions.png')
    )

    plot_metric_distributions_histogram(
        metric_values_dict, title=title, show_metrics=args.show_metrics,
        save_path=os.path.join(args.visualization_dir, f'metrics_distributions_histogram.png')
    )

    # Save results
    ####################

    # save distributions: metric -> values
    save_path = args.visualization_dir + f'/clustering_metrics_distributions.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(metric_values_dict, f)
        print(f"Metrics distributions saved to: {save_path}")

    # save selection: index -> cluster_path
    cluster_index_to_path = dict(enumerate(selected_cluster_results))

    with open(args.selection_save_path, 'w') as f:
        json.dump(cluster_index_to_path, f, indent=4)
        print("Cluster index to path dict saved to:", args.selection_save_path)



