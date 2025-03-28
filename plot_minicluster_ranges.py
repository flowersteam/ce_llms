import os
import sys
import json
import glob
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(files):
    all_results = []
    for filepath in files:
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
            all_results.append(results)
    return all_results


def extract_cluster_metrics(all_results, metrics_to_plot):
    metric_values = {metric: [] for metric in metrics_to_plot}

    for result in all_results:
        for cluster, cluster_metrics in result["clusters"].items():
            for metric in metrics_to_plot:
                if metric in cluster_metrics:
                    metric_values[metric].append(np.mean(cluster_metrics[metric]))

    return metric_values


import os
import matplotlib.pyplot as plt
import numpy as np


def plot_metric_distributions(metric_values_dict, save_path=None, title=""):
    n_metrics = len(list(metric_values_dict.values())[0])
    print(n_metrics)
    f, a = plt.subplots(n_metrics)
    f.set_size_inches(10, 14)
    a = a.ravel()

    # Reverse the items list before plotting
    items = list(metric_values_dict.items())[::-1]

    for i, (label, metric_values) in enumerate(items):
        metrics = list(metric_values.keys())

        for idx, ax in enumerate(a):
            metric = metrics[idx]
            y = (len(items)-1-i)/len(metric_values_dict)*np.ones_like(metric_values[metric])
            ax.scatter(metric_values[metric], y, alpha=0.5, label=label, s=10)
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

            ax.set_ylabel(metric, rotation=0, labelpad=10, ha='right', color='black')

            # Remove left, right, and top borders
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

    plt.suptitle(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print("Plot saved to:", save_path)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze clustering results')
    # parser.add_argument('files', nargs='+')
    parser.add_argument('-o', '--save-path', default='viz_results/webis_many_clusters/')
    parser.add_argument('-n', '--save-name', default='clustering_metrics_distribution')
    parser.add_argument('-m', '--metrics', nargs='+', default=[
        'text_len',
        # 'pc_unique_posts',
        "gaussian_loss", "gaussian_bic", "gaussian_aic",
        'word_entropy', 'kl_entropy',
        'llama_quality_scale',
        'diversity_selfbleu',
        'cos_diversity', 'knn_5_cos_diversity', 'knn_10_cos_diversity', 'knn_50_cos_diversity',
    ])

    args = parser.parse_args()
    all_results = {}
    metric_values_dict = {}
    title = ""
    res_paths = {

        "250": "viz_results/webis_many_clusters_min_size_250/results/*",
        "80k": "viz_results/webis_many_clusters_min_size_80k/results/*",
        "80k_complex_merge": "viz_results/webis_many_clusters_min_size_80k_complex_merge/results/*",

        # "250": "viz_results/webis_many_clusters_min_size_250/results/*0001*",
        # "80k": "viz_results/webis_many_clusters_min_size_80k/results/*0001*",
        # "80k_merge": "viz_results/webis_many_clusters_min_size_80k_merged/results/*0001*",
        # "80k_diverse_merge": "viz_results/webis_many_clusters_min_size_80k_distance_merged/results/*0001*",
        # "80k_complex_merge": "viz_results/webis_many_clusters_min_size_80k_complex_merge/results/*0001*",

        # "250(no_cache)": "viz_results/webis_many_clusters_no_cache_min_size_250/results/*",
        # "250": "viz_results/webis_many_clusters_min_size_250/results/*",
        # "80k(no_cache)": "viz_results/webis_many_clusters_no_cache_min_size_80k/results/*",
        # "80k": "viz_results/webis_many_clusters_min_size_80k/results/*",

        # "250(no_cache)": "viz_results/webis_many_clusters_no_cache_min_size_250/results/*",
        # "27k(no_cache)": "viz_results/webis_many_clusters_no_cache_min_size_65k/results/*",
        # "65k(no_cache)": "viz_results/webis_many_clusters_no_cache_min_size_65k/results/*",
        # "80k(no_cache)": "viz_results/webis_many_clusters_no_cache_min_size_80k/results/*",

        # "250": "viz_results/webis_many_clusters_min_size_250/results/*",
        # "27k(r=75%)": "viz_results/webis_many_clusters_min_size_65k/results/*",
        # "65k(r=50%)": "viz_results/webis_many_clusters_min_size_65k/results/*",
        # "80k": "viz_results/webis_many_clusters_min_size_80k/results/*",

        # "km": "viz_results/webis_many_clusters_no_cache_min_size_80k/results/*kmean*",
        # "gmm": "viz_results/webis_many_clusters_no_cache_min_size_80k/results/*gmm*",
        # "dbscan": "viz_results/webis_many_clusters_no_cache_min_size_80k/results/*_dbscan*",
        # "hdbscan": "viz_results/webis_many_clusters_no_cache_min_size_80k/results/*_hdbscan*",

        # "250_umap_0.0001": "viz_results/webis_many_clusters_no_cache_min_size_250/results/*0001)*",
        # "250_umap_0.001": "viz_results/webis_many_clusters_no_cache_min_size_250/results/*001)*",
        # "250_umap_0.01": "viz_results/webis_many_clusters_no_cache_min_size_250/results/*01)*",
        # "80k_umap_0.0001": "viz_results/webis_many_clusters_no_cache_min_size_80k/results/*.0001)*",
        # "80k_umap_0.001": "viz_results/webis_many_clusters_no_cache_min_size_80k/results/*.001)*",
        # "80k_umap_0.01": "viz_results/webis_many_clusters_no_cache_min_size_80k/results/*.01)*",
        # "80k_umap_0.1": "viz_results/webis_many_clusters_no_cache_min_size_80k/results/*.1)*",

        # "250_no_noise": "viz_results/webis_many_clusters_min_size_250/results/*noise.pkl",
        # "250_w_noise": "viz_results/webis_many_clusters_min_size_250/results/*).pkl",
        # "80k_no_noise": "viz_results/webis_many_clusters_min_size_80k/results/*noise.pkl",
        # "80k_w_noise": "viz_results/webis_many_clusters_min_size_80k/results/*).pkl",
    }
    for label, r_path in res_paths.items():
        if type(r_path) is list:
            files = []
            for path in r_path:
                files += glob.glob(path)
        else:
            files = glob.glob(r_path)
        results = load_results(files)
        n_clusters = sum([len(r['clusters']) for r in results])
        print(f"Label: {label} number of clusters: {n_clusters} num clusterings: {len(results)}"),
        label += f" ({n_clusters})"
        metric_values = extract_cluster_metrics(results, args.metrics)
        metric_values_dict[label] = metric_values

    # save
    save_path = args.save_path + f'/{args.save_name}'
    # plot_metric_distributions(metric_values, save_path+".png", title=save_path)
    plot_metric_distributions(metric_values_dict, title=title)
    with open(save_path+".pkl", 'wb') as f:
        pickle.dump(metric_values_dict, f)
