import glob
import json
import pickle
import sys
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, train_test_split
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import cross_val_score
from sklearn.cross_decomposition import PLSRegression
import numpy as np
from termcolor import cprint

from regression_analysis.regression_analysis_utils import load_clusters_data_from_indices_to_path_dict, parse_metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import spearmanr, pearsonr

import seaborn as sns

def plot_metric_correlations(X, save_path=None):

    metrics = X.columns

    # Create correlation matrix
    corr_matrix = np.zeros((len(metrics), len(metrics)))

    # Compute correlations between each pair of metrics
    for i, metric1 in enumerate(metrics):
        for j, metric2 in enumerate(metrics):
            corr, _ = spearmanr(X[metric1], X[metric2])
            # corr = np.corrcoef(X[metric1], X[metric2])[0, 1]
            corr_matrix[i, j] = corr

    # compute VIF
    X_scaled = StandardScaler().fit_transform(X)
    vif = np.array([variance_inflation_factor(X_scaled, i) for i in range(len(metrics))])
    vif_dict = dict(zip(metrics, vif))

    # Create figure
    plt.figure(figsize=(30, 30))

    # Create heatmap
    sns.set(font_scale=3)
    sns.heatmap(
        np.abs(corr_matrix),  # absolute
        xticklabels=metrics,
        yticklabels=[m+f"\n(VIF:{vif_dict[m]:.2f})" for m in metrics],
        annot=corr_matrix,  # Show correlation values
        fmt='.2f',  # Format to 2 decimal places
        cmap='RdBu_r',  # Red-Blue diverging colormap
        vmin=0,
        # vmin=-1,
        vmax=1,
        center=0,
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


def vip(model):
    "https://github.com/scikit-learn/scikit-learn/issues/7050"
    t = model.x_scores_
    # w = model.x_weights_  # replace with x_rotations_ if needed
    w = model.x_rotations_
    q = model.y_loadings_
    features_, _ = w.shape
    inner_sum = np.diag(t.T @ t @ q.T @ q)
    SS_total = np.sum(inner_sum)
    vip = np.sqrt(features_ * (w ** 2 @ inner_sum) / SS_total)
    return vip


def load_results(json_path):
    with open(json_path, "r") as f:
        res = json.load(f)
        # parse str generations to ints
        for metr_, vals_ in res.items():
            if isinstance(vals_, dict):
                res[metr_] = {int(gen): v for gen, v in vals_.items()}
    return res

def load_clusters_data():
    with open("./data/webis/selected_clusters_indices_to_path.json", 'r') as f:
        cluster_index_to_path_dict = json.load(f)

    # load clusters metrics
    clusters_data = {}
    for cl_index, cluster_path in cluster_index_to_path_dict.items():
        results_path = "./aggregated_clustering_results/" + cluster_path.replace("./", "").replace("/", "_") + ".pkl"
        with open(results_path, 'rb') as f:
            metrics = pickle.load(f)
        clusters_data[cl_index] = metrics
    return clusters_data


if __name__ == "__main__":
    import pandas as pd, numpy as np
    # parse args
    # q is quality; d is diversity

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Select the dependent variable.")
    parser.add_argument("--save-dir", type=str, default=f"regression_analysis/results/all_datasets")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--cv", type=int, default=10)
    parser.add_argument("--legend", action="store_true")
    # parser.add_argument("--gen-ns", "-gs", type=int, nargs='+', default=[500, 1000, 2000], help="List of generation numbers (e.g., 500 1000 2000)")
    parser.add_argument("--gen-ns", "-gs", type=int, nargs='+', default=[1000], help="List of generation numbers (e.g., 500 1000 2000)")
    parser.add_argument("--select-variables", action="store_true")

    # Parse arguments
    args = parser.parse_args()

    independent_variables = [
        'text_len_cap_10000',
        # 'ttr_cap_10000',
        # 'n_unique_words_total_cap_10000',
        'llama_quality_scale',
        # 'word_entropy_cap_10000', # remove ent?
        'diversity_selfbleu_cap_500',
        'cos_diversity_cap_10000',
        # 'knn_50_cos_diversity_cap_10000',
        # 'knn_1000_cos_diversity_cap_10000',
        # 'kl_entropy_cap_10000_k_50', # remove ent?
        # 'kl_entropy_cap_10000_k_1000',
        'gaussian_aic_cap_10000',
        # 'toxicity_cap_10000',
        'positivity_cap_10000',
    ]
    # independent_variables = [
    #     'text_len_cap_10000',
    #     'ttr_cap_10000',
    #     'n_unique_words_total_cap_10000',
    #     'llama_quality_scale',
    #     'word_entropy_cap_10000',
    #     'diversity_selfbleu_cap_500',
    #     'cos_diversity_cap_10000',
    #     'knn_50_cos_diversity_cap_10000',
    #     'knn_1000_cos_diversity_cap_10000',
    #     'kl_entropy_cap_10000_k_50',
    #     'kl_entropy_cap_10000_k_1000',
    #     'gaussian_aic_cap_10000',
    #     'toxicity_cap_10000',
    #     'positivity_cap_10000',
    # ]

    data = {v: [] for v in independent_variables}
    data["ratio"] = []

    datasets_params = {
        "webis_reddit": {
            "clusters_indices_to_path_json": "data/webis/selected_clusters_indices_to_path.json",
            "clusters_evaluation_dir": "data/webis/webis_dataset_clusters/evaluation",
            "clusters_simulation_results_dir": "eval_results/simulation_results/webis_clusters"
        },
        "100m_tweets": {
            "clusters_indices_to_path_json": "data/webis/selected_clusters_indices_to_path.json",
            "clusters_evaluation_dir": "data/webis/webis_dataset_clusters/evaluation",
            "clusters_simulation_results_dir": "eval_results/simulation_results/webis_clusters"
        },
        "reddit_submissions": {
            "clusters_indices_to_path_json": "data/reddit_submissions/selected_clusters_indices_to_path.json",
            "clusters_evaluation_dir": "data/reddit_submissions/reddit_submissions_dataset_clusters/evaluation",
            "clusters_simulation_results_dir": "eval_results/simulation_results/reddit_submissions_clusters"
        }
    }

    print("Loading evaluations")
    for dataset, d_params in datasets_params.items():
        d_data = {v: [] for v in independent_variables}
        d_data["ratio"] = []

        with open(d_params['clusters_indices_to_path_json'], 'r') as f:
            cluster_index_to_path_dict = json.load(f)

        clusters_data = load_clusters_data_from_indices_to_path_dict(
            cluster_index_to_path_dict,
            clusters_evaluation_dir=d_params['clusters_evaluation_dir']
        )
        clusters_data = parse_metrics(clusters_data)  # averages and remove string metrics

        # load evaluations
        for gen_n in args.gen_ns:
            for cl_index in clusters_data.keys():
                results_json = glob.glob(f"{d_params['clusters_simulation_results_dir']}/*cluster_{cl_index}_part*/generated_{gen_n}_*/*/*/results.json")
                if len(results_json) == 0:
                    cprint(f"No results found for cluster {cl_index}", "red")
                    continue

                assert len(results_json) == 1
                results = load_results(results_json[0])

                for var in independent_variables:
                    d_data[var].append(clusters_data[cl_index][var])

                d_data['ratio'].append(gen_n / 4000)

        for v in independent_variables:
            data[v].extend(d_data[v])

        data["ratio"].extend(d_data["ratio"])

    # Create a DataFrame
    df = pd.DataFrame(data)
    X = df[independent_variables]
    X_scaled = StandardScaler().fit_transform(X)

    vif = np.array([variance_inflation_factor(X_scaled, i) for i in range(len(X.columns))])

    print("VIFs:")
    for i, metric in enumerate(X.columns):
        print(f"{metric}: {vif[i]}")

    plot_metric_correlations(X)