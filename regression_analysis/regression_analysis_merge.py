import glob
import json
import pickle
import sys
import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, train_test_split
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import cross_val_score
from sklearn.cross_decomposition import PLSRegression
import numpy as np
from termcolor import cprint

from statsmodels.tools.tools import pinv_extended
import statsmodels.api as sm
import sklearn, statsmodels

from regression_analysis.regression_analysis_utils import load_clusters_data_from_indices_to_path_dict, parse_metrics, regression_analysis_results

from regression_analysis.regression_analysis import plot_coefficient_intervals


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
    # parse args
    # q is quality; d is diversity

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Select the dependent variable.")
    parser.add_argument("dep", choices=["q", "d"], help="Choose 'q' for quality or 'd' for diversity")
    parser.add_argument("--partition", type=str, required=True, choices=["webis_reddit", "100m_tweets", "wikipedia"])
    parser.add_argument("--save-dir", type=str, default=f"regression_analysis/results/all_datasets")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--only-labels", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--cv", type=int, default=10)
    parser.add_argument("--legend", action="store_true")
    parser.add_argument("--gen-ns", "-gs", type=int, nargs='+', default=[500, 1000], help="List of generation numbers (e.g., 500 1000 2000)")
    parser.add_argument("--select-variables", action="store_true")

    # Parse arguments
    args = parser.parse_args()

    dep = {
        # "q": "llama_quality_scale_cap_250",
        # "d": "cos_diversity_stella_cap_250",
        # "q": "llama_quality_scale_cap_80",
        # "d": "cos_diversity_stella_cap_80",
        "q": "llama_quality_scale_cap_160",
        "d": "cos_diversity_stella_cap_160",
    }[args.dep]

    print("Dependent variable:", dep)

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

    data = defaultdict(list)
    # data = {v: [] for v in independent_variables}
    # data["ratio"] = []
    # data[dep] = []

    cluster_simulation_results_dir = "eval_results/simulation_results/merged_clusters"

    datasets_params = {
        "webis_reddit": {
            "clusters_indices_to_path_json": "data/webis/selected_clusters_indices_to_path.json",
            "clusters_evaluation_dir": "data/webis/webis_dataset_clusters/evaluation",
        },
        "100m_tweets": {
            "clusters_indices_to_path_json": "data/twitter_100m/selected_clusters_indices_to_path.json",
            "clusters_evaluation_dir": "data/twitter_100m/100m_tweets_dataset_clusters/evaluation",
        },
        "wikipedia": {
            "clusters_indices_to_path_json": "data/wikipedia/selected_clusters_indices_to_path.json",
            "clusters_evaluation_dir": "data/wikipedia/wikipedia_dataset_clusters/evaluation",
        }
    }

    print("Loading cluster metrics")
    dataset_clusters_data = {}
    # load cluster metrics (predictors)
    for dataset, d_params in datasets_params.items():
        print("Loading clusters data for dataset:", dataset)
        with open(d_params['clusters_indices_to_path_json'], 'r') as f:
            cluster_index_to_path_dict = json.load(f)

        d_clusters_data = load_clusters_data_from_indices_to_path_dict(
            cluster_index_to_path_dict,
            clusters_evaluation_dir=d_params['clusters_evaluation_dir']
        )
        dataset_clusters_data[dataset] = parse_metrics(d_clusters_data)  # averages and remove string metrics

    print("Loading evaluations")
    # load evaluations
    for gen_n in args.gen_ns:
        for cl_index in range(200):
            cl_index = str(cl_index)

            # create row
            results_json = glob.glob(f"eval_results/simulation_results/merged_clusters/clusters_*_cluster_{cl_index}_participants_1_partition_{args.partition}/generated_{gen_n}_*/*/*/results.json")

            if len(results_json) == 0:
                cprint(f"No results found for cluster {cl_index}", "red")
                continue

            assert len(results_json) == 1
            results = load_results(results_json[0])

            # dependant variable
            smoothed_score = np.mean([results[dep][gen] for gen in range(15, 20)])
            normalizer = np.mean(results[dep][0])
            normalized_score = smoothed_score / normalizer
            data[dep].append(normalized_score)

            # controls
            data['ratio'].append(gen_n / 4000)
            # d_data['dataset'].append(dataset)

            # predictors
            for dataset, clusters_data in dataset_clusters_data.items():
                for var in independent_variables:
                    data[f"{dataset}_{var}"].append(clusters_data[cl_index][var])


    # Create a DataFrame
    df = pd.DataFrame(data)
    print("Clusters found:", df.shape[0])

    # dependent variable
    y = df[dep]
    all_independent_variables = [f"{dataset}_{var}" for dataset in datasets_params.keys() for var in independent_variables]
    X = df[all_independent_variables]

    dummies_dataset = pd.get_dummies(data['dataset'], drop_first=True)
    # dummies_ratio = pd.get_dummies(data['ratio'], drop_first=True)
    dummies_ratio = pd.DataFrame(data['ratio'], columns=["ratio"])
    X = pd.concat([X, dummies_dataset, dummies_ratio], axis=1)
    # X = sm.add_constant(X)

    feature_names = list(X.columns)
    X = StandardScaler().fit_transform(X)

    X = pd.DataFrame(X, columns=feature_names)

    # The 'LinearRegression' model is initialized and fitted to the training data.
    model = LinearRegression(fit_intercept=True)
    r2s = cross_val_score(model, X, y, cv=args.cv, scoring='r2')
    print("cv-R2:", np.mean(r2s))

    model.fit(X, y)
    result = regression_analysis_results(X, y, model)
    print(result.summary())

    names = result.model.exog_names
    conf_ints = result.conf_int()
    coefs = result.params
    p_values = result.pvalues
    assert names[0] == "const"  # don't show the intercept
    print("CONST: ", coefs[0], p_values[0])

    # control variables
    do_not_show = ["const", "ratio"]
    names, conf_ints, coefs, p_values = zip(*[
        (n, ci, c, p) for n, ci, c, p in zip(names, conf_ints, coefs, p_values) if n not in do_not_show
    ])

    # names = names[1:]
    # conf_ints = conf_ints[1:]
    # coefs = coefs[1:]
    # p_values = p_values[1:]

    import re
    highlight_mask = []
    highlight_pattern = args.partition
    for idx, name in enumerate(names):
        if re.search(highlight_pattern, name):
            highlight_mask.append(True)
        else:
            highlight_mask.append(False)

    if args.no_labels:
        names = ["" for n in names]
        y_scale = 3.5
        subplots_adjust_args = dict(left=0.04, right=0.96, top=1.0, bottom=0.03)
        xticks_fontsize = 50
        yticks_fontsize = 30

    else:
        y_scale = 2
        subplots_adjust_args = dict(left=0.65, right=0.96, top=1.0, bottom=0.03)
        xticks_fontsize = 20
        yticks_fontsize = 50

    # figsize = (3, y_scale * (len(coefs)-1))
    figsize = (1, y_scale * (len(coefs)-1))

    plot_coefficient_intervals(
        names=names,
        conf_ints=conf_ints,
        coefs=coefs,
        p_values=p_values,
        save_path=os.path.join(args.save_dir, f"ols_{'q' if 'quality' in dep else 'd'}_{args.partition}_{args.gen_ns}{'' if args.no_labels else '_labs'}.pdf"),
        no_show=args.no_show,
        legend=args.legend,
        # highlight_pattern=args.partition,
        highlight_mask=highlight_mask,
        subplots_adjust_args=subplots_adjust_args,
        text_shift=0.01, y_scale=y_scale, xlim=(-0.16, 0.16), figsize=figsize,
        xticks_fontsize=xticks_fontsize,
        yticks_fontsize=yticks_fontsize,
        coef_fontsize=xticks_fontsize
    )

    results_dict = {
        "args": vars(args),
        "coefs": {n: c for n, c in zip(names, coefs)},
        "conf_ints": {n: c.tolist() for n, c in zip(names, conf_ints)},
        "p_values": {n: p for n, p in zip(names, p_values)},
    }

    # save results
    json_save_path = os.path.join(args.save_dir, f"ols_{'q' if 'quality' in dep else 'd'}_{args.gen_ns}.json")
    os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
    with open(json_save_path, "w") as f:
        json.dump(results_dict, f, indent=4)
