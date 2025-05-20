import glob
import json
import pickle
import sys
import os
import argparse
from termcolor import cprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

from regression_analysis.regression_analysis_utils import regression_analysis_results
import statsmodels.api as sm

# Set random seed for reproducibility
# np.random.seed(42)


from regression_analysis.regression_analysis_utils import load_clusters_data_from_indices_to_path_dict, parse_metrics


def aic_scorer(estimator, X, y):
    aic = regression_analysis_results(X, y, estimator).aic
    return -aic


def adjusted_r2(estimator, X, y):
    if len(X.shape) == 2:
        n = X.shape[0]
        p = X.shape[1]
    else:
        n = X.shape[0]
        p = 1

    adj_r2 = 1 - (1 - r2_score(y, estimator.predict(X))) * ((n - 1) / (n - p - 1))
    return adj_r2

def plot_feature_interactions(features, coefs, vips, min_vip=1.0, title=None, save_path=None, no_show=False):
    # Get features, coefficients and VIP scores
    features_coef_vip = list(zip(features, coefs, vips))

    # Sort by VIP score and filter
    significant_features = sorted(
        [(f, c, v) for f, c, v in features_coef_vip if v >= min_vip],
        key=lambda x: x[2],
        reverse=True
    )

    if not significant_features:
        print("No features with VIP scores >= min_vip found.")
        return

    # Unzip the sorted features
    features, coefficients, vip_scores = zip(*significant_features)

    # Create figure
    plt.figure(figsize=(20, max(5, int(len(significant_features)*0.2))))

    # Create bars
    y_pos = np.arange(len(features))
    bars = plt.barh(y_pos, np.abs(coefficients), alpha=0.8, height=1.0)
    plt.gca().set_ylim(y_pos[0]-0.5, y_pos[-1]+0.5)

    # Color bars based on coefficient sign
    for i, coef in enumerate(coefficients):
        bars[i].set_color('red' if coef < 0 else 'blue')

    # Customize plot
    fontsize=10
    # y_labels = [f.replace("_cap_250","").replace("_cap_80000","") for f in features]
    y_labels = [f.replace("_cap_250", "(250)").replace("_cap_80000", "(80k)") for f in features]
    plt.yticks(y_pos, [f[:50] for f in y_labels], fontsize=fontsize)
    plt.xlabel('VIP Score')
    if title:
        plt.title(title)

    # Add coefficient values as text
    for i, (coef, vip_score) in enumerate(zip(coefficients, vip_scores)):
        plt.text(0, i, f' coef: {coef:.3f} p-value: {vip_score:.2f}', verticalalignment='center', fontsize=fontsize)

    # Add legend
    plt.plot([], [], color='blue', label='Positive coefficient', linewidth=5)
    plt.plot([], [], color='red', label='Negative coefficient', linewidth=5)
    plt.legend()
    

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if not no_show:
        plt.show()

def load_results(json_path):
    with open(json_path, "r") as f:
        res = json.load(f)
        # parse str generations to ints
        for metr_, vals_ in res.items():
            if isinstance(vals_, dict):
                res[metr_] = {int(gen): v for gen, v in vals_.items()}
    return res



def plot_coefficient_intervals(
    names, coefs, conf_ints, p_values, title=None, save_path=None, no_show=False, legend=False, highlight_pattern=None, highlight_mask=None,
    text_shift=0.02, hl_height=0.049, y_scale=2, xlim=(-0.3, 0.3), xticks_fontsize=13, figsize=None, subplots_adjust_args=None, coef_fontsize=20, yticks_fontsize=30

):
    import re

    # Create figure
    if figsize:
        plt.figure(figsize=(20, y_scale*len(coefs)))
    else:
        plt.figure(figsize=figsize)

    # Plot coefficients and CIs
    y_pos = np.linspace(0, 1, len(coefs)+2)[1:-1]
    cs = ['red' if c < 0 else 'blue' for c in coefs]

    vlines_ymax = y_pos[-1]+0.002*(len(coefs)+1)
    vlines_ymin = y_pos[0]-0.002*(len(coefs)+1)

    # plt.axvline(x=0, ymin=vlines_ymin, ymax=vlines_ymax, color='black', linestyle='--', alpha=0.3)
    # plt.hlines(y=y_pos, xmin=conf_ints[:, 0], xmax=conf_ints[:, 1], color=cs, linewidth=20, alpha=0.2)
    # plt.hlines(y=y_pos, xmin=-1, xmax=1, color="gray", alpha=0.2)

    plt.axvline(x=0, ymin=vlines_ymin, ymax=vlines_ymax, color='black', linestyle='--', alpha=0.8)
    plt.hlines(y=y_pos, xmin=-1, xmax=1, color="black", alpha=0.9)

    # plt.hlines(y=y_pos, xmin=conf_ints[:, 0], xmax=conf_ints[:, 1], color=cs, linewidth=20, alpha=0.9)

    for co, c, y, p, conf_int in zip(coefs, cs, y_pos, p_values, conf_ints):
        if p < 0.05:
            alpha = 0.7
        else:
            alpha = 0.3

        plt.hlines(y=y, xmin=conf_int[0], xmax=conf_int[1], color=c, linewidth=40, alpha=alpha)
        plt.axvline(x=co, color=c, linestyle='-', ymax=y+0.35/(len(coefs)+1), ymin=y-0.35/(len(coefs)+1), linewidth=10, alpha=alpha)

    if highlight_mask is not None:
        for idx, name in enumerate(names):
            if highlight_mask[idx]:
                plt.gca().add_patch(plt.Rectangle((-0.3, y_pos[idx] - hl_height/2), 0.6, hl_height, color='gray', alpha=0.2))
    # Highlight rows based on the regex pattern
    elif highlight_pattern:
        for idx, name in enumerate(names):
            if re.search(highlight_pattern, name):
                plt.gca().add_patch(plt.Rectangle((-0.3, y_pos[idx] - hl_height/2), 0.6, hl_height, color='gray', alpha=0.2))

    # Define y limits
    plt.ylim(0, 1)
    plt.xlim(*xlim)
    # enfoce names to be of the same length

    # remove dataset name from name
    names = [n.replace("wikipedia_", "").replace("100m_tweets_", "").replace("webis_reddit_", "") for n in names]

    names = [{
        # 'kl_entropy_cap_10000_k_5': "KL-Entropy (k=5)",
        'kl_entropy_cap_10000_k_50': "KL-Entropy (k=50)",
        'kl_entropy_cap_10000_k_1000': "KL-Entropy (k=1000)",
        # "cos_diversity_cap_10000": "Cosine diversity",
        "cos_diversity_cap_10000": "Semantic diversity",
        'knn_50_cos_diversity_cap_10000': 'k-nn Cosine diversity (k=50)',
        'knn_1000_cos_diversity_cap_10000': 'k-nn Cosine diversity (k=1000)',
        'gaussian_aic_cap_10000': "Gaussianity",
        'llama_quality_scale': "Quality",
        'diversity_selfbleu_cap_500': 'Lexical diversity',
        'text_len_cap_10000': 'Text length',
        'n_unique_words_total_cap_10000': 'Vocabulary size',
        'toxicity_cap_10000': "Toxicity",
        'positivity_cap_10000': "Positivity",
        'ttr_cap_10000': "TTR",
        'word_entropy_cap_10000': "Word entropy",
    }.get(n, n) for n in names]

    names = [n.split("_cap_")[0] for n in names]

    # names = [n.split("_cap_")[0] for n in names]
    # names = [{
    #     "n_unique_words_total": "vocabulary size"
    # }.get(n, n) for n in names]
    names = [n.rjust(22) for n in names]

    plt.yticks(y_pos, names, fontsize=yticks_fontsize)
    # remove tick marks
    plt.tick_params(axis='y', which='major', left=False, right=False, labelleft=True, pad=50)
    plt.xticks(fontsize=xticks_fontsize)
    plt.tick_params(axis='x', length=40, width=2, direction="inout")
    # plt.xlabel('Coefficient Value')

    # Add coefficient values and p-values as text
    for idx, (coef, p_value) in enumerate(zip(coefs, p_values)):
        stars = ''
        if p_value < 0.001:
            stars = '***'
        elif p_value < 0.01:
            stars = '**'
        elif p_value < 0.05:
            stars = '*'

        text_color = 'black' if p_value < 0.05 else 'grey'
        plt.text(coef, y_pos[idx]+text_shift, f' {coef:.3f} {stars}',
                 verticalalignment='bottom',
                 color=text_color, fontsize=coef_fontsize)

    # Add legend for significance
    if legend:
        plt.text(
            plt.xlim()[1], plt.ylim()[0],
            '* p<0.05, ** p<0.01, *** p<0.001',
            verticalalignment='bottom',
            horizontalalignment='right',
            fontsize=15
        )

    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)

    if title:
        # make half the title bold
        plt.title(title, fontsize=20)

    if subplots_adjust_args is None:
        plt.tight_layout()
    else:
        plt.subplots_adjust(**subplots_adjust_args)

    if save_path:

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        # plt.savefig(save_path.replace(".pdf", ".svg"))
        print(f"Plot saved to {save_path}")


    if not no_show:
        plt.show()


if __name__ == "__main__":
    # parse args
    # q is quality; d is diversity

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Select the dependent variable.")
    parser.add_argument("dep", choices=["q", "d"], help="Choose 'q' for quality or 'd' for diversity")
    parser.add_argument("--save-dir", type=str, default=f"regression_analysis/results/webis_clusters")
    parser.add_argument("--clusters-indices-to-path-json", type=str, default=f"data/webis/selected_clusters_indices_to_path.json")
    parser.add_argument("--clusters-evaluation-dir", type=str, default=f"data/webis/webis_dataset_clusters/evaluation")
    parser.add_argument("--clusters-simulation-results-dir", type=str, default=f"eval_results/simulation_results/webis_clusters")
    parser.add_argument("--gen-n", "-g", type=str, default="1000", choices=["500", "1000"])
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--legend", action="store_true")
    parser.add_argument("--cv", type=int, default=None)

    # Parse arguments
    args = parser.parse_args()

    dep = {
        "q": "llama_quality_scale_cap_250",
        "d": "cos_diversity_stella_cap_250",
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

    dependant_variables = [dep]

    data = {v: [] for v in independent_variables}
    data[dep] = []

    with open(args.clusters_indices_to_path_json, 'r') as f:
        cluster_index_to_path_dict = json.load(f)

    clusters_data = load_clusters_data_from_indices_to_path_dict(
        cluster_index_to_path_dict, clusters_evaluation_dir=args.clusters_evaluation_dir
    )
    clusters_data = parse_metrics(clusters_data)  # averages and remove string metrics

    print("Loading evaluations")
    # load evaluations
    for cl_index in clusters_data.keys():
        results_json = glob.glob(f"{args.clusters_simulation_results_dir}/*cluster_{cl_index}_part*/generated_{args.gen_n}_*/*/*/results.json")
        if len(results_json) == 0:
            # raise ValueError(f"No results found for cluster {cl_index} in {args.clusters_simulation_results_dir}/*cluster_{cl_index}_part*/generated_{args.gen_n}_*/*/*/results.json")
            cprint(f"No results found for cluster {cl_index}", "red")
            continue

        assert len(results_json) == 1
        results = load_results(results_json[0])

        for var in independent_variables:
            data[var].append(clusters_data[cl_index][var])

        # dependant variable
        # for var in dependant_variables:
        smoothed_score = np.mean([results[dep][gen] for gen in range(15, 20)])
        normalizer = np.mean(results[dep][0])
        normalized_score = smoothed_score / normalizer
        data[dep].append(normalized_score)

    # Create a DataFrame
    df = pd.DataFrame(data)
    print("Clusters found:", df.shape[0])

    # dependent variable
    y = df[dep]
    X = df[independent_variables]

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
    plot_coefficient_intervals(
        names=names[1:],
        conf_ints=conf_ints[1:],
        coefs=coefs[1:],
        p_values=p_values[1:],
        save_path=os.path.join(args.save_dir, f"ols_{'q' if 'quality' in dep else 'd'}_{args.gen_n}.pdf"),
        no_show=args.no_show,
        legend=args.legend
    )

    results_dict = {
        "args": vars(args),
        "coefs": {n: c for n, c in zip(names, coefs)},
        "p_values": {n: p for n, p in zip(names, p_values)},
        "conf_ints": {n: c.tolist() for n, c in zip(names, conf_ints)},
    }

    # save results
    json_save_path = os.path.join(args.save_dir, f"ols_{'q' if 'quality' in dep else 'd'}_{args.gen_n}.json")
    os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
    with open(json_save_path, "w") as f:
        json.dump(results_dict, f, indent=4)