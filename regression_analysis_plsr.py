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


class PLSRegressionCVWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, n_components_list=range(1, 11), cv=5, scoring='r2'):
        self.n_components_list = n_components_list
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y):
        best_score = -np.inf
        best_n_components = None
        best_model = None

        # Try each candidate number of components
        for n in self.n_components_list:
            if n > X.shape[1]:
                continue
            model = PLSRegression(n_components=n)
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            score = np.mean(scores)
            if score > best_score:
                best_score = score
                best_n_components = n
                best_model = clone(model).fit(X, y)

        self.n_components_ = best_n_components
        self.best_score_ = best_score
        self.model_ = best_model
        # print(f"BEST SCORE {X.shape} ({best_n_components}):", best_score)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    @property
    def coef_(self):
        return self.model_.coef_


# Set random seed for reproducibility
# np.random.seed(42)

from statsmodels.tools.tools import pinv_extended
import statsmodels.api as sm
import sklearn, statsmodels

def plot_coefficient_intervals(
    names, coefs, vips,
    title=None, save_path=None, no_show=False, legend=False
):

    # Create figure
    plt.figure(figsize=(20, len(coefs)))


    # Plot coefficients and CIs
    y_pos = np.linspace(0, 1, len(coefs)+2)[1:-1]
    cs = ['red' if c < 0 else 'blue' for c in coefs]
    vlines_ymax = y_pos[-1]+0.002*(len(coefs)+1)
    vlines_ymin = y_pos[0]-0.002*(len(coefs)+1)

    x_lim_min = -0.06
    coef_vip_separator_x = 0.05
    coef_area = coef_vip_separator_x-x_lim_min
    vip_scaler = coef_area/2
    x_lim_max = 2*vip_scaler  # on the plot half if for coef half for vip

    # coef 0 line
    plt.axvline(x=0, ymax=vlines_ymax, ymin=vlines_ymin, color='black', linestyle='--', alpha=0.3)
    plt.text(0, y_pos[-1]+0.003*(len(coefs)+1), f'Coefficient',
             horizontalalignment="center", color="black", fontsize=20, weight="bold")

    plt.hlines(y=y_pos, xmin=-1, xmax=1, color="gray", alpha=0.2) # gray line
    for y_, v_ in zip(y_pos, vips):
        if v_ >= 1:
            plt.hlines(y=y_, xmin=-1, xmax=1, color="gray", alpha=0.1, linewidth=40) # gray line

    # coefs
    plt.hlines(y=y_pos, xmin=np.minimum(0, coefs), xmax=np.maximum(0, coefs), color=cs, linewidth=10, alpha=0.5)

    # separator
    plt.axvline(x=coef_vip_separator_x, ymax=vlines_ymax, ymin=vlines_ymin, color="black", linestyle='-')

    # vip=1 line
    plt.axvline(x=coef_vip_separator_x+1*vip_scaler, ymax=vlines_ymax, ymin=vlines_ymin, color="purple", linestyle='--', alpha=0.3)
    plt.text(coef_vip_separator_x+1*vip_scaler, y_pos[-1]+0.003*(len(coefs)+1), f'VIP',
             horizontalalignment="center", color="black", fontsize=20, weight="bold")

    # vips
    plt.hlines(y=y_pos, xmin=coef_vip_separator_x, xmax=coef_vip_separator_x+vips*vip_scaler, color="purple", linewidth=10, alpha=0.5)


    # define y limits
    plt.ylim(0, 1)
    plt.xlim(x_lim_min, x_lim_max)
    # enforce names to be of the same length
    # from IPython import embed; embed();
    # names = [n.capitalize() for n in names]
    names = [{
        'kl_entropy_cap_10000_k_5': "KL-Entropy (k=5)",
        'kl_entropy_cap_10000_k_1000': "KL-Entropy (k=1000)",
        "cos_diversity_cap_10000": "Cosine diversity",
        'knn_50_cos_diversity_cap_10000': 'k-nn Cosine diversity (k=50)',
        'knn_1000_cos_diversity_cap_10000': 'k-nn Cosine diversity (k=1000)',
        'gaussian_aic_cap_10000': "Gaussianity",
        'llama_quality_scale': "Quality",
        'diversity_selfbleu_cap_10000': 'Self-BLEU',
        'text_len_cap_10000': 'Text length',
        'n_unique_words_total_cap_10000': 'Vocabulary size',
        'toxicity_cap_10000': "Toxicity",
        'positivity_cap_10000': "Positivity",
        'ttr_cap_10000': "TTR",
        'word_entropy_cap_10000': "Word entropy",
    }.get(n, n) for n in names]
    names = [n.split("_cap_")[0] for n in names]
    # names = [{
    #              "n_unique_words_total": "vocabulary size"
    #          }.get(n, n) for n in names]
    names = [n.rjust(22) for n in names]
    # print(names)

    plt.yticks(y_pos, names, fontsize=30)
    # add x ticks manually 0-1 by 0.1
    plt.xticks(
        np.concatenate((
            np.arange(x_lim_min, coef_vip_separator_x, 0.02),
            np.arange(coef_vip_separator_x, coef_vip_separator_x+2*vip_scaler, 0.2*vip_scaler)[1:]
        )),
        [l.round(2) for l in
            np.concatenate((
                np.arange(x_lim_min, coef_vip_separator_x, 0.02),
                np.arange(0, 2, 0.2)[1:]
            ))
        ],  # vips
        fontsize=25
    )
    plt.tick_params(axis='x', length=30, width=2, pad=30, direction="inout")

    plt.tick_params(axis='y', which='major', left=False, right=False, labelleft=True)

    # plt.xlabel('Coefficient Value')

    if title:
        # make half the title bold
        plt.title(title, fontsize=40)

    # Add coefficient values and p-values as text
    # for idx, (coef, vip) in enumerate(zip(coefs, vips)):
    #     stars = ''
        # if vip < 0.001:
        #     stars = '***'
        # elif p_value < 0.01:
        #     stars = '**'
        # elif p_value < 0.05:
        #     stars = '*'

        # text_color = 'black' if vip < 0.05 else 'grey'
        # plt.text(coef, y_pos[idx], f' {coef:.3f} {stars}',
        #          verticalalignment='bottom',
        #          color=text_color, fontsize=15)

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
    plt.gca().spines['bottom'].set_visible(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if not no_show:
        plt.show()


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
    plt.figure(figsize=(18, max(5, int(len(significant_features)*0.2))))

    # Create bars
    y_pos = np.arange(len(features))
    bars = plt.barh(y_pos, vip_scores, alpha=0.8, height=1.0)
    plt.gca().set_ylim(y_pos[0]-0.5, y_pos[-1]+0.5)

    # Color bars based on coefficient sign
    for i, coef in enumerate(coefficients):
        bars[i].set_color('red' if coef < 0 else 'blue')

    # Customize plot
    fontsize=10
    # y_labels = [f.replace("_cap_250","").replace("_cap_80000","") for f in features]
    y_labels = [f.replace("_cap_250","(250)").replace("_cap_80000","(80k)") for f in features]
    plt.yticks(y_pos, [f[:50] for f in y_labels], fontsize=fontsize)
    plt.xlabel('VIP Score')
    if title:
        plt.title(title)

    # Add coefficient values as text
    for i, (coef, vip_score) in enumerate(zip(coefficients, vip_scores)):
        plt.text(0, i, f' coef: {coef:.3f} vip: {vip_score:.2f}', verticalalignment='center', fontsize=fontsize)

    # Add legend
    plt.plot([], [], color='blue', label='Positive coefficient', linewidth=5)
    plt.plot([], [], color='red', label='Negative coefficient', linewidth=5)
    plt.legend()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if not no_show:
        plt.show()


def PLSRegressionCV(X, y, n_componentss, k=5, scoring='r2'):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Store cross-validation results
    cv_scores = []

    # Try different numbers of components
    for n in n_componentss:
        pls = PLSRegression(n_components=n)
        scores = cross_val_score(pls, X, y, cv=kf, scoring=scoring)
        cv_scores.append(np.mean(scores))

    # Select the best number of components
    max_i = np.argmax(cv_scores)
    optimal_n_components = n_componentss[max_i]
    best_score = cv_scores[max_i]

    model = PLSRegression(n_components=optimal_n_components).fit(X, y)
    return model, best_score


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
    # parse args
    # q is quality; d is diversity

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Select the dependent variable.")
    parser.add_argument("dep", choices=["q", "d"], help="Choose 'q' for quality or 'd' for diversity")
    parser.add_argument("--gen-n", "-g", type=str, default="1000", choices=["500", "1000"])
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--cv", type=int, default=None)
    parser.add_argument("--no-feature-selection", "-nfs", action="store_true")
    parser.add_argument("--legend", action="store_true")

    # Parse arguments
    args = parser.parse_args()

    dep = {
        "q": "llama_quality_scale_cap_250",
        "d": "cos_diversity_stella_cap_250",
    }[args.dep]

    print("Dependent variable:", dep)

    independent_variables = [
        'text_len_cap_10000',
        'ttr_cap_10000',
        'n_unique_words_total_cap_10000',
        'llama_quality_scale',
        'word_entropy_cap_10000',
        'diversity_selfbleu_cap_10000',  # actually 1000
        'cos_diversity_cap_10000',
        'knn_50_cos_diversity_cap_10000',
        'knn_1000_cos_diversity_cap_10000',
        'kl_entropy_cap_10000_k_5',
        'kl_entropy_cap_10000_k_1000',
        'gaussian_aic_cap_10000',
        'toxicity_cap_10000',
        'positivity_cap_10000',
    ]

    # dummy_variables = ["ai_ratio"]
    # dependant_variables = ["cos_diversity_stella_cap_250", "llama_quality_score_cap_250"]
    # dependant_variables = ["cos_diversity_stella_cap_250"]
    # dependant_variables = [dep]

    data = {v: [] for v in independent_variables}
    data[dep] = []

    clusters_data = load_clusters_data()
    # cprint("Removing outliers", "red")
    # from select_representative_clusters import extract_cluster_metrics, remove_outlier_clusters_iqr
    # clusters_data = remove_outlier_clusters_iqr(
    #     clusters_data, extract_cluster_metrics(clusters_data), selection_metrics=independent_variables
    # )

    # load evaluations
    for cl_index in clusters_data.keys():
        results_json = glob.glob(f"eval_results/webis_clusters_results_v2/*cluster_{cl_index}_part*/generated_{args.gen_n}_*/*/*/results.json")
        if len(results_json) == 0:
            cprint(f"No results found for cluster {cl_index}", "red")
            continue

        assert len(results_json) == 1
        results = load_results(results_json[0])

        if "toxicity_cap_10000" not in clusters_data[cl_index]:
            cprint(f"No tox found for cluster {cl_index}", "red")
            continue

        if dep not in results:
            cprint(f"No {dep} found for cluster {cl_index}", "red")
            continue


        # if dep == "cos_diversity_stella_cap_250" or True:
        #     if results["cos_diversity_stella_cap_250"][0] < 0.55:
        #         from termcolor import cprint
        #         cprint(f"Skipping cluster {cl_index} due to low diversity (rock bottom)", "red")
        #         continue

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
    print("Clusters left:", df.shape[0])
    # dependent variable
    y = df[dep]

    X = df[independent_variables]
    feature_names = list(X.columns)
    print("Num features: ", len(independent_variables))

    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X, columns=feature_names)

    # model, r2 = PLSRegressionCV(X, y, n_componentss=range(1, len(feature_names)), k=10, scoring='r2')
    # print("R2 (all feat):", r2)

    model = RFECV(
        estimator=PLSRegressionCVWrapper(
            n_components_list=range(1, len(feature_names)),
            cv=args.cv if args.cv is not None else (len(X) - 2) // 2,
            scoring='r2'
        ),
        cv=args.cv if args.cv is not None else len(X) // 2,
        step=1,
        scoring="r2",
        min_features_to_select=len(feature_names) if args.no_feature_selection else 1,
        n_jobs=8,
    )
    model.fit(X, y)

    y_pred = model.predict(X)
    print("N features:", np.sum(model.support_))

    X = X.iloc[:, model.support_]  # mask features
    model = model.estimator_.model_
    r2 = model.score(X, y)  # this is what they use, but this is on the train set?

    # plot PLSRegression
    cv_str = f'cv_{args.cv}_' if args.cv is not None else 'cv_max_'
    fs_str = f'_all_preds' if args.no_feature_selection else ''
    plot_feature_interactions(
        X.columns, model.coef_[0], vip(model), min_vip=0.0,
        title=f"{dep} (rel) - ratio {int(args.gen_n)/4000} (R^2 = {r2:.3f})",
        save_path=f"viz_results/feature_interactions_v2/PLSR/{cv_str[:-1]}/interactions_{dep}_{cv_str}gen_n_{args.gen_n}{fs_str}.png",
        no_show=args.no_show
    )

    plot_coefficient_intervals(
        names=X.columns,
        coefs=model.coef_[0],
        vips=vip(model),
        # title=f"Relative {'Quality' if 'q' in dep else 'Diversity'} - Synthetic data ratio {int(args.gen_n) / 4000}", # (R^2 = {r2:.3f})",
        save_path=f"viz_results/feature_interactions_v2/PLSR/{cv_str[:-1]}/plsr_{'q' if 'quality' in dep else 'd'}_{args.gen_n}.pdf",
        no_show=args.no_show,
        legend=args.legend
    )
