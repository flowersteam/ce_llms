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
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split


# Set random seed for reproducibility
# np.random.seed(42)

from statsmodels.tools.tools import pinv_extended
import statsmodels.api as sm
import sklearn, statsmodels






def aic_scorer(estimator, X, y):
    aic = regression_analysis(X, y, estimator).aic
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

# check this function: https://stackoverflow.com/questions/40072870/statistical-summary-table-in-sklearn-linear-model-ridge
def regression_analysis(X, y, model):
    is_statsmodels = False
    is_sklearn = False

    # check for accepted linear models
    if type(model) in [sklearn.linear_model._base.LinearRegression,
                       sklearn.linear_model._ridge.Ridge,
                       sklearn.linear_model._ridge.RidgeCV,
                       sklearn.linear_model._coordinate_descent.Lasso,
                       sklearn.linear_model._coordinate_descent.LassoCV,
                       sklearn.linear_model._coordinate_descent.ElasticNet,
                       sklearn.linear_model._coordinate_descent.ElasticNetCV,
                       ]:
        is_sklearn = True
    # elif type(model) in [statsmodels.regression.linear_model.OLS,
    #                      statsmodels.base.elastic_net.RegularizedResults,
    #                      ]:
    #     is_statsmodels = True
    else:
        print("Only linear models are supported!")
        return None

    has_intercept = False

    if is_statsmodels and all(np.array(X)[:, 0] == 1):
        # statsmodels add_constant has been used already
        has_intercept = True
    elif is_sklearn and model.intercept_:
        has_intercept = True

    if is_statsmodels:
        # add_constant has been used already
        x = X
        model_params = model.params
    else:  # sklearn model
        if has_intercept:
            x = sm.add_constant(X)
            model_params = np.hstack([np.array([model.intercept_]), model.coef_])
        else:
            x = X
            model_params = model.coef_

    # y = np.array(y).ravel()

    # define the OLS model
    olsModel = sm.OLS(y, x)

    pinv_wexog, _ = pinv_extended(x)
    normalized_cov_params = np.dot(pinv_wexog, np.transpose(pinv_wexog))

    return sm.regression.linear_model.OLSResults(olsModel, model_params, normalized_cov_params)


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


def plot_coefficient_intervals(
    names, coefs, conf_ints, p_values,
    title=None, save_path=None, no_show=False, legend=False
):

    # Create figure
    plt.figure(figsize=(10, len(coefs)))

    # Plot coefficients and CIs
    y_pos = np.linspace(0, 1, len(coefs)+2)[1:-1]
    cs = ['red' if c < 0 else 'blue' for c in coefs]
    vlines_ymax = y_pos[-1]+0.002*(len(coefs)+1)
    vlines_ymin = y_pos[0]-0.002*(len(coefs)+1)
    plt.axvline(x=0, ymin=vlines_ymin, ymax=vlines_ymax, color='black', linestyle='--', alpha=0.3)
    plt.hlines(y=y_pos, xmin=conf_ints[:, 0], xmax=conf_ints[:, 1], color=cs, linewidth=10, alpha=0.2)
    plt.hlines(y=y_pos, xmin=-1, xmax=1, color="gray", alpha=0.2)
    for co, c, y in zip(coefs, cs, y_pos):
        plt.axvline(x=co, color=c, linestyle='-', ymax=y+0.1/(len(coefs)+1), ymin=y-0.1/(len(coefs)+1), linewidth=1)

    # define y limits
    plt.ylim(0, 1)
    plt.xlim(-0.2, 0.2)
    # enfoce names to be of the same length

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

    # names = [n.split("_cap_")[0] for n in names]
    # names = [{
    #     "n_unique_words_total": "vocabulary size"
    # }.get(n, n) for n in names]
    names = [n.rjust(22) for n in names]

    plt.yticks(y_pos, names, fontsize=20)
    # remove tick marks
    plt.tick_params(axis='y', which='major', left=False, right=False, labelleft=True)
    plt.xticks(fontsize=13)
    plt.tick_params(axis='x', length=5, width=1, direction="inout")
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
        plt.text(coef, y_pos[idx], f' {coef:.3f} {stars}',
                 verticalalignment='bottom',
                 color=text_color, fontsize=15)

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

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    if not no_show:
        plt.show()


if __name__ == "__main__":
    # parse args
    # q is quality; d is diversity

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Select the dependent variable.")
    parser.add_argument("dep", choices=["q", "d"], help="Choose 'q' for quality or 'd' for diversity")
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


    dependant_variables = [dep]

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

        if dep not in results:
            cprint(f"No {dep} found for cluster {cl_index}", "red")
            continue

        if "toxicity_cap_10000" not in clusters_data[cl_index]:
            cprint(f"No tox found for cluster {cl_index}", "red")
            continue


        # if dep == "cos_diversity_stella_cap_250" or True:
        #     if results["cos_diversity_stella_cap_250"][0] < 0.55:
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

    X = StandardScaler().fit_transform(X)
    X = pd.DataFrame(X, columns=feature_names)

    model = RFECV(
        estimator=RidgeCV(
            alphas=[0.001, 0.01, 0.1, 0.5, 1.0],
            scoring="neg_mean_squared_error",
            cv=args.cv if args.cv is not None else (len(X)-2)//2,
        ),
        cv=args.cv if args.cv is not None else len(X) // 2,
        step=1,
        scoring=aic_scorer,
        min_features_to_select=1,
        n_jobs=2,
    )
    model.fit(X, y)
    print(f"Optimal number of features: {model.n_features_}")
    print(f"Optimal alpha: {model.estimator_.alpha_}")

    y_pred = model.predict(X)
    X = X.iloc[:, model.support_]  # mask features
    model = model.estimator_



    result = regression_analysis(X, y, model)
    print(result.summary())
    print(f"AIC: {result.aic}")
    print(f"R^2: {result.rsquared:.3f}")
    r2 = result.rsquared
    print("R2", r2)
    
    cv_str = f'cv_{args.cv}_' if args.cv is not None else 'cv_max_'
    plot_feature_interactions(
        X.columns, model.coef_, result.pvalues, min_vip=-1,  # all p
        title=f"Relative {dep} - ratio {int(args.gen_n)/4000} (R^2 = {r2:.3f})",
        save_path=f"viz_results/feature_interactions_v2/Ridge/{cv_str[:-1]}/interactions_{dep}_{cv_str}gen_n_{args.gen_n}.png",
        no_show=args.no_show
    )

    names = result.model.exog_names
    conf_ints = result.conf_int()
    coefs = result.params
    p_values = result.pvalues

    assert names[0] == "const"  # don't show the intercept
    plot_coefficient_intervals(
        names=names[1:],
        conf_ints=conf_ints[1:],
        coefs=coefs[1:],
        p_values=p_values[1:],
        # title=f"Relative {'Quality' if 'q' in dep else 'Diversity'} (Synthetic data ratio 1/{'8' if args.gen_n == 500 else '4'})", # (R^2 = {r2:.3f})",
        save_path=f"viz_results/feature_interactions_v2/Ridge/{cv_str[:-1]}/ridge_{'q' if 'quality' in dep else 'd'}_{args.gen_n}.pdf",
        no_show=args.no_show,
        legend=args.legend
    )



    # # Plot actual vs. predicted performance for visual evaluation
    # plt.figure(figsize=(8, 6))
    # plt.scatter(y, y_pred, color='blue', edgecolor='k', alpha=0.7)
    # plt.xlabel("Actual Performance")
    # plt.ylabel("Predicted Performance")
    # plt.title("Actual vs. Predicted Performance (Ridge Regression)")
    # plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line for perfect predictions
    # plt.show()
