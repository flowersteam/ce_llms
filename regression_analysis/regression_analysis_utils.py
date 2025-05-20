from tqdm import tqdm
import os
import pickle
import numpy as np

def load_clusters_data_from_indices_to_path_dict(cluster_index_to_path_dict, clusters_evaluation_dir="data/webis/webis_dataset_clusters/evaluation"):

    # load clusters metrics
    clusters_data = {}
    for cl_index, cluster_path in cluster_index_to_path_dict.items():
        cluster_eval_dir = os.path.join(clusters_evaluation_dir, os.path.basename(cluster_path)+".pkl")
        with open(cluster_eval_dir, 'rb') as f:
            metrics = pickle.load(f)
        clusters_data[cl_index] = metrics
    return clusters_data

def load_per_cluster_results(cluster_evaluation_files):
    cluster_results = {}

    for cluster_eval_path in tqdm(cluster_evaluation_files, desc="Loading cluster evaluation files"):
        with open(cluster_eval_path, 'rb') as f:
            metrics = pickle.load(f)

        c_path = metrics['cluster_path']
        cluster_results[c_path] = metrics

    return cluster_results


def parse_metrics(cluster_results, metrics=None):
    """
    Removes string metrics, and averages over different texts
    """
    cluster_results_parsed = {}
    # parse metrics
    for cluster, cl_metrics in cluster_results.items():
        # remove cluster path from metrics
        cl_metrics.pop('cluster_path', None)

        # remove text_cap from metrics
        for k in list(cl_metrics.keys()):
            if k.startswith('text_cap_'):
                cl_metrics.pop(k, None)

        # average
        if metrics:
            cl_metrics = {m: np.mean(v) for m, v in cl_metrics.items() if m in metrics}
        else:
            cl_metrics = {m: np.mean(v) for m, v in cl_metrics.items()}

        cluster_results_parsed[cluster] = cl_metrics

    return cluster_results_parsed

from statsmodels.tools.tools import pinv_extended
import statsmodels.api as sm
import sklearn, statsmodels

# check this function: https://stackoverflow.com/questions/40072870/statistical-summary-table-in-sklearn-linear-model-ridge
def regression_analysis_results(X, y, model):
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


