import torch
import os
import glob
import argparse
import pickle
import time

import datasets
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import eval_utils
from umap import UMAP

stella_embedder = None


def evaluate_clusters(clusters, dataset_embeddings, dataset_projections):
    per_cluster_metrics = defaultdict(dict)

    for c_i, (cluster_path, cluster_results) in enumerate(clusters.items()):
        print(f"Cluster [{c_i}/{len(clusters)}]")
        cluster_eval_time_start = time.time()
        cluster = datasets.load_from_disk(cluster_path)

        cluster_indices = cluster_results['cluster_indices']
        cluster_embeddings = dataset_embeddings[cluster_indices]
        cluster_projections = dataset_projections[cluster_indices]

        # subsample
        cap_size = 250
        cap_indices = np.random.choice(range(len(cluster)), cap_size)
        cluster_capped = cluster.select(cap_indices)
        cluster_embeddings_capped = cluster_embeddings[cap_indices]
        cluster_projections_capped = cluster_projections[cap_indices]

        mid_cap = 10_000
        mid_cap_indices = np.random.choice(range(len(cluster)), mid_cap)
        cluster_mid_capped = cluster.select(mid_cap_indices)
        cluster_embeddings_mid_capped = cluster_embeddings[mid_cap_indices]
        cluster_projections_mid_capped = cluster_projections[mid_cap_indices]

        big_cap = 80_000
        big_cap_indices = np.random.choice(range(len(cluster)), big_cap)
        cluster_big_capped = cluster.select(big_cap_indices)
        cluster_embeddings_big_capped = cluster_embeddings[big_cap_indices]
        cluster_projections_big_capped = cluster_projections[big_cap_indices]

        # embeddings
        print("Quick metrics")
        s = time.time()
        cluster_results = {k+f"_cap_{cap_size}": v for k, v in eval_utils.compute_quick_metrics(cluster_capped).items()}
        cluster_results.update({k+f"_cap_{mid_cap}": v for k, v in eval_utils.compute_quick_metrics(cluster_mid_capped).items()})
        print("Time:", time.time()-s)

        print("Quality")
        s = time.time()
        # full
        llama_scores = cluster["llama_quality_scale"]
        llama_scores_no_nan = [s for s in llama_scores if s is not None]
        pc_llama_nans = len(llama_scores_no_nan)/len(llama_scores)
        cluster_results["pc_llama_nans"] = pc_llama_nans
        cluster_results["llama_quality_scale"] = np.mean(llama_scores_no_nan)

        # small_cap
        llama_scores = cluster_capped["llama_quality_scale"]
        llama_scores_no_nan = [s for s in llama_scores if s is not None]
        pc_llama_nans = len(llama_scores_no_nan)/len(llama_scores)
        cluster_results[f"pc_llama_nans_cap_{cap_size}"] = pc_llama_nans
        cluster_results[f"llama_quality_scale_cap_{cap_size}"] = np.mean(llama_scores_no_nan)

        # mid cap
        llama_scores = cluster_mid_capped["llama_quality_scale"]
        llama_scores_no_nan = [s for s in llama_scores if s is not None]
        pc_llama_nans = len(llama_scores_no_nan)/len(llama_scores)
        cluster_results[f"pc_llama_nans_cap_{mid_cap}"] = pc_llama_nans
        cluster_results[f"llama_quality_scale_cap_{mid_cap}"] = np.mean(llama_scores_no_nan)

        # big cap
        llama_scores = cluster_big_capped["llama_quality_scale"]
        llama_scores_no_nan = [s for s in llama_scores if s is not None]
        pc_llama_nans = len(llama_scores_no_nan)/len(llama_scores)
        cluster_results[f"pc_llama_nans_cap_{big_cap}"] = pc_llama_nans
        cluster_results[f"llama_quality_scale_cap_{big_cap}"] = np.mean(llama_scores_no_nan)
        print("Time:", time.time()-s)

        print("Word entropy")
        s = time.time()
        cluster_results[f"word_entropy_cap_{cap_size}"] = eval_utils.compute_word_entropy(cluster_capped)
        cluster_results[f"word_entropy_cap_{mid_cap}"] = eval_utils.compute_word_entropy(cluster_mid_capped)
        # cluster_results[f"word_entropy_cap_{big_cap}"] = eval_utils.compute_word_entropy(cluster_big_capped)
        print("Time:", time.time()-s)

        print("SelfBleu")
        s = time.time()
        cluster_results[f"selfbleu_cap_{cap_size}"], cluster_results[f"diversity_selfbleu_cap_{cap_size}"] =\
            eval_utils.compute_selfbleu_parallel(cluster_capped["text"], n_procs=32)
        print("Time:", time.time()-s)

        sb_cap = 500  # 10k is too long
        cluster_results[f"selfbleu_cap_{sb_cap}"], cluster_results[f"diversity_selfbleu_cap_{sb_cap}"] = \
            eval_utils.compute_selfbleu_parallel(
                cluster.select(np.random.choice(range(len(cluster)), sb_cap))["text"],
                n_procs=32
            )
        print("Time:", time.time()-s)

        # div
        print("Diversity")
        s = time.time()
        cluster_results[f"cos_diversity_cap_{cap_size}"] = eval_utils.compute_cos_diversity(cluster_embeddings_capped)
        cluster_results[f"cos_diversity_cap_{mid_cap}"] = eval_utils.compute_cos_diversity(cluster_embeddings_mid_capped)
        # can go up to 60k-> use 60 as big cap?
        # cluster_results[f"cos_diversity_cap_{big_cap}"] = eval_utils.compute_cos_diversity(cluster_embeddings_big_capped)
        print("Time:", time.time()-s)

        print("Diversity knn")
        s = time.time()
        d_matrix = None
        for k in [5, 10, 25, 50]:
            cos_div, d_matrix = eval_utils.compute_knn_cos_diversity(
                cluster_embeddings_capped, k=k, dist_matrix=d_matrix, return_dist_matrix=True)
            cluster_results[f"knn_{k}_cos_diversity_cap_{cap_size}"] = cos_div

        d_matrix = None
        for k in [5, 10, 25, 50, 100, 1000]:
            cos_div, d_matrix = eval_utils.compute_knn_cos_diversity(
                cluster_embeddings_mid_capped, k=k, dist_matrix=d_matrix, return_dist_matrix=True)
            cluster_results[f"knn_{k}_cos_diversity_cap_{mid_cap}"] = cos_div

        # can go up to 60k-> use 60 as big cap?
        # d_matrix = None
        # for k in [5, 10, 25, 50, what here]:
        #     cos_div, d_matrix = eval_utils.compute_knn_cos_diversity(
        #         cluster_embeddings_big_capped, k=k, dist_matrix=d_matrix, return_dist_matrix=True)
        #     cluster_results[f"knn_{k}_cos_diversity_cap_{big_cap}"] = cos_div
        print("Time:", time.time()-s)

        print("Entropy")
        s=time.time()
        for k in [5, 10, 25, 50]:
            cluster_results[f'kl_entropy_cap_{cap_size}_k_{k}'] = eval_utils.compute_kl_entropy(
                cluster_projections_capped, k=k, norm='euclidean')

        for k in [5, 10, 25, 50, 100, 1000, 2000]:
            cluster_results[f'kl_entropy_cap_{mid_cap}_k_{k}'] = eval_utils.compute_kl_entropy(
                cluster_projections_mid_capped, k=k, norm='euclidean')

        # for k in [40, 400, 4000, 10_000]:
        #     cluster_results[f'kl_entropy_cap_{big_cap}_k_{k}'] = eval_utils.compute_kl_entropy(cluster_projections_big_capped, k=k)
        # print("Time:", time.time()-s)

        print("Gaussianess")
        s = time.time()
        cluster_results[f'gaussian_loss_cap_{cap_size}'],\
        cluster_results[f'gaussian_bic_cap_{cap_size}'],\
        cluster_results[f'gaussian_aic_cap_{cap_size}'] = eval_utils.compute_gaussianes(cluster_projections_capped)

        cluster_results[f'gaussian_loss_cap_{mid_cap}'], \
        cluster_results[f'gaussian_bic_cap_{mid_cap}'], \
        cluster_results[f'gaussian_aic_cap_{mid_cap}'] = eval_utils.compute_gaussianes(cluster_projections_mid_capped)

        cluster_results[f'gaussian_loss_cap_{big_cap}'],\
        cluster_results[f'gaussian_bic_cap_{big_cap}'],\
        cluster_results[f'gaussian_aic_cap_{big_cap}'] = eval_utils.compute_gaussianes(cluster_projections_big_capped)
        print("Time:", time.time()-s)

        # toxicity
        print("Toxicity")
        s = time.time()
        cluster_results[f"toxicity_cap_{cap_size}"] = eval_utils.get_toxicity_batch(cluster_capped["text"])
        # check if cuda available
        cluster_results[f"toxicity_cap_{mid_cap}"] = eval_utils.get_toxicity_batch(cluster_mid_capped["text"])
        # cluster_results[f"toxicity_cap_{big_cap}"] = eval_utils.get_toxicity_batch(cluster_big_capped["text"])
        print("Time:", time.time()-s)

        print("Positivity")
        s = time.time()
        cluster_results[f"positivity_cap_{cap_size}"] = eval_utils.get_positivity_batch(cluster_capped["text"])
        cluster_results[f"positivity_cap_{mid_cap}"] = eval_utils.get_positivity_batch(cluster_mid_capped["text"])
        # cluster_results[f"positivity_cap_{big_cap}"] = eval_utils.get_positivity_batch(cluster_big_capped["text"])
        print("Time:", time.time()-s)

        print("Reading level")
        s = time.time()
        cluster_results[f'aggregate_reading_level_cap_{cap_size}'] = eval_utils.aggregate_reading_level(texts=cluster_capped['text'])
        cluster_results[f'aggregate_reading_level_cap_{mid_cap}'] = eval_utils.aggregate_reading_level(texts=cluster_mid_capped['text'])
        # cluster_results[f'aggregate_reading_level_cap_{big_cap}'] = eval_utils.aggregate_reading_level(texts=cluster_big_capped['text'])

        print("Time:", time.time()-s)

        cluster_results["cluster_path"] = cluster_path

        per_cluster_metrics[cluster_path] = cluster_results

        cluster_eval_time_end = time.time()
        print(f"Cluster evaluation time: {cluster_eval_time_end - cluster_eval_time_start:.2f}s")
        del d_matrix

    return per_cluster_metrics


def load_clusters(file):
    all_results = {}
    with open(file, 'rb') as f:
        results = pickle.load(f)
        for c_number, v in results['clusters'].items():
            all_results[v['cluster_path']] = v
    return all_results


def extract_cluster_metrics(cluster_results, metrics_to_plot):
    metric_values = {metric: [] for metric in metrics_to_plot}

    for cluster, cluster_metrics in cluster_results.items():
        for metric in metrics_to_plot:
            if metric in cluster_metrics:
                metric_values[metric].append(cluster_metrics[metric])

    return metric_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze clustering results')
    parser.add_argument('--clustering-pkl', required=True, type=str, default="./data/webis/webis_dataset_clusters/results/umap_(5,0.0001)_gmm_(5).pkl")
    parser.add_argument('--dataset-path', required=True, type=str, default="./data/webis/webis_dataset_with_qualities")
    parser.add_argument('--embeddings-path', required=True, type=str, default="./data/webis/webis_dataset_embeddings.npy")
    parser.add_argument('--projections-path', required=True, type=str, default="./data/webis/webis_dataset_projections.npy")
    parser.add_argument('--clustering-evaluation-save-dir', required=True, type=str, default="./data/webis/webis_dataset_clusters/evaluation/")

    start_time = time.time()

    args = parser.parse_args()
    all_results = {}
    metric_values_dict = {}

    # assert args.clustering_evaluation_save_path.endswith(os.path.basename(args.clustering_pkl))

    print("Load clusters from: ", args.clustering_pkl)
    cluster_results = load_clusters(args.clustering_pkl)

    print("Load dataset from: ", args.dataset_path)
    dataset = datasets.load_from_disk(args.dataset_path)

    print("Load embeddings from: ", args.embeddings_path)
    with open(args.embeddings_path, "rb") as f:
        dataset_embeddings = np.load(f)

    print("Load projections from: ", args.projections_path)
    with open(args.projections_path, "rb") as f:
        dataset_projections = np.load(f)

    assert len(dataset) == len(dataset_embeddings) == len(dataset_projections)

    print(f"Evaluating {len(cluster_results)} clusters.")
    cluster_metrics = evaluate_clusters(
        clusters=cluster_results,
        dataset_embeddings=dataset_embeddings,
        dataset_projections=dataset_projections
    )

    # save per cluster evaluation results
    # create directory if it doesn't exist
    os.makedirs(args.clustering_evaluation_save_dir, exist_ok=True)
    print("Saving results to:", args.clustering_evaluation_save_dir)

    for c, metrics in cluster_metrics.items():
        cluster_eval_save_path = os.path.join(
            args.clustering_evaluation_save_dir, f"{os.path.basename(c)}.pkl"
        )

        with open(cluster_eval_save_path, 'wb') as f:
            pickle.dump(metrics, f)
            print("Cluster results saved to:", cluster_eval_save_path)

    end_time = time.time()
    print("Total time:", end_time - start_time)