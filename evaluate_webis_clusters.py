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

def get_embeddings(cluster):
    global stella_embedder
    if stella_embedder is None:
        stella_embedder = eval_utils.StellaEmbedder(multigpu=True)
    return np.array(stella_embedder.add_embeddings(cluster, batch_size=256)['stella_embeddings'])


# def get_umap(embeddings, n_components=2, n_neighbors=5, min_dist=0.01, sample_size=90_000):
#     # dataset_path = "./data/webis/prepared-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17"
#     # print("Load embeddings")
#     # embeddings_path = "./data/webis/prepared-quality-embeddings-cleaned-200-minus-20-plus-corpus-webis-tldr-17_embeddings.npy"
#     # with open(embeddings_path, "rb") as f:
#     #     embeddings = np.load(f)
#
#     # embedding size
#     sample_embeddings = embeddings[:sample_size]
#     umap = UMAP(n_components=n_components, metric="cosine", n_neighbors=n_neighbors, min_dist=min_dist).fit(sample_embeddings)
#     sample_projections = umap.transform(sample_embeddings)
#     return umap, sample_embeddings, sample_projections

def get_webis_emb_proj_lookup():
    dataset_path = "./data/webis/prepared-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17"

    split_dataset = datasets.load_from_disk(dataset_path)
    dataset = datasets.concatenate_datasets([split_dataset['train'], split_dataset['test']])

    # embeddings
    print("Embedding")
    embeddings_path = "./data/webis/prepared-quality-embeddings-cleaned-200-minus-20-plus-corpus-webis-tldr-17_embeddings.npy"
    # from create_webis_miniclusters import compute_embeddings
    # embeddings = compute_embeddings(dataset,save_embeddings_path=embeddings_path)
    # load embeddings
    with open(embeddings_path, "rb") as f:
        embeddings = np.load(f)

    embeddings_lookup = dict(zip(dataset['text'], embeddings))
    print("Creating lookup")

    # projections
    print("Projecting")
    sample_size = 90_000
    umap = UMAP(n_components=2, metric="cosine", n_neighbors=5, min_dist=0.01).fit(embeddings[:sample_size])
    projections = umap.transform(embeddings)
    print("Creating lookup")
    projections_lookup = dict(zip(dataset['text'], projections))

    return embeddings_lookup, projections_lookup



def evaluate_clusters(clusters):
    per_cluster_metrics = defaultdict(dict)

    for c_i, (cluster_path, cluster_results) in enumerate(clusters.items()):
        print(f"Cluster [{c_i}/{len(clusters)}]")
        cluster_eval_time_start = time.time()
        cluster = datasets.load_from_disk(cluster_path)

        embeddings_lookup, projections_lookup = eval_utils.get_or_compute_cache(
            cache_path=str(f".cache/webis_lookups.pickle"),
            compute_fn=get_webis_emb_proj_lookup
        )

        cluster_embeddings = np.stack(list(embeddings_lookup[t] for t in cluster['text']))
        cluster_projections = np.array([projections_lookup[t] for t in cluster['text']])

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
        # cluster_results.update({k+f"_cap_{big_cap}": v for k, v in eval_utils.compute_quick_metrics(cluster_big_capped).items()})
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
        cluster_results[f"selfbleu_cap_{sb_cap}"], cluster_results[f"diversity_selfbleu_cap_{mid_cap}"] = \
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
        s=time.time()
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
        s=time.time()
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

        if torch.cuda.is_available():
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
            s=time.time()
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


def select_representative_clusters(cluster_results, metric_values, edge_pc=1):
    # calculate metric safe intervals (lower and upper 10%)
    metric_intervals = {}
    for metric, values in metric_values.items():
        values = sorted(values)
        # find percentiles
        lower = np.percentile(values, edge_pc)
        upper = np.percentile(values, 100-edge_pc)

        metric_intervals[metric] = (lower, upper)

    must_keep_clusters = set()

    for cluster, c_metrics in cluster_results.items():
        for m, vs in c_metrics.items():
            if vs < metric_intervals[m][0] or vs > metric_intervals[m][1]:
                must_keep_clusters.add(cluster)

    cluster_results = {k: v for k, v in cluster_results.items() if k in must_keep_clusters}
    return cluster_results



# python evaluate_webis_clusters.py viz_results/webis_miniclusters_merge_80k/results/umap_\(5\,0.01\)_kmeans_\(1000\)_no_noise.pkl
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze clustering results')
    parser.add_argument('clustering_pkl', type=str)

    args = parser.parse_args()
    all_results = {}
    metric_values_dict = {}

    cluster_results = load_clusters(args.clustering_pkl)
    cluster_metrics = evaluate_clusters(cluster_results)

    # save per cluster
    for c, metrics in cluster_metrics.items():
        save_path = "./clustering_results/"+c.replace("./", "").replace("/", "_")+".pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(metrics, f)
            print("Cluster results saved to:", save_path)

    for c, metrics in cluster_metrics.items():
        save_path = "./aggregated_clustering_results/" + c.replace("./", "").replace("/", "_") + ".pkl"
        with open(save_path, 'wb') as f:
            aggregated_metrics = {}
            for metric_name, metric_value in list(metrics.items()):
                if metric_name.startswith("text_cap_"):
                    continue
                try:
                    aggregated_metrics[metric_name] = np.mean(metric_value)
                except:
                    aggregated_metrics[metric_name] = metric_value

            pickle.dump(aggregated_metrics, f)
            print("Cluster results saved to:", save_path)
