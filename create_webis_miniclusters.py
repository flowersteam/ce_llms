import os
from collections import defaultdict, Counter
import argparse
import numpy as np
import time
import pickle
import pprint
import json
import random

from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import DBSCAN, HDBSCAN, MiniBatchKMeans
from sklearn.mixture import GaussianMixture
import hdbscan

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from termcolor import cprint
import datasets

from sklearn.metrics.pairwise import euclidean_distances

import eval_utils

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

import faiss

def compute_cluster_metrics(cluster_dataset, cluster_embeddings, cluster_projections, cap_size=250):
    # subsample
    cap_indices = np.random.choice(range(len(cluster_dataset)), cap_size)
    cluster_dataset_capped = dataset.select(cap_indices)
    cluster_embeddings_capped = cluster_embeddings[cap_indices]
    cluster_projections_capped = cluster_projections[cap_indices]

    cprint(f"Cluster {cluster_label} results", "yellow")

    # quick
    s = time.time()
    cluster_results = eval_utils.compute_quick_metrics(cluster_dataset_capped)
    print("Quick metrics computed in: ", time.time() - s)

    # quality
    s = time.time()
    cluster_results["llama_quality_scale"] = cluster_dataset_capped['llama_quality_scale']
    print("Quality metrics computed in: ", time.time() - s)

    # div
    s = time.time()
    cos_div, d_matrix = eval_utils.compute_cos_diversity(cluster_embeddings_capped, return_dist_matrix=True)
    cluster_results["cos_diversity"] = cos_div
    print("Cos diversity computed in: ", time.time() - s)

    for k in [5, 10, 20, 50]:
        s = time.time()
        cos_div = eval_utils.compute_knn_cos_diversity(cluster_embeddings_capped, k=k, dist_matrix=d_matrix)
        cluster_results[f"knn_{k}_cos_diversity"] = cos_div
        print(f"KNN {k} cos diversity computed in: ", time.time() - s)

    s = time.time()
    cluster_results['kl_entropy'] = eval_utils.compute_kl_entropy(cluster_projections_capped, k=5)  # in the end this could be done in the focus projections for consistency?
    print("KL entropy computed in: ", time.time() - s)

    s = time.time()
    cluster_results['gaussian_loss'], cluster_results['gaussian_bic'], cluster_results['gaussian_aic'] = eval_utils.compute_gaussianes(cluster_projections_capped) # in the end this could be done in the focus projections for consistency?
    print("KL entropy computed in: ", time.time() - s)

    metrics_to_show = ["text_len", "word_entropy", "kl_entropy", "diversity_selfbleu", "cos_diversity", "llama_quality_scale"]
    print("\t".join([f"{m}: {np.mean(cluster_results[m]):.2f}" for m in metrics_to_show]))
    return cluster_results


def get_clustering_string(
        clustering_method,
        dbscan_min_samples, dbscan_eps,
        hdbscan_min_cluster_size, hdbscan_min_samples, hdbscan_cluster_selection_method, hdbscan_cluster_selection_epsilon,
        kmean_n_clusters,
        gmm_n_components,
        no_noise
):
    if clustering_method == 'dbscan':
        return f"dbscan_({dbscan_min_samples},{dbscan_eps}){'_no_noise' if no_noise else ''}"
    elif clustering_method == 'hdbscan':
        return f"hdbscan_({hdbscan_min_cluster_size},{hdbscan_min_samples},{hdbscan_cluster_selection_method},{hdbscan_cluster_selection_epsilon}){'_no_noise' if no_noise else ''}"
    elif clustering_method == f'kmeans':
        return f"kmeans_({kmean_n_clusters}){'_no_noise' if no_noise else ''}"
    elif clustering_method == f'gmm':
        return f"gmm_({gmm_n_components}){'_no_noise' if no_noise else ''}"
    else:
        return f"clustering_{clustering_method}{'_no_noise' if no_noise else ''}"


def evaluate_clustering(predicted_labels, true_labels):
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)

    mask = predicted_labels != -1
    true_labels = true_labels[mask]
    predicted_labels = predicted_labels[mask]

    # Compute clustering metrics
    metrics = {
        'Adjusted Rand Index': adjusted_rand_score(true_labels, predicted_labels),
        'Adjusted Mutual Information': adjusted_mutual_info_score(true_labels, predicted_labels),
        'Homogeneity Score': homogeneity_score(true_labels, predicted_labels),
        'Completeness Score': completeness_score(true_labels, predicted_labels),
        'V-measure Score': v_measure_score(true_labels, predicted_labels),
    }

    return metrics


def compute_l2_diversities(top_clusters, projections, cluster_labels, sample_size=None):
    cprint("Computing l2 diversities", "blue")
    lab_2_div = {}

    for cluster_label, cluster_size in top_clusters:
        cluster_indices = np.where(cluster_labels == cluster_label)[0]
        if cluster_indices is not None:
            cluster_indices = cluster_indices[:sample_size]

        pairwise_distances = euclidean_distances(projections[cluster_indices])

        l2_div = pairwise_distances[np.triu_indices(len(pairwise_distances), k=1)].mean()
        lab_2_div[cluster_label] = l2_div

    return lab_2_div


def plot_and_save(points, colors, title, save_path, legend_label_color_pairs=None):
    plt.clf()
    plt.scatter(points[:, 0], points[:, 1], c=colors, s=1, marker=".", edgecolors='none', linewidths=0)
    plt.title(title)
    if legend_label_color_pairs is not None:
        legend_elements = [
            plt.Line2D([0], [0], marker="o", color=color, label=label, linestyle="None") for label, color in
            legend_label_color_pairs
        ]
        plt.legend(handles=legend_elements, loc="upper right", fontsize="x-small")

    # create parent dirs if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")


def compute_embeddings(dataset, save_embeddings_path=None):

    texts = dataset['text']

    embed_model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
    embed_model.max_seq_length = 512

    print("Embedding the full dataset")
    pool = embed_model.start_multi_process_pool()
    embeddings = embed_model.encode_multi_process(
        texts,
        pool=pool, batch_size=64,
        show_progress_bar=True, normalize_embeddings=True,
    )
    embed_model.stop_multi_process_pool(pool)

    if save_embeddings_path:
        with open(save_embeddings_path, "wb") as f:
            np.save(f, embeddings)
        print("Embeddings saved to: ", save_embeddings_path)
        # dataset = dataset.add_column("stella_embeddings", list(full_dataset_embeddings))
        # dataset.save_to_disk("./data/webis/prepared-quality-embeddings-cleaned-200-minus-20-plus-corpus-webis-tldr-17")

    return embeddings

# len based clusters
def len_based_clustering(dataset):
    lens = dataset.select(range(full_umap_sample_size))['text_len']
    q1, q2, q3 = np.percentile(lens, [25, 50, 75])
    sample_labels = []
    for l in lens:
        if l <= q1:
            sample_labels.append(0)
        elif l <= q2:
            sample_labels.append(1)
        elif l <= q3:
            sample_labels.append(2)
        else:
            sample_labels.append(3)
    return sample_labels

# pipeline would be
# python create_webis_miniclusters.py (run_miniclusters.sh)
# python evaluate_webis_clusters.py ....pkl (run_eval_webis_clusters.sh)
# python select_representative_clusters.py ....pkl (run_select_representative_clusters.sh)

if __name__ == "__main__":
    # take 90k texts
    full_umap_sample_size = 90_000

    # def
    # umap_n_neighbors = 15
    # umap_min_dist = 0.1
    # dbscan_eps = 0.08
    # dbscan_min_samples = 50  # def
    # exp_tag = "def"

    parser = argparse.ArgumentParser(description="Clustering parameters")
    parser.add_argument('--umap_n_neighbors', type=int, default=5, help='Number of neighbors for UMAP')
    parser.add_argument('--umap_min_dist', type=float, default=0.0001, help='Minimum distance for UMAP')
    parser.add_argument('--clustering_method', type=str, default='dbscan', choices=['dbscan', 'hdbscan', 'kmeans', 'gmm'], help='Clustering method')
    parser.add_argument('--dbscan_eps', type=float, default=0.2, help='Epsilon for DBSCAN')
    parser.add_argument('--dbscan_min_samples', type=int, default=100, help='Minimum samples for DBSCAN')
    parser.add_argument('--hdbscan_min_cluster_size', type=int, default=4000, help='Minimum cluster size for HDBSCAN')
    parser.add_argument('--hdbscan_min_samples', type=int, default=10, help='Minimum samples for HDBSCAN')
    parser.add_argument('--hdbscan_cluster_selection_method', type=str, default='leaf', help='Cluster selection method for HDBSCAN')
    parser.add_argument('--hdbscan_cluster_selection_epsilon', type=float, default=0.0)
    parser.add_argument('--kmeans_n_clusters', type=int, default=5)
    parser.add_argument('--gmm_n_components', type=int, default=5)
    parser.add_argument('--exp_tag', type=str, default='many_clusters', help='Experiment tag')
    parser.add_argument('--use_cache', action='store_true', help='Use cache')
    parser.add_argument('--plot_focus_umaps', action='store_true')
    parser.add_argument('--no_noise', action='store_true')
    parser.add_argument('--min_focus_cluster_size', type=int, default=250)
    args = parser.parse_args()
    print(args)

    umap_n_neighbors = args.umap_n_neighbors
    umap_min_dist = args.umap_min_dist
    dbscan_eps = args.dbscan_eps
    dbscan_min_samples = args.dbscan_min_samples
    hdbscan_min_cluster_size = args.hdbscan_min_cluster_size
    hdbscan_min_samples = args.hdbscan_min_samples
    hdbscan_cluster_selection_method = args.hdbscan_cluster_selection_method
    hdbscan_cluster_selection_epsilon = args.hdbscan_cluster_selection_epsilon
    kmeans_n_clusters = args.kmeans_n_clusters
    gmm_n_components = args.gmm_n_components
    exp_tag = args.exp_tag
    clustering_method = args.clustering_method
    no_noise = args.no_noise
    min_focus_cluster_size = args.min_focus_cluster_size

    clustering_string = get_clustering_string(
        clustering_method,
        dbscan_min_samples, dbscan_eps,
        hdbscan_min_cluster_size, hdbscan_min_samples, hdbscan_cluster_selection_method, hdbscan_cluster_selection_epsilon,
        kmeans_n_clusters,
        gmm_n_components,
        no_noise
    )

    # exp_tag = "subreddit_cluster"

    # chosen
    # umap_n_neighbors = 5
    # umap_min_dist = 0.001
    # dbscan_eps=0.2
    # dbscan_min_samples = 50

    # save_dir = "./viz_results/webis_per_cluster/"
    save_dir = f"./viz_results/webis_{exp_tag}/"
    cache_dir = f"./viz_results/.cache/"
    print(f"Save dir: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    print("Loading dataset")
    dataset_path = "./data/webis/prepared-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17"
    embeddings_path = "./data/webis/prepared-quality-embeddings-cleaned-200-minus-20-plus-corpus-webis-tldr-17_embeddings.npy"

    split_dataset = datasets.load_from_disk(dataset_path)
    dataset = datasets.concatenate_datasets([split_dataset['train'], split_dataset['test']])

    # embed dataset
    # embeddings = compute_embeddings(dataset=dataset, save_embeddings_path=embeddings_path)

    # load precomputed embeddings
    print("Load embeddings")
    with open(embeddings_path, "rb") as f:
        full_dataset_embeddings = np.load(f)
    print(full_dataset_embeddings[0])
    exit()

    # fit sample UMAP
    ################
    sample_cache_path = f"{cache_dir}/sample_projections_sample_umap_{umap_n_neighbors}_{umap_min_dist}.npy"
    full_dataset_cache_full_path = f"{cache_dir}/full_dataset_projections_sample_umap_{umap_n_neighbors}_{umap_min_dist}.npy"
    umap_cache_path = f"{cache_dir}/sample_umap_{umap_n_neighbors}_{umap_min_dist}.pkl"

    try:
        assert args.use_cache  # ugly
        with open(sample_cache_path, "rb") as f:
            cprint("Loading projections from cache", "blue")
            sample_projections_sample_umap = np.load(f)

        # load umap
        with open(umap_cache_path, "rb") as f:
            cprint("Loading UMAP from cache", "blue")
            sample_umap = pickle.load(f)

        with open(full_dataset_cache_full_path, "rb") as f:
            cprint("Loading full dataset projections from cache", "blue")
            full_dataset_projections_sample_umap = np.load(f)
    except:
        cprint(f"Fitting sample_umap ({full_umap_sample_size})", "blue")
        sample_embeddings = full_dataset_embeddings[:full_umap_sample_size]
        sample_umap = UMAP(n_components=2, metric="cosine", n_neighbors=umap_n_neighbors, min_dist=umap_min_dist).fit(sample_embeddings)

        sample_projections_sample_umap = sample_umap.transform(sample_embeddings)

        if args.use_cache:
            with open(umap_cache_path, "wb") as f:
                pickle.dump(sample_umap, f)

            with open(sample_cache_path, "wb") as f:
                print("Saving projections to cache")
                np.save(f, sample_projections_sample_umap)

        full_dataset_projections_sample_umap = None

    if clustering_method == 'dbscan':
        print(f"dbscan clustering in the sample UMAP; params: {dbscan_eps:.2f}, {dbscan_min_samples}")
        clustering = DBSCAN(
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            n_jobs=32, metric="euclidean"
        ).fit(sample_projections_sample_umap)
        sample_labels = clustering.labels_
    elif clustering_method == 'hdbscan':
        print("hdbscan clustering in the sample UMAP")
        clustering = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_method=hdbscan_cluster_selection_method,
            cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
            core_dist_n_jobs=32, metric="euclidean"
        ).fit(sample_projections_sample_umap)
        sample_labels = clustering.labels_
    elif clustering_method == 'kmeans':
        print(f"kmeans clustering in the sample UMAP; params: {kmeans_n_clusters}")
        clustering = MiniBatchKMeans(n_clusters=kmeans_n_clusters).fit(sample_projections_sample_umap)
        sample_labels = clustering.labels_
    elif clustering_method == 'gmm':
        print(f"gmm clustering in the sample UMAP; params: {gmm_n_components}")
        clustering = GaussianMixture(n_components=gmm_n_components).fit(sample_projections_sample_umap)
        sample_labels = clustering.predict(sample_projections_sample_umap)
    elif clustering_method == 'subreddit':
        top_s = {
            'AskReddit', 'leagueoflegends', 'relationships', 'funny', 'gaming', 'AdviceAnimals',
            'pics', 'trees', 'atheism', 'politics', 'WTF', 'todayilearned', 'explainlikeimfive',
        }
        s_to_i = dict(enumerate(top_s))
        sample_labels = [s_to_i.get(s, -1) for s in dataset.select(range(full_umap_sample_size))['subreddit']]

    true_labels = dataset.select(range(full_umap_sample_size))['subreddit']
    clustering_metrics = evaluate_clustering(sample_labels, true_labels)
    pprint.pprint(clustering_metrics)

    # define color mapping
    unique_clusters = set(sample_labels)
    # remove white colors
    all_colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.XKCD_COLORS.keys()) * 10

    cluster_2_color_dict = dict(zip(unique_clusters, all_colors))
    cluster_2_color_dict[-1] = "black"

    # cluster_counts = Counter(sample_labels)
    cprint("Plot sample in the sample_umap", "green")
    legend_label_color_pairs = [
        (f"Cluster {cluster_id} ({n})", cluster_2_color_dict[cluster_id]) for cluster_id, n in Counter(sample_labels).most_common(10)
    ]

    plot_and_save(
        points=sample_projections_sample_umap,
        colors=[cluster_2_color_dict[ci] for ci in sample_labels],
        title=f"Sample in the sample UMAP; Compl {clustering_metrics['Completeness Score']:.2f}",
        save_path=f"{save_dir}/sample_in_sample_umap/umap_({umap_n_neighbors},{umap_min_dist})/{clustering_string}.png",
        legend_label_color_pairs=legend_label_color_pairs
    )
    # pprint.pprint([(c, n, 15*n) for c, n in Counter(sample_labels).most_common()[:20]])

    # Infer cluster labels for the full dataset
    ###########################################
    cprint("Infer cluster labels for the full dataset", "blue")
    print("Building classifier faiss index without noise")
    start_time = time.time()
    # use only top clusters for faiss index

    # top_k_faiss = 30
    # top_cs_no_noise, top_cs_no_noise_ns = zip(*Counter(sample_labels[sample_labels != -1]).most_common(top_k_faiss))
    # top_cs, top_cs_ns = zip(*Counter(sample_labels).most_common(top_k_faiss))
    # mask = np.isin(sample_labels, top_cs)

    mask = np.ones_like(sample_labels).astype(bool)
    if args.no_noise:
        mask = sample_labels != -1  # remove noise cluster

    faiss_labels = sample_labels[mask]
    faiss_projections = sample_projections_sample_umap[mask]
    print("Faiss size: ", faiss_projections.shape[0])

    classifier_faiss_index = faiss.IndexFlatL2(faiss_projections.shape[1])
    classifier_faiss_index.add(faiss_projections)
    end_time = time.time()
    print("Faiss index built in: ", end_time - start_time)

    start_time = time.time()
    if full_dataset_projections_sample_umap is None:
        print("Projecting the full dataset")
        full_dataset_projections_sample_umap = sample_umap.transform(full_dataset_embeddings)

        if args.use_cache:
            # save to cache
            with open(full_dataset_cache_full_path, "wb") as f:
                print("Saving full dataset projections to cache")
                np.save(f, full_dataset_projections_sample_umap)
    else:
        print("Using cached full dataset projections")

    end_time = time.time()
    print("Projected the full dataset in: ", end_time - start_time)

    # find nearest neighbours
    print("Classifying the full dataset")
    top_k = 1
    start_time = time.time()
    queries = full_dataset_projections_sample_umap
    dist, neighbours_inds = classifier_faiss_index.search(queries, top_k)

    full_dataset_labels = []
    for n_inds in neighbours_inds:
        n_labels = [faiss_labels[i] for i in n_inds]
        full_dataset_labels.append(Counter(n_labels).most_common(1)[0][0])
    end_time = time.time()
    cprint(f"Inference time: {end_time - start_time}", "green")

    # create labels
    full_dataset_cluster_count = Counter(full_dataset_labels)
    legend_label_color_pairs = [
        (f"Cluster {cluster_id} ({n})", cluster_2_color_dict[cluster_id]) for cluster_id, n in full_dataset_cluster_count.most_common(10)
    ]

    # plot inferred labels for the full dataset in sample_UMAP
    ########################################################
    cprint("Plot full dataset labels in sample_UMAP", "green")
    plot_and_save(
        points=full_dataset_projections_sample_umap,
        colors=[cluster_2_color_dict[ci] for ci in full_dataset_labels],
        title="Full dataset in sample_UMAP",
        save_path=f"{save_dir}/full_dataset_in_sample_umap/umap_({umap_n_neighbors},{umap_min_dist})/{clustering_string}.png",
        legend_label_color_pairs=legend_label_color_pairs
    )


    def compute_cluster_center(label, projections, labels):
        cluster_indices = np.where(labels == label)[0]
        assert cluster_indices.size > 0
        return projections[cluster_indices].mean(axis=0)

    # calculate cluster centers
    cluster_centers_sample_umap = {}
    for cluster_id in full_dataset_cluster_count.keys():
        cluster_centers_sample_umap[cluster_id] = compute_cluster_center(
            cluster_id, full_dataset_projections_sample_umap, full_dataset_labels
        )

    # Select clusters
    ###########################
    n_clusters_to_sample = 10
    full_dataset_cluster_count.pop(-1, None)  # remove noise cluster if exists
    # top_clusters_no_noise = clusters_count.most_common(10)
    clusters_to_sample = {k: v for k, v in full_dataset_cluster_count.items() if v > min_focus_cluster_size}
    selected_clusters = random.sample(list(clusters_to_sample.items()), min(10, len(clusters_to_sample)))
    # top_clusters_no_noise = list(clusters_to_sample.items())

    # Merge clusters
    ###################
    if len(selected_clusters) < n_clusters_to_sample:
        n_merged_clusters = n_clusters_to_sample - len(selected_clusters)
        print(f"Adding {n_merged_clusters} merged clusters")

        remaining_cluster_labels = list(full_dataset_cluster_count.keys())

        for c_l, c_n in selected_clusters:
            remaining_cluster_labels.remove(c_l)

        random.shuffle(remaining_cluster_labels)
        merge_cluster_mapping = {}

        for i in range(n_merged_clusters):
            diversity_merging = bool(i % 2 == 0)  # complex merging

            merged_cluster_id = None
            merge_cluster_size = 0
            currently_added_clusters = []

            # are there enough datapoints
            if sum([full_dataset_cluster_count[l] for l in remaining_cluster_labels]) >= min_focus_cluster_size:

                # create a merged cluster
                while merge_cluster_size < min_focus_cluster_size:
                    if diversity_merging:  # uniform merging or first cluster
                        if merged_cluster_id is None:
                            # first cluster
                            cl = remaining_cluster_labels.pop(0)  # take first
                        else:
                            # find the farthest cluster from currently added clusters
                            distances = []
                            # distances of all added clusters to all remaining clusters
                            for a_c in currently_added_clusters:
                                distances.append(np.linalg.norm([
                                    cluster_centers_sample_umap[a_c] - cluster_centers_sample_umap[cl] \
                                for cl in remaining_cluster_labels], axis=1))

                            distances = np.array(distances)
                            assert distances.shape == (len(currently_added_clusters), len(remaining_cluster_labels))
                            # find the cluster that is the farther away from all added clusters
                            distances = np.sum(distances, axis=0)
                            cl = remaining_cluster_labels.pop(np.argmax(distances))  # pop the farthest cluster

                    else:
                        # uniform merging
                        cl = remaining_cluster_labels.pop(0)  # take first

                    if merged_cluster_id is None:
                        # first label as merged cluster label
                        merged_cluster_id = cl

                    merge_cluster_size += full_dataset_cluster_count[cl]
                    merge_cluster_mapping[cl] = merged_cluster_id
                    currently_added_clusters.append(cl)

        # remaps full_cluster_labels
        full_dataset_labels = [merge_cluster_mapping.get(l, l) for l in full_dataset_labels]
        full_dataset_cluster_count = Counter(full_dataset_labels)

        added_clusters_labels = list(set(merge_cluster_mapping.values()))
        selected_clusters.extend([(c, full_dataset_cluster_count[c]) for c in added_clusters_labels])

    # plot selected clusters (full) in sample_UMAP
    ##########################################################
    legend_label_color_pairs = [
        (f"Cluster {cluster_id} ({n})", cluster_2_color_dict[cluster_id]) for cluster_id, n in selected_clusters
    ]

    selected_clusters_labels = list(zip(*selected_clusters))[0]
    cprint("Plot full dataset SELECTED labels in sample_UMAP", "green")
    plot_and_save(
        points=full_dataset_projections_sample_umap,
        colors=[cluster_2_color_dict[ci] if ci in selected_clusters_labels else "black" for ci in full_dataset_labels],
        title="Selected clusters (full) in sample_UMAP",
        save_path=f"{save_dir}/full_dataset_selected_in_sample_umap/umap_({umap_n_neighbors},{umap_min_dist})/{clustering_string}.png",
        legend_label_color_pairs=legend_label_color_pairs
    )

    per_cluster_results = {"args": vars(args), "clusters": {}}

    for c_i, (cluster_label, cluster_size) in enumerate(selected_clusters):
        print(f"Cluster {c_i}/{n_clusters_to_sample}")
        print(f"Focusing on cluster {cluster_label} ({cluster_size})")
        cluster_indices_sample = np.where(sample_labels == cluster_label)[0]

        # Full Cluster
        #########################################################
        full_cluster_indices = np.where(full_dataset_labels == cluster_label)[0]

        if args.plot_focus_umaps:
            # full cluster in focus UMAP
            cprint(f"Fitting full_focus_umap for cluster {cluster_label}", "blue")
            full_cluster_projections_full_focus_umap = UMAP(n_components=2, metric="cosine", n_neighbors=umap_n_neighbors, min_dist=umap_min_dist).fit_transform(
                full_dataset_embeddings[full_cluster_indices]
            )

            cprint("Plot full cluster in the full_focus_UMAP", "green")
            plot_and_save(
                points=full_cluster_projections_full_focus_umap,
                colors=[cluster_2_color_dict[cluster_label]] * len(full_cluster_indices),
                title=f"Full cluster {cluster_label} in full_focus_UMAP",
                save_path=f"{save_dir}/cluster_in_focus_umap/umap_({umap_n_neighbors},{umap_min_dist})/{clustering_string}/cluster_{cluster_label}.png",
            )

        # compute cluster metrics
        cluster_dataset = dataset.select(full_cluster_indices)
        cluster_embeddings = full_dataset_embeddings[full_cluster_indices]
        cluster_projections = full_dataset_projections_sample_umap[full_cluster_indices]

        per_cluster_results["clusters"][cluster_label] = compute_cluster_metrics(
            cluster_dataset, cluster_embeddings, cluster_projections, cap_size=250
        )

        cluster_save_path = f"./data/webis/clusters/umap_({umap_n_neighbors},{umap_min_dist})_{clustering_string}_cluster_{cluster_label}"
        per_cluster_results["clusters"][cluster_label]["cluster_path"] = cluster_save_path
        per_cluster_results["clusters"][cluster_label]["cluster_indices"] = full_cluster_indices
        cluster_dataset.save_to_disk(cluster_save_path)

    # save per cluster results
    save_path = f"{save_dir}/results/umap_({umap_n_neighbors},{umap_min_dist})_{clustering_string}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(per_cluster_results, f)

    print(f"Results saved to {save_path}")

