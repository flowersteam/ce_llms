import os
from collections import defaultdict, Counter
import argparse
import numpy as np
import time
import pickle
import pprint

from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import DBSCAN, HDBSCAN
import hdbscan

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from termcolor import cprint
import datasets

from sklearn.metrics.pairwise import euclidean_distances

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)

import faiss

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
    plt.savefig(save_path, dpi=300)


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
    parser.add_argument('--umap_min_dist', type=float, default=0.001, help='Minimum distance for UMAP')
    parser.add_argument('--dbscan_eps', type=float, default=0.2, help='Epsilon for DBSCAN')
    parser.add_argument('--dbscan_min_samples', type=int, default=40, help='Minimum samples for DBSCAN')
    parser.add_argument('--hdbscan_min_cluster_size', type=int, default=4000, help='Minimum cluster size for HDBSCAN')
    parser.add_argument('--hdbscan_min_samples', type=int, default=10, help='Minimum samples for HDBSCAN')
    parser.add_argument('--hdbscan_cluster_selection_method', type=str, default='leaf', help='Cluster selection method for HDBSCAN')
    parser.add_argument('--hdbscan_cluster_selection_epsilon', type=float, default=0.0)
    parser.add_argument('--exp_tag', type=str, default='many_clusters', help='Experiment tag')
    parser.add_argument('--clustering_method', type=str, default='dbscan', choices=['dbscan', 'hdbscan'], help='Clustering method')
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
    exp_tag = args.exp_tag
    clustering_method = args.clustering_method

    # exp_tag = "subreddit_cluster"

    # chosen
    # umap_n_neighbors = 5
    # umap_min_dist = 0.001
    # dbscan_eps=0.2
    # dbscan_min_samples = 50

    # save_dir = "./viz_results/webis_per_cluster/"
    save_dir = f"./viz_results/webis_{exp_tag}/"
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

    # fit sample UMAP
    ################
    sample_cache_path = f"{save_dir}/sample_projections_sample_umap_{umap_n_neighbors}_{umap_min_dist}.npy"
    full_dataset_cache_full_path = f"{save_dir}/full_dataset_projections_sample_umap_{umap_n_neighbors}_{umap_min_dist}.npy"
    umap_cache_path = f"{save_dir}/sample_umap_{umap_n_neighbors}_{umap_min_dist}.pkl"
    try:
        # assert umap_min_dist == 0.001
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

        with open(umap_cache_path, "wb") as f:
            pickle.dump(sample_umap, f)

        sample_projections_sample_umap = sample_umap.transform(sample_embeddings)
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
    elif clustering_method == 'hdbscan':
        print("hdbscan clustering in the sample UMAP")
        # clustering = HDBSCAN(
        #     min_cluster_size=hdbscan_min_cluster_size,
        #     min_samples=hdbscan_min_samples,
        #     cluster_selection_method=hdbscan_cluster_selection_method,
        #     n_jobs=32, metric="euclidean"
        # ).fit(sample_projections_sample_umap)
        clustering = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_method=hdbscan_cluster_selection_method,
            cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
            core_dist_n_jobs=32, metric="euclidean"
        ).fit(sample_projections_sample_umap)
    sample_labels = clustering.labels_

    # top_s = {
    #     'AskReddit', 'leagueoflegends', 'relationships', 'funny', 'gaming', 'AdviceAnimals',
    #     'pics', 'trees', 'atheism', 'politics', 'WTF', 'todayilearned', 'explainlikeimfive',
    # }
    # assert exp_tag == "subreddit_cluster"
    true_labels = dataset.select(range(full_umap_sample_size))['subreddit']
    clustering_metrics = evaluate_clustering(sample_labels, true_labels)
    pprint.pprint(clustering_metrics)

    # define color mapping
    unique_clusters = set(sample_labels)
    # remove white colors
    all_colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.XKCD_COLORS.keys()) * 10

    cluster_2_color_dict = dict(zip(unique_clusters, all_colors))
    cluster_2_color_dict[-1] = "black"

    clusters_count = Counter(sample_labels)

    cprint("Plot sample in the sample_umap", "green")
    legend_label_color_pairs = [
        (f"Cluster {cluster_id} ({n})", cluster_2_color_dict[cluster_id]) for cluster_id, n in clusters_count.most_common(10)
    ]

    if clustering_method == 'dbscan':
        save_path = f"{save_dir}/sample_umap_sample_umap_({umap_n_neighbors},{umap_min_dist})_dbscan_({dbscan_min_samples},{dbscan_eps}).png"
    elif clustering_method == 'hdbscan':
        save_path = f"{save_dir}/sample_umap_sample_umap_({umap_n_neighbors},{umap_min_dist})_hdbscan_({hdbscan_min_cluster_size},{hdbscan_min_samples},{hdbscan_cluster_selection_method},{hdbscan_cluster_selection_epsilon}).png"

    plot_and_save(
        points=sample_projections_sample_umap,
        colors=[cluster_2_color_dict[ci] for ci in sample_labels],
        title=f"Sample in the sample UMAP; Compl {clustering_metrics['Completeness Score']:.2f}",
        # save_path=f"{save_dir}/sample_umap_sample.png",
        save_path=save_path,
        legend_label_color_pairs=legend_label_color_pairs
    )
    print(f"Plotted to {save_path}")
    pprint.pprint([(c, n, 15*n) for c, n in Counter(sample_labels).most_common()[:20]])

    # Infer cluster labels for the full dataset
    ###########################################
    cprint("Infer cluster labels for the full dataset", "blue")
    print("Building classifier faiss index without noise")
    start_time = time.time()
    # use only top clusters for faiss index
    top_k_faiss = 30
    # top_cs_no_noise, top_cs_no_noise_ns = zip(*Counter(sample_labels[sample_labels != -1]).most_common(top_k_faiss))
    top_cs_no_noise, top_cs_no_noise_ns = zip(*Counter(sample_labels).most_common(top_k_faiss))
    mask = np.isin(sample_labels, top_cs_no_noise)
    mask = np.ones_like(mask)
    # mask = sample_labels != -1
    faiss_projections = sample_projections_sample_umap[mask]
    faiss_labels = sample_labels[mask]
    print("Faiss size: ", faiss_projections.shape[0])

    classifier_faiss_index = faiss.IndexFlatL2(faiss_projections.shape[1])
    classifier_faiss_index.add(faiss_projections)
    # classifier_faiss_index = faiss.IndexFlatL2(sample_projections_sample_umap.shape[1])
    # classifier_faiss_index.add(sample_projections_sample_umap)
    end_time = time.time()
    print("Faiss index built in: ", end_time - start_time)

    start_time = time.time()
    if full_dataset_projections_sample_umap is None:
        print("Projecting the full dataset")
        full_dataset_projections_sample_umap = sample_umap.transform(full_dataset_embeddings)

        # save to cache
        with open(full_dataset_cache_full_path, "wb") as f:
            print("Saving full dataset projections to cache")
            np.save(f, full_dataset_projections_sample_umap)
    else:
        print("Using cached full dataset projections")

    end_time = time.time()
    print("Projected the full dataset in: ", end_time - start_time)
    queries = full_dataset_projections_sample_umap

    top_k = 1
    # find nearest neighbours
    print("Classifying the full dataset")
    start_time = time.time()
    dist, neighbours_inds = classifier_faiss_index.search(queries, top_k)

    full_dataset_labels = []
    for n_inds in neighbours_inds:
        n_labels = [faiss_labels[i] for i in n_inds]
        full_dataset_labels.append(Counter(n_labels).most_common(1)[0][0])
    end_time = time.time()
    cprint(f"Inference time: {end_time - start_time}", "green")

    # check that for the sample the inferred labels are the same as the original labels
    # _, neighbours_inds_sample = classifier_faiss_index.search(sample_projections_sample_umap, top_k)
    # for lab, n_inds in zip(sample_labels, neighbours_inds_sample):
    #     # this gives indices in the sample
    #     n_labels = [sample_labels[i] for i in n_inds]
    #     inferred_label = Counter(n_labels).most_common(1)[0][0]
    #     assert inferred_label == lab


    # compute l2 diversities for the full dataset
    full_dataset_cluster_count = Counter(full_dataset_labels)
    legend_label_color_pairs = [
        (f"Cluster {cluster_id} ({n})", cluster_2_color_dict[cluster_id]) for cluster_id, n in full_dataset_cluster_count.most_common(10)
    ]

    # plot inferred labels for the full dataset in sample_UMAP
    ########################################################
    clusters_count.pop(-1, None)  # remove noise cluster if exists
    top_clusters_no_noise = clusters_count.most_common(10)

    if clustering_method == 'dbscan':
        save_path = f"{save_dir}/sample_umap_full_dataset_({umap_n_neighbors},{umap_min_dist})_dbscan_({dbscan_min_samples},{dbscan_eps}).png"
    elif clustering_method == 'hdbscan':
        save_path = f"{save_dir}/sample_umap_full_dataset_({umap_n_neighbors},{umap_min_dist})_hdbscan_({hdbscan_min_cluster_size},{hdbscan_min_samples},{hdbscan_cluster_selection_method},{hdbscan_cluster_selection_epsilon}).png"

    cprint("Plot full dataset labels in sample_UMAP", "green")
    plot_and_save(
        points=full_dataset_projections_sample_umap,
        colors=[cluster_2_color_dict[ci] for ci in full_dataset_labels],
        title="Full dataset in sample_UMAP",
        # save_path=f"{save_dir}/sample_umap_full_dataset.png",
        save_path=save_path,
        legend_label_color_pairs=legend_label_color_pairs
    )
    for cluster_label, cluster_size in top_clusters_no_noise:
        print(f"Focusing on cluster {cluster_label} ({cluster_size})")
        cluster_indices_sample = np.where(sample_labels == cluster_label)[0]

        # plot cluster in sample_umap
        ##############################
        # cprint(f"Plot cluster {cluster_label} in sample_UMAP", "green")
        # plot_and_save(
        #     points=sample_projections_sample_umap[cluster_indices_sample],
        #     colors=[cluster_2_color_dict[cluster_label]] * len(cluster_indices_sample),
        #     title=f"Cluster {cluster_label} on sample_UMAP",
        #     save_path=f"{save_dir}/sample_umap_cluster_{cluster_label}.png"
        # )

        # fit UMAP on focus cluster (from sample)
        #########################################
        # cprint(f"Fitting sample_focus_umap for cluster {cluster_label}", "blue")
        # cluster_projections_sample_focus_umap = UMAP(n_components=2, metric="cosine", n_neighbors=n_neighbors, min_dist=min_dist).fit_transform(
        #     sample_embeddings[cluster_indices_sample]
        # )

        # cprint(f"Plot cluster {sample_labels} on sample_focus_UMAP", "green")
        # plot_and_save(
        #     points=cluster_projections_sample_focus_umap,
        #     colors=[cluster_2_color_dict[cluster_label]] * len(cluster_indices_sample),
        #     title=f"Cluster {cluster_label} on sample_focus_UMAP",
        #     save_path=f"{save_dir}/sample_focus_umap_cluster_{cluster_label}.png"
        # )

        # Full Cluster
        #########################################################
        full_cluster_indices = np.where(full_dataset_labels == cluster_label)[0]

        # # full cluster in sample UMAP
        # cprint("Plot full cluster in sample_UMAP", "green")
        # plot_and_save(
        #     points=full_dataset_projections_sample_umap[full_cluster_indices],
        #     colors=[cluster_2_color_dict[cluster_label]] * len(full_cluster_indices),
        #     title=f"Full Cluster {cluster_label} in sample_UMAP",
        #     save_path=f"{save_dir}/sample_umap_full_cluster_{cluster_label}.png"
        # )

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
            save_path=f"{save_dir}/full_focus_umap_cluster_{cluster_label}.png"
        )

        # # create the focus dataset
        save_dataset_path = f"./data/webis/prepared-{exp_tag}-cluster-{cluster_label}-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17"
        na_dataset = dataset.select(full_cluster_indices)
        na_dataset.save_to_disk(save_dataset_path)
        print(f"Cluster {cluster_label} dataset saved to: ", save_dataset_path)