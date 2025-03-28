import datasets
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt

import glob
# from eval_utils import *

from dataset_utils import *
# from visualization_utils import *

from datasets import load_dataset


dataset_names = [
    "webis_reddit",
    "reddit_submissions",
    "100m_tweets",
    "senator_tweets"
]

types = [
    "standard",
    # "Q51",
    # "Q80",
    # "Q20",
    # "Q40",
    # "Q60",
    # "Q80",
    # "short",
    # "medium",
    # "long"
]
# dataset_name = "webis_reddit"
# dataset_name = "reddit_submissions"
# dataset_name = "100m_tweets"

# dataset_name = "senator_tweets"

from eval_utils import llama_quality_scale, compute_quick_metrics, StellaEmbedder, compute_knn_cos_diversity, compute_var_diversity, compute_cos_diversity

stella_embedder = StellaEmbedder()

ds = defaultdict(dict)
for dataset_name in dataset_names:
    for type in types:
        print(f"dataset_name: {dataset_name}")
        d = load_human_dataset(
            dataset_name=dataset_name,
            split="all",
            load_n=250,
            dataset_type=type
        )
        d = stella_embedder.add_embeddings(d, batch_size=256)
        ds[dataset_name][type] = d

# Path(f"viz_results/webis_per_cluster_chosen_2/q_hist").mkdir(parents=True, exist_ok=True)
Path(f"viz_results/q_hist").mkdir(parents=True, exist_ok=True)
for dataset_name in dataset_names:
    print(dataset_name)
    for type in types:
        print(type)

        d = ds[dataset_name][type]
        div = compute_cos_diversity(d['stella_embeddings'])
        print("\tQ: ", np.mean(d['llama_quality_scale']), " D: ", div)
        # llama_qualities = d['llama_quality_scale']
        # llama_qualities = np.mean(llama_quality_scale(d['text']))
        # avg_q = np.mean(llama_qualities)
        # plt.clf()
        # plt.hist(llama_qualities, bins=20)
        # plt.title(f"Avg Q: {str(avg_q)}")
        # plt.savefig(f"viz_results/q_hist/{dataset_name}_{type}_llama_quality_hist.png")

        # quick_metrics = compute_quick_metrics(d)
        # print("\t :", np.mean(quick_metrics['diversity_selfbleu']))

        # print("\t quality:", np.mean(
        # print("\t selfbleu:", np.mean(quick_metrics['text_len']))
        # div, d_mat = compute_cos_diversity(d['stella_embeddings'], return_dist_matrix=True)
        # print(
        #     f"\t cos: {100*div:.1f}" + \
        #     f"\t k-3: {100*compute_knn_cos_diversity(d['stella_embeddings'], k=2, dist_matrix=d_mat):.1f}" + \
        #     f" ({div/compute_knn_cos_diversity(d['stella_embeddings'], k=2, dist_matrix=d_mat):.3f})" + \
        #     f"\t k-5: {100*compute_knn_cos_diversity(d['stella_embeddings'], k=5, dist_matrix=d_mat):.1f}" + \
        #     f" ({div/compute_knn_cos_diversity(d['stella_embeddings'], k=5, dist_matrix=d_mat):.3f})" + \
        #     f"\t k-15: {100*compute_knn_cos_diversity(d['stella_embeddings'], k=15, dist_matrix=d_mat):.1f}" + \
        #     f" ({div/compute_knn_cos_diversity(d['stella_embeddings'], k=15, dist_matrix=d_mat):.3f})" + \
        #     f"\t k-30: {100*compute_knn_cos_diversity(d['stella_embeddings'], k=30, dist_matrix=d_mat):.1f}" + \
        #     f" ({compute_knn_cos_diversity(d['stella_embeddings'], k=30, dist_matrix=d_mat):.3f})" + \
        #     f"\t k-50: {100*compute_knn_cos_diversity(d['stella_embeddings'], k=50, dist_matrix=d_mat):.1f}"
        #     f" ({div/compute_knn_cos_diversity(d['stella_embeddings'], k=50, dist_matrix=d_mat):.3f})" + \
        #     f"\t var: {compute_var_diversity(d['stella_embeddings'])}" + \
        #     f"\t len: {np.mean([len(t) for t in d['text']])}"
        # )
        # print("\t quality", np.mean(d['llama_quality_scale']))
        # print(np.mean(llama_quality_scale(d['text'])))
