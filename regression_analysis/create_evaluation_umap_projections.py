import argparse
import pickle

import numpy as np
from umap import UMAP


def fit_umap(embeddings, sample_size=90_000, n_components=2, metric="cosine", n_neighbors=5, min_dist=0.01):

    # projections
    print("Fitting UMAP")
    umap = UMAP(n_components=n_components, metric=metric, n_neighbors=n_neighbors, min_dist=min_dist).fit(embeddings[:sample_size])
    print("Projecting")
    projections = umap.transform(embeddings)

    return umap, projections


# python regression_analysis.fit_evaluation_map --viz_results/webis_miniclusters_merge_80k/results/umap_\(5\,0.01\)_kmeans_\(1000\)_no_noise.pkl
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze clustering results')
    parser.add_argument('--embeddings-path', required=True, type=str, default="./data/webis/webis_dataset_embeddings.npy")
    parser.add_argument('--umap-save-path', required=True, type=str, default="./data/webis/webis_dataset_umap.pkl")
    parser.add_argument('--projections-save-path', required=True, type=str, default="./data/webis/webis_dataset_umap_projections.npy")

    args = parser.parse_args()

    print("Load embeddings from: ", args.embeddings_path)
    with open(args.embeddings_path, "rb") as f:
        embeddings = np.load(f)

    umap, projections = fit_umap(embeddings)

    print("Saving UMAP to: ", args.umap_save_path)
    with open(args.umap_save_path, "wb") as f:
        pickle.dump(umap, f)

    print("Saving projections to: ", args.projections_save_path)
    with open(args.projections_save_path, "wb") as f:
        np.save(f, projections)
