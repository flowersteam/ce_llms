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



def load_clusters(file):
    all_results = {}
    with open(file, 'rb') as f:
        results = pickle.load(f)
        for c_number, v in results['clusters'].items():
            all_results[v['cluster_path']] = v
    return all_results

# python evaluate_webis_clusters.py viz_results/webis_miniclusters_merge_80k/results/umap_\(5\,0.01\)_kmeans_\(1000\)_no_noise.pkl
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze clustering results')
    parser.add_argument('--results-path', required=True, type=str, default="data/webis/webis_dataset_clusters/results/umap_(5,0.0001)_gmm_(5).pkl")
    parser.add_argument('--dataset-path', required=True, type=str, default="data/webis/webis_dataset_with_qualities")
    args = parser.parse_args()

    print("Load dataset from: ", args.dataset_path)
    dataset = datasets.load_from_disk(args.dataset_path)

    print("Load clusters from: ", args.results_path)
    clusters = load_clusters(args.results_path)

    for _, cluster_info in clusters.items():
        cluster_path = cluster_info['cluster_path']
        cluster_indices = cluster_info['cluster_indices']


        cluster = dataset.select(cluster_indices)

        # use this if you want to overwrite the existing cluster
        # assert cluster['text'] == datasets.load_from_disk(cluster_path)['text'], "Text mismatch between clusters"

        cluster.save_to_disk(cluster_path)
        print("Saved cluster to: ", cluster_path)





