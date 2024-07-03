import time
from contextlib import contextmanager
import matplotlib.pyplot as plt

import numpy as np

from datasets import concatenate_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


@contextmanager
def timer_block():
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


def plot_and_save(x, ys, labels, ylabel=None, save_path=None, yticks=None, no_show=True):

    assert len(ys[0]) == len(x)
    assert len(labels) == len(ys)

    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)

    if yticks:
        plt.yticks(yticks)

    if ylabel:
        plt.ylabel(ylabel)

    plt.legend()

    if not no_show:
        plt.show()

    if save_path:
        plt.savefig(save_path+".png", dpi=300)
        plt.savefig(save_path+".svg")
        plt.clf()
        print(f"Saved to: {save_path}")


def plot_repr(embs, dataset_lens, labels, save_path):
    assert len(embs) == sum(dataset_lens)

    start = 0
    for d_i, d_l in enumerate(dataset_lens):
        s, e = start, start + d_l
        plt.scatter(embs[s:e, 0], embs[s:e, 1], s=1)
        start = e

    plt.legend(labels, loc="right", markerscale=5)

    for ext in ["svg", "png"]:
        fig_save_path = f"{save_path}.{ext}"
        plt.savefig(fig_save_path, dpi=300)
        print(f"Saved to: {fig_save_path}")

    plt.clf()


def visualize_datasets(datasets, dataset_labels, experiment_tag):
    # Visualize embeddings
    dataset = concatenate_datasets(datasets, axis=0, info=None)

    dataset_lens = [len(d) for d in datasets]

    X = np.array(dataset['embeddings'])
    ss_X = StandardScaler().fit_transform(X)

    # PCA
    print("PCA fitting")
    with timer_block():
        pca_dataset = PCA(n_components=2).fit_transform(ss_X)
        plot_repr(pca_dataset, dataset_lens, dataset_labels, save_path=f"viz_results/{experiment_tag}_pca")

    print("UMAP fitting")
    with timer_block():
        umap_X = umap.UMAP().fit_transform(ss_X)
        plot_repr(umap_X, dataset_lens, dataset_labels, save_path=f"viz_results/{experiment_tag}_umap")

    print("TSNE fiting")
    with timer_block():
        tsne_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)
        plot_repr(tsne_embedded, dataset_lens, dataset_labels, save_path=f"viz_results/{experiment_tag}_tsne")



