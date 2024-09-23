import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np

from datasets import concatenate_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os
from scipy.spatial import ConvexHull



@contextmanager
def timer_block():
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


def plot_and_save(x, ys, labels, ylabel=None, save_path=None, yticks=None, no_show=True, per_text=False):

    assert len(ys[0]) == len(x)
    assert len(labels) == len(ys)

    colors = list(mcolors.CSS4_COLORS.keys())
    colors = ["red", "blue", "green", "black", "brown"] + colors[::-1]

    for y, label, col in zip(ys, labels, colors):

        if per_text:
            # per text
            print("PER TEXT")
            # todo: doesn't work with different size datasets -> fix
            vp = plt.violinplot(y, showmeans=True)





            # Manually set the color
            for body in vp['bodies']:
                body.set_facecolor(col)
                # body.set_edgecolor(col)
                body.set_alpha(0.2)  # Optionally set the transparency
                # body.set_alpha(1)

            vp['cmeans'].set_color(col)
            vp['cbars'].set_color(col)
            vp['cmaxes'].set_color(col)
            vp['cmins'].set_color(col)

            ax = plt.gca()
            ax.plot([], [], c=col, label=label)

            ax.set_xticks(range(1, len(x) + 1))
            ax.set_xticklabels(x)
            # for gen_label, gen_y in zip(x, y):
            #     plt.scatter([gen_label] * len(gen_y), gen_y, s=1)
        else:
            # # std
            # plt.plot(x, [np.std(y[i]) for i in range(len(y))], label=label)
            # ylabel = ylabel + "_std"

            # mean
            plt.plot(x, [np.mean(y[i]) for i in range(len(y))], label=label, c=col)
            try:
                # plot standard deviation
                # plt.fill_between(x, [np.mean(y[i]) - np.std(y[i]) for i in range(len(y))], [np.mean(y[i]) + np.std(y[i]) for i in range(len(y))], alpha=0.2)

                # plot standard error
                plt.fill_between(x, [np.mean(y[i]) - np.std(y[i]) / np.sqrt(len(y[i])) for i in range(len(y))], [np.mean(y[i]) + np.std(y[i]) / np.sqrt(len(y[i])) for i in range(len(y))], alpha=0.2)
            except:
                ...

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




#####################################################################
## TO PLOT TOXICITY AND POLITICAL BIAS EVOLUTION ON THE SAME PLOT ###
#####################################################################

def plot_2D(y1, y2, label, save_path, no_show):
   
    plt.plot([np.mean(y1[i]) for i in range(len(y1))], [np.mean(y2[i]) for i in range(len(y2))], label=label, marker='o')
    plt.legend()
    plt.xlabel("Toxicity")
    plt.ylabel("Political Bias")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.hlines(0, -1, 1, linestyles='dashed', colors='black')
    plt.vlines(0, -1, 1, linestyles='dashed', colors='black')

    ## Add generation numbers
    for i in range(len(y1)):
        plt.annotate(i, (np.mean(y1[i]), np.mean(y2[i]) + 0.02))


    if not no_show:
        plt.show()

    if save_path:
        plt.savefig(save_path+"_2D.png", dpi=300)
        plt.savefig(save_path+"_2D.svg")
        plt.clf()
        print(f"Saved to: {save_path}_2D")



#####################################################################
## TO PLOT THE EMBEDDINGS IN 2D WITH CONVEX HULLS AND MEAN POINTS ###
#####################################################################

def plot_repr(embs, dataset_lens, labels, save_path):
    colors= ['b', 'g', 'r', 'c', 'm', 'y', 'orange', 'purple']
    assert len(embs) == sum(dataset_lens)

    start = 0
    points = {}
    previous_mean = None
    for d_i, d_l in enumerate(dataset_lens):
        s, e = start, start + d_l
        plt.scatter(embs[s:e, 0], embs[s:e, 1], s=1, alpha=0.2, c=colors[d_i], label = labels[d_i])
        #Display mean
        plt.scatter(embs[s:e, 0].mean(), embs[s:e, 1].mean(), s=100, alpha=1, c=colors[d_i], marker='x')
        if previous_mean is not None:
            plt.plot([previous_mean[0], embs[s:e, 0].mean()], [previous_mean[1], embs[s:e, 1].mean()], c='black')
        previous_mean = embs[s:e, :].mean(axis=0)
        start = e
        points[labels[d_i]] = embs[s:e]
    
    #Add a convex hull
    for d_i, d_l in enumerate(dataset_lens):
        points_hull = points[labels[d_i]]
        hull = ConvexHull(points_hull)
        for simplex in hull.simplices:
            plt.plot(points_hull[simplex, 0], points_hull[simplex, 1], alpha=1, c=colors[d_i], linewidth=2, linestyle='dashed')

        #plt.fill(points_hull[hull.vertices,0], points_hull[hull.vertices,1], alpha=0.3,  c=colors[d_i])


    plt.legend()

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
    os.makedirs("viz_results", exist_ok=True)
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



