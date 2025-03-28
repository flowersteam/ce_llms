import os
from pathlib import Path
import re
import itertools
import datasets
import json
from collections import defaultdict
import pickle

import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys

if __name__ == '__main__':

    # Ensure at least one argument is provided
    if len(sys.argv) < 2:
        print("Usage: python script.py <experiment_dir>")
        sys.exit(1)

    # Use the first argument as the experiment directory
    experiment_dir = Path(sys.argv[1])

    generations = list(range(0, 20, 3))
    sample_size = 500
    umap_path = "viz_results/webis_per_cluster_v3/sample_umap.pkl"

    # assumes seeds are seed_0_.....
    seeds = sorted(set([s.name.split("_202")[0] for s in experiment_dir.glob("generated_*/seed_1_*")]))
    seed = seeds[0]

    save_dir = Path("eval_results") / experiment_dir / "visualizations"

    dirs = list((experiment_dir.glob(f"generated_*/{seed}_*")))
    # dirs = list((experiment_dir.glob(f"generated_2000_*/{seed}_*"))) + list((experiment_dir.glob(f"generated_4000_*/{seed}_*")))

    dirs = sorted(dirs, key=lambda d: int(str(d).split("generated_")[1].split("_")[0]))
    print("Dirs:", len(dirs))

    # compute embeddings

    logs = []
    for di in dirs:
        for gen_i in generations:
            logs.append(json.loads((Path(di) / f"gen_{gen_i}" / f"part_0/log.json").read_text(encoding="UTF-8")))

    gen_n_s = []
    for di in dirs:
        for gen_i in generations:
            gen_n_s.append(json.loads((Path(di) / f"gen_1" / "log_sample_datasets.json").read_text(encoding="UTF-8"))["args"]["per_participant_ai_dataset_size"])

    ft_sizes = [log['data']['dataset_size'] for log in logs]

    print("Loading and embedding data")
    ds = []
    for d_i, di in enumerate(dirs):
        print(f"Dir {d_i+1}/{len(dirs)}")
        for gen_i in generations:
            # print(f"N generations: {gen_i}/{generations[-1]}")
            # d = datasets.load_from_disk(str(Path(di) / f"gen_{gen_i}" / f"part_0/input_dataset"))
            # d = datasets.load_from_disk(str(Path(di) / f"gen_{gen_i}" / f"part_0/full_output_dataset"))
            d = datasets.load_from_disk(str(Path(di) / f"gen_{gen_i}" / f"new_dataset"))

            if sample_size is not None:
                try:
                    d = d.select(range(sample_size))
                except:
                    from IPython import embed; embed();

            if "source" not in d.column_names:
                source_column = [f"AI_gen_{gen_i}_part_part_0"] * len(d['text'])
                d = d.add_column("source", source_column)

            ds.append(d)

    assert len(ds) == len(dirs) * len(generations)

    # # create labels
    # cos_divs = []
    # for d in ds:
    #     d_ = d.filter(lambda ex: [s.startswith("AI") for s in ex['source']], batched=True)
    #     if len(d_) == 0:
    #         cos_divs.append(compute_cos_diveristy(d[embedder.embedding_column_name]))
    #     else:
    #         cos_divs.append(compute_cos_diveristy(d_[embedder.embedding_column_name]))

    labels = [f"Gen: {n}/{s}" for n, s in zip(gen_n_s, ft_sizes)]

    # Visualize

    # 1. compute joint representation space
    joint_dataset = datasets.concatenate_datasets(ds, axis=0, info=None)
    dataset_lens = [len(d) for d in ds]

    from create_per_cluster_webis_datasets import compute_embeddings
    print("Embedding")
    X = compute_embeddings(joint_dataset)
    # Assuming dataset_lens contains lengths of original datasets
    cumulative_lens = np.cumsum([0] + dataset_lens)

    for method in ["umap"]:
        print(f"Computing {method}")
        if method == "umap":
            with open(umap_path, "rb") as f:
                umap = pickle.load(f)
            repr_dataset = umap.transform(X)
        elif method == "pca":
            ss_X = StandardScaler().fit_transform(X)
            repr_dataset = PCA(n_components=2).fit_transform(ss_X)
        elif method == "tsne":
            ss_X = StandardScaler().fit_transform(X)
            repr_dataset = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(ss_X)
        else:
            raise NotImplementedError(f"Method {method} unknown.")

        split_datasets = [repr_dataset[cumulative_lens[i]:cumulative_lens[i + 1]] for i in range(len(dataset_lens))]

        # Create a single figure for ALL plots (all dirs and generations)
        plt.clf()
        plt.style.use('seaborn-v0_8-darkgrid')

        # Calculate total number of rows (dirs) and columns (generations)
        n_rows = len(dirs)
        n_cols = len(generations)

        # Create a single figure with subplots for all directories and generations
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows + 1), sharex=True, sharey=True)
        if n_rows == 1:
            axes = [axes]
        axes = np.array(axes)

        # Calculate the overall min and max for both axes
        all_data = np.vstack(split_datasets)
        x_min, x_max = all_data[:, 0].min(), all_data[:, 0].max()
        y_min, y_max = all_data[:, 1].min(), all_data[:, 1].max()

        # Add some padding to the limits
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

        human_color = "black"
        ai_color = "red"
        print(f"Plotting {method}")

        for dir_i in range(len(dirs)):
            indices = range(dir_i * len(generations), (dir_i + 1) * len(generations))
            lab = labels[indices[0]]

            for gen_i, ind in enumerate(indices):
                d = ds[ind]
                split_data = split_datasets[ind]
                # div = cos_divs[ind]

                ax = axes[dir_i, gen_i]
                colors = [human_color if s.startswith("human") else ai_color for s in d['source']]
                ax.scatter(split_data[:, 0], split_data[:, 1], alpha=0.6, s=5, c=colors)
                # ax.set_title(f"{lab}\nGen {generations[gen_i]} Cos div {div:.2f}")
                ax.set_title(f"{lab}\nGen {generations[gen_i]}")

                # Set the same limits for all subplots
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)

                ax.grid(True)
                if dir_i == len(dirs) - 1:  # Only add xlabel on bottom row
                    ax.set_xlabel(f"{method} Component 1")
                if gen_i == 0:  # Only add ylabel on leftmost column
                    ax.set_ylabel(f"{method} Component 2")

        # Adjust spacing between subplots
        fig.suptitle(f"{method} (sample size: {sample_size})\n{experiment_dir}", fontsize=30)
        plt.tight_layout()

        custom_legend = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=human_color, markersize=20, label='Human'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=ai_color, markersize=20, label='AI')
        ]
        fig.legend(handles=custom_legend, loc='upper right', fontsize=60, frameon=False)

        method_dir = save_dir / "stella" / method
        os.makedirs(method_dir, exist_ok=True)
        savepath = method_dir / f"{seed}_combined.svg"
        plt.savefig(savepath)
        print(f"Saved to: {savepath}")