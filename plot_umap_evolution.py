import logging
from pathlib import Path
import datasets
import sys

logging.basicConfig(level=logging.INFO)


# ./viz_results/clustering/stella_5/100m_tweets/results/umap_kn_5_md_0.001/dbscan/eps_0.3_min_samples_100/

from text_clustering import ClusterClassifier
import argparse

if __name__ == "__main__":

    # SAMPLE = 100_000
    # SAMPLE = 25_000

    # NAME
    #######
    allowed_names = [
        "joint",
        "senators_tweets",
        "reddit_submissions",
        "100m_tweets",
        "webis"
    ]
    parser = argparse.ArgumentParser(description="Set dataset name from command-line arguments.")
    parser.add_argument("--name", type=str, default="joint", choices=allowed_names)
    args = parser.parse_args()
    name = args.name


    # Use the first argument as the experiment directory
    experiment_dir = Path(sys.argv[1])

    generations = list(range(0, 20, 3))
    sample_size = 500

    # assumes seeds are seed_0_.....
    seeds = sorted(set([s.name.split("_202")[0] for s in experiment_dir.glob("generated_*/seed_1_*")]))
    seed = seeds[0]

    save_dir = Path("eval_results") / experiment_dir / "visualizations"

    dirs = list((experiment_dir.glob(f"generated_*/{seed}_*")))
    # dirs = list((experiment_dir.glob(f"generated_2000_*/{seed}_*"))) + list((experiment_dir.glob(f"generated_4000_*/{seed}_*")))

    dirs = sorted(dirs, key=lambda d: int(str(d).split("generated_")[1].split("_")[0]))
    print("Dirs:", len(dirs))

    ds = []
    for d_i, di in enumerate(dirs):
        print(f"Dir {d_i + 1}/{len(dirs)}")
        for gen_i in generations:
            print(f"N generations: {gen_i}/{generations[-1]}")
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


    # folder = f"./viz_results/clustering/stella_5/{name}/"
    folder = f"./viz_results/clustering/stella_6/{name}/"

    project_params_list = [
        {"n_neighbors": 5, "min_dist": 0.001},
        # {"n_neighbors": 5, "min_dist": 0.01},
        # {"n_neighbors": 5, "min_dist": 0.1},
        # {"n_neighbors": 15, "min_dist": 0.001},
        # {"n_neighbors": 15, "min_dist": 0.01},
        # {"n_neighbors": 15, "min_dist": 0.1},
        # {"n_neighbors": 25, "min_dist": 0.001},
        # {"n_neighbors": 25, "min_dist": 0.01},
        # {"n_neighbors": 25, "min_dist": 0.1},
        # {"n_neighbors": 50, "min_dist": 0.001},
        # {"n_neighbors": 50, "min_dist": 0.01},
        # {"n_neighbors": 50, "min_dist": 0.1},
    ]

    cluster_params = [
        # (0.1, 10),
        # (0.1, 25),
        # (0.2, 10),
        # (0.2, 25),
        # (0.2, 50),
        # (0.2, 100),
        # (0.3, 50),
        # (0.2, 50),
        (0.3, 100),
        # (0.2, 50),
    ]
    cluster_method = "dbscan"

    for eps, min_samples in cluster_params:
        for i, proj_par in enumerate(project_params_list):
            cc = ClusterClassifier(
                embed_device="cuda", multigpu=True,
                cluster_method=cluster_method, dbscan_eps=eps, dbscan_min_samples=min_samples,
            )

            # now for the full dataset
            full_dataset_folder = folder + "full_dataset/"
            texts = d['text']
            _, projections, cluster_labels = cc.transform(
                texts,
                # load_embeddings_folder=None if i == 0 else full_dataset_folder, save_embeddings_folder=full_dataset_folder,
                load_embeddings_folder=full_dataset_folder, save_embeddings_folder=full_dataset_folder,
                load_umap_pickle_path=folder + f"umap/umap_kn_{proj_par['n_neighbors']}_md_{proj_par['min_dist']}/{cluster_method}/eps_{eps}_min_samples_{min_samples}.pickle",
                load_classifier_dir=folder + f"classifier/umap_kn_{proj_par['n_neighbors']}_md_{proj_par['min_dist']}/{cluster_method}/eps_{eps}_min_samples_{min_samples}/",
                labels=labels, cluster_summaries=cluster_summaries
            )
            cc.show(
                save_path=full_dataset_folder+f"umap_kn_{proj_par['n_neighbors']}_md_{proj_par['min_dist']}/{cluster_method}/eps_{eps}_min_samples_{min_samples}.png", title=name,
                projections=projections,
                cluster_labels=cluster_labels
            )
            clustering_results = cc.show_cluster_histograms(save_path=full_dataset_folder + f"histogram/umap_kn_{proj_par['n_neighbors']}_md_{proj_par['min_dist']}/{cluster_method}/eps_{eps}_min_samples_{min_samples}.png", title=name)
            results_folder = full_dataset_folder + f"results/umap_kn_{proj_par['n_neighbors']}_md_{proj_par['min_dist']}/{cluster_method}/eps_{eps}_min_samples_{min_samples}/"
            save_results(results_folder, texts, projections, cluster_labels)
