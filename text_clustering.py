import json
import logging
import os
import random
import textwrap
from collections import Counter, defaultdict
import pickle

import datasets
import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from tqdm import tqdm
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import numpy as np

logging.basicConfig(level=logging.INFO)


DEFAULT_INSTRUCTION = (
    instruction
) = "Use three words total (comma separated)\
to describe general topics in above texts. Under no circumstances use enumeration. \
Example format: Tree, Cat, Fireman"

DEFAULT_TEMPLATE = "<s>[INST]{examples}\n\n{instruction}[/INST]"


def save_results(folder, texts, projections, cluster_labels):
    # save projections and cluster_labels and texts
    embeddings_path = folder + "/projections.npy"
    labels_path = folder + f"/labels.npy"
    texts_path = folder + f"/texts.json"
    os.makedirs(folder, exist_ok=True)

    with open(embeddings_path, "wb") as f:
        np.save(f, projections)
    with open(labels_path, "wb") as f:
        np.save(f, cluster_labels)
    with open(texts_path, "w") as f:
        json.dump(texts, f)
    logging.info(f"Results saved to: {folder}")

# ./viz_results/clustering/stella_5/100m_tweets/results/umap_kn_5_md_0.001/dbscan/eps_0.3_min_samples_100/
def load_results(folder):
    embeddings_path = folder + "/projections.npy"
    labels_path = folder + f"/labels.npy"
    texts_path = folder + f"/texts.json"

    with open(embeddings_path, "rb") as f:
        projections = np.load(f)
    with open(labels_path, "rb") as f:
        cluster_labels = np.load(f)
    with open(texts_path, "r") as f:
        texts = json.load(f)
    logging.info(f"Results loaded from: {folder}")
    return texts, projections, cluster_labels


class ClusterClassifier:
    def __init__(
        self,
        embed_model_name="dunzhang/stella_en_1.5B_v5",
        # embed_model_name="all-MiniLM-L6-v2",
        embed_device="cpu",
        multigpu=False,
        embed_batch_size=64,
        embed_max_seq_length=512,
        umap_components=2,
        umap_metric="cosine",
        # dbscan_eps=0.1, # def
        dbscan_eps=0.2,  # works well
        dbscan_min_samples=50, # default
        # dbscan_min_samples=4, # po PS-u
        dbscan_n_jobs=16,
        cluster_method="dbscan",
    ):
        self.embed_model_name = embed_model_name
        self.embed_device = embed_device
        self.multigpu = multigpu
        self.embed_batch_size = embed_batch_size
        self.embed_max_seq_length = embed_max_seq_length

        self.proj_components = umap_components
        self.umap_metric = umap_metric

        self.cluster_method = cluster_method
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_n_jobs = dbscan_n_jobs

        # self.cluster_labels = None
        self.umap_mapper = None
        self.id2label = None
        self.cluster_label_2_datapoint_indices = None
        self.cluster_entropy = None
        self.aic = None

    def transform(self, texts,
            load_embeddings_folder=None, save_embeddings_folder=None,  # embedding
            load_umap_pickle_path=None,  # projection
            load_classifier_dir=None,  # classifier/clusterer
            classifier_faiss_index_path=None,
            labels=None, cluster_summaries=None,
            ):
        # self.texts = texts

        # embeddings
        if load_embeddings_folder:
            logging.info(f"loading precomputed embeddings from {load_embeddings_folder}")
            embeddings = self.load_embeddings(load_embeddings_folder)

        else:
            logging.info("embedding texts...")
            self.embed_model = SentenceTransformer(self.embed_model_name, device=self.embed_device)
            self.embed_model.max_seq_length = self.embed_max_seq_length
            embeddings = self.embed(texts)

        if save_embeddings_folder:
            self.save_embeddings(save_embeddings_folder, embeddings)
        logging.info("embeddings saved")

        logging.info("embedding done")

        # projection
        if load_umap_pickle_path:
            logging.info("projecting...")
            self.umap_mapper = self.load_umap(load_umap_pickle_path)
            projections = self.umap_mapper.transform(embeddings)

        else:
            raise ValueError("Umap path must be provided")
        logging.info("projections done")

        # clustering
        logging.info("classifying...")
        if labels is not None:
            logging.info("using provided cluster labels")
            cluster_labels = labels

        elif load_classifier_dir:
            logging.info("loading classifier...")
            faiss_path = load_classifier_dir+"/faiss.index"
            self.classifier_faiss_index = faiss.read_index(faiss_path)

            faiss_path = load_classifier_dir+"/faiss_proj.index"
            self.classifier_faiss_index_proj = faiss.read_index(faiss_path)

            embeddings_save_path = load_classifier_dir+"/embeddings.npy"
            with open(embeddings_save_path, "rb") as f:
                self.classifier_embeddings = np.load(f)

            cluster_labels_save_path = load_classifier_dir + "/cluster_labels.pickle"
            with open(cluster_labels_save_path, "rb") as f:
                self.classifier_cluster_labels = pickle.load(f)

        cluster_labels = self.infer_knn(embeddings=embeddings, projections=projections, top_k=1, use_projections=True)

        logging.info("classifying done...")

        # compute metrics and analysis
        self.cluster_unique_labels, self.cluster_counts, self.cluster_percentages, self.cluster_entropy = \
            self.compute_cluster_metrics(cluster_labels)

        self.cluster_centers = self.compute_cluster_centers(cluster_labels, projections)

        if cluster_summaries:
            self.cluster_summaries = cluster_summaries
        else:
            self.cluster_summaries = self.create_cluster_summaries(cluster_labels)

        return embeddings, projections, cluster_labels

    def create_cluster_summaries(self, cluster_labels):
        unique_labels = len(set([cl for cl in cluster_labels if cl != -1]))  # exclude the "-1" label if exists
        cluster_summaries = {l: str(l) for l in range(unique_labels)}
        cluster_summaries[-1] = "Noise"
        return cluster_summaries

    def compute_cluster_centers(self, cluster_labels, projections):
        cluster_label_2_datapoint_indices = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            cluster_label_2_datapoint_indices[label].append(i)

        cluster_centers = {}
        for label in cluster_label_2_datapoint_indices.keys():
            x = np.mean([projections[doc, 0] for doc in cluster_label_2_datapoint_indices[label]])
            y = np.mean([projections[doc, 1] for doc in cluster_label_2_datapoint_indices[label]])
            cluster_centers[label] = (x, y)

        return cluster_centers

    def fit(self, texts,
            load_embeddings_folder=None, save_embeddings_folder=None,  # embed
            projection="umap", projection_params={},
            save_umap_pickle_path=None,  # proj
            classifier_params={"n_neighbors": 10}, save_classifier_dir=None,  # classifier / cluster
            labels=None, cluster_summaries=None,
            ):
        # self.texts = texts

        # embeddings
        if load_embeddings_folder:
            logging.info(f"loading precomputed embeddings from {load_embeddings_folder}")
            embeddings = self.load_embeddings(load_embeddings_folder)

        else:
            logging.info("embedding texts...")
            self.embed_model = SentenceTransformer(self.embed_model_name, device=self.embed_device)
            self.embed_model.max_seq_length = self.embed_max_seq_length
            embeddings = self.embed(texts)

        if save_embeddings_folder:
            self.save_embeddings(save_embeddings_folder, embeddings)
        logging.info("embeddings done")

        # project
        logging.info("projecting ...")
        projections, self.umap_mapper = self.project(embeddings, projection, projection_params)

        if save_umap_pickle_path:
            self.save_umap(umap_pickle_path=save_umap_pickle_path)
        logging.info("projections done")

        # clustering
        if labels is not None:
            logging.info("using cluster labels")
            cluster_labels = labels
        else:
            logging.info("clustering...")
            cluster_labels = self.cluster(projections)
        logging.info("clustering done")

        # logging.info("fitting a classifier")
        # # fit classifier
        # self.classifier = KNeighborsClassifier(**classifier_params).fit(projections, cluster_labels)
        # if save_classifier_pickle_path:
        #     os.makedirs(os.path.dirname(save_classifier_pickle_path), exist_ok=True)
        #     with open(save_classifier_pickle_path, "wb") as f:
        #         pickle.dump(self.classifier, f)
        #     logging.info(f"classifier saved to : {save_classifier_pickle_path}...")

        logging.info("building classifier...")
        self.classifier_faiss_index = self.build_faiss_index(embeddings)
        self.classifier_faiss_index_proj = self.build_faiss_index(projections)
        self.classifier_embeddings = embeddings
        self.classifier_cluster_labels = cluster_labels

        if save_classifier_dir:
            os.makedirs(save_classifier_dir, exist_ok=True)
            faiss_save_path = save_classifier_dir+"/faiss.index"
            faiss.write_index(self.classifier_faiss_index, faiss_save_path)

            faiss_save_path = save_classifier_dir+"/faiss_proj.index"
            faiss.write_index(self.classifier_faiss_index_proj, faiss_save_path)

            embeddings_save_path = save_classifier_dir+"/embeddings.npy"
            with open(embeddings_save_path, "wb") as f:
                np.save(f, self.classifier_embeddings)

            cluster_labels_save_path = save_classifier_dir+"/cluster_labels.pickle"
            with open(cluster_labels_save_path, "wb") as f:
                pickle.dump(self.classifier_cluster_labels, f)

            logging.info(f"classifier saved to : {save_classifier_dir}")
            assert len(cluster_labels) == len(embeddings) == self.classifier_faiss_index.ntotal
            assert len(cluster_labels) == len(embeddings) == self.classifier_faiss_index_proj.ntotal

        logging.info(f"classifier done")


        # compute metrics and analysis
        self.cluster_unique_labels, self.cluster_counts, self.cluster_percentages, self.cluster_entropy = \
            self.compute_cluster_metrics(cluster_labels)

        self.cluster_centers = self.compute_cluster_centers(cluster_labels, projections)

        if cluster_summaries:
            self.cluster_summaries = cluster_summaries
        else:
            self.cluster_summaries = self.create_cluster_summaries(cluster_labels)

        return embeddings, projections, cluster_labels

    def compute_cluster_metrics(self, cluster_labels):
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        percentages = counts / len(cluster_labels)
        ent = float(entropy(percentages))
        return unique_labels, counts, percentages, ent

    def show_elbow(self, save_path, title="", projections=None):
        nbrs = NearestNeighbors(n_neighbors=self.dbscan_min_samples + 1, metric="euclidean").fit(projections)
        dist, ind = nbrs.kneighbors(projections)
        k_dist = np.sort(dist[:, -1])
        plt.clf()
        plt.plot(k_dist)
        plt.title(title + f" nn {self.dbscan_min_samples} + 1")

        plt.xlabel('Distance sorted points')
        plt.ylabel(f'{self.dbscan_min_samples}-Distance')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Elbow saved to: {save_path}")

    def infer_knn(self, embeddings=None, projections=None, top_k=1, batch_size=1000, use_projections=True):
        logging.info(f"Inferring knn... {'(projection based)' if use_projections else ''}")

        import time
        s = time.time()
        queries = projections if use_projections else embeddings

        inferred_labels = []
        for i in tqdm(range(0, len(queries), batch_size), desc="Infering knn"):
            batch = queries[i:i+batch_size]

            if use_projections:
                dist, neighbours = self.classifier_faiss_index_proj.search(batch, top_k)
            else:
                dist, neighbours = self.classifier_faiss_index.search(batch, top_k)

            for i in range(len(batch)):
                labels = [self.classifier_cluster_labels[doc] for doc in neighbours[i]]
                inferred_labels.append(Counter(labels).most_common(1)[0][0])

        print(f"Time (for {len(queries)}): {time.time()-s}")
        return inferred_labels

    def embed(self, texts):

        if self.multigpu:
            pool = self.embed_model.start_multi_process_pool()
            embeddings = self.embed_model.encode_multi_process(
                texts,
                pool=pool,
                batch_size=self.embed_batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
            self.embed_model.stop_multi_process_pool(pool)
        else:
            embeddings = self.embed_model.encode(
                texts,
                batch_size=self.embed_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        return embeddings

    def project(self, embeddings, projection='umap', project_params={}):

        if projection == "umap":
            mapper = UMAP(
                n_components=self.proj_components, metric=self.umap_metric,
                **project_params
            ).fit(embeddings)

            return mapper.embedding_, mapper

        elif projection == "pca":
            mapper = PCA(n_components=self.proj_components, **project_params).fit(embeddings)
            projections = mapper.transform(embeddings)
            return projections, mapper
        else:
            raise ValueError(f"Unknown projection: {projection}")

    def cluster(self, embeddings):
        print(
            f"Using DBSCAN (eps, nim_samples)=({self.dbscan_eps,}, {self.dbscan_min_samples})"
        )

        if self.cluster_method == "dbscan":
            self.clustering = DBSCAN(
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                n_jobs=self.dbscan_n_jobs,
                metric="euclidean"
            ).fit(embeddings)

        elif self.cluster_method == "hdbscan":
            self.clustering = HDBSCAN(
                min_cluster_size=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                n_jobs=self.dbscan_n_jobs,
                metric="euclidean"
            ).fit(embeddings)
        elif self.cluster_method == "gmm":
            self.clustering = GaussianMixture(n_components=self.dbscan_min_samples).fit(embeddings)

            self.clustering.labels_ = self.clustering.predict(embeddings)
            self.aic = self.clustering.aic(embeddings)

        else:
            raise ValueError(f"Unknown cluster method: {self.cluster_method}")
        return self.clustering.labels_

    def build_faiss_index(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    import numpy as np

    def save(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        # with open(f"{folder}/embeddings.npy", "wb") as f:
        #     np.save(f, self.embeddings)

        # faiss.write_index(self.faiss_index, f"{folder}/faiss.index")

        # with open(f"{folder}/projections.npy", "wb") as f:
        #     np.save(f, self.projections)


        # with open(f"{folder}/cluster_labels.npy", "wb") as f:
        #     np.save(f, self.cluster_labels)

        # with open(f"{folder}/texts.json", "w") as f:
        #     json.dump(self.texts, f)

        if self.cluster_summaries is not None:
            with open(f"{folder}/cluster_summaries.json", "w") as f:
                json.dump(self.cluster_summaries, f)

    def load_embeddings(self, folder):
        with open(f"{folder}/embeddings.npy", "rb") as f:
            embeddings = np.load(f)
        return embeddings

    def save_embeddings(self, folder, embeddings):
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(f"{folder}/embeddings.npy", "wb") as f:
            np.save(f, embeddings)

        logging.info(f"saved embeddings to: {folder}")

    def load_umap(self, umap_pickle_path):
        with open(umap_pickle_path, "rb") as f:
            self.umap_mapper = pickle.load(f)

        return self.umap_mapper
    def save_umap(self, umap_pickle_path):
        os.makedirs(os.path.dirname(umap_pickle_path), exist_ok=True)
        with open(umap_pickle_path, "wb") as f:
            pickle.dump(self.umap_mapper, f)
            print(f"Umap saved to: {umap_pickle_path}")

    def load(self, folder):
        if not os.path.exists(folder):
            raise ValueError(f"The folder '{folder}' does not exsit.")

        # with open(f"{folder}/embeddings.npy", "rb") as f:
        #     self.embeddings = np.load(f)

        # self.faiss_index = faiss.read_index(f"{folder}/faiss.index")

        # with open(f"{folder}/projections.npy", "rb") as f:
        #     self.projections = np.load(f)

        # with open(f"{folder}/cluster_labels.npy", "rb") as f:
        #     self.cluster_labels = np.load(f)

        # with open(f"{folder}/texts.json", "r") as f:
        #     self.texts = json.load(f)

        if os.path.exists(f"{folder}/cluster_summaries.json"):
            with open(f"{folder}/cluster_summaries.json", "r") as f:
                self.cluster_summaries = json.load(f)
                keys = list(self.cluster_summaries.keys())
                for key in keys:
                    self.cluster_summaries[int(key)] = self.cluster_summaries.pop(key)

        # self.cluster_label_2_datapoint_indices = defaultdict(list)
        # for i, label in enumerate(self.cluster_labels):
        #     self.cluster_label_2_datapoint_indices[label].append(i)

        # self.cluster_centers = {}
        # for label in self.cluster_label_2_datapoint_indices.keys():
        #     x = np.mean([self.projections[doc, 0] for doc in self.cluster_label_2_datapoint_indices[label]])
        #     y = np.mean([self.projections[doc, 1] for doc in self.cluster_label_2_datapoint_indices[label]])
        #     self.cluster_centers[label] = (x, y)

    def show(self, projections, cluster_labels, save_path=None, title=None):
        df = pd.DataFrame(
            data={
                "X": projections[:, 0],
                "Y": projections[:, 1],
                "labels": cluster_labels,
                # "content_display": [
                #     textwrap.fill(txt[:1024], 64) for txt in self.texts
                # ],
            }
        )

        self._show_mpl(df, title=title)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Plot saved to: {save_path}")

    def make_ellipses(self, gmm, ax, lab_2_col):
        import matplotlib as mpl
        for n in range(gmm.n_components):
            color = lab_2_col[n]
            if gmm.covariance_type == "full":
                covariances = gmm.covariances_[n][:2, :2]
            elif gmm.covariance_type == "tied":
                covariances = gmm.covariances_[:2, :2]
            elif gmm.covariance_type == "diag":
                covariances = np.diag(gmm.covariances_[n][:2])
            elif gmm.covariance_type == "spherical":
                covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi  # convert to degrees
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            ell = mpl.patches.Ellipse(
                gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color, alpha=0.3
            )
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
            ax.set_aspect("equal", "datalim")

    def show_cluster_histograms(self, save_path, title=None):

        # Plot histogram
        plt.figure(figsize=(10, 5))
        plt.bar(self.cluster_unique_labels, self.cluster_counts)
        plt.title(f"{title} Cluster entropy {self.cluster_entropy}")
        plt.xlabel('Cluster Label')
        plt.ylabel('Number of Points')

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Histogram saved to: {save_path}")

        return {
            'unique_clusters': self.cluster_unique_labels,
            'cluster_sizes': self.cluster_counts,
            'cluster_entropy': self.cluster_entropy
        }

    def _show_mpl(self, df, title=None):
        fig, ax = plt.subplots(figsize=(12, 8), dpi=300)

        unique_labels = df["labels"].unique()
        if len(unique_labels) <= 4:
            lab_2_col = dict(zip(unique_labels, ["red", "lime", "aqua", "magenta"]))
        else:
            lab_2_col = {l: f"C{(l % 9) + 1}" if l != -1 else "C0" for l in unique_labels}

        df["color"] = df["labels"].apply(lambda x: lab_2_col[x])

        df.plot(
            kind="scatter",
            x="X",
            y="Y",
            s=0.75,
            alpha=0.8,
            linewidth=0,
            color=df["color"],
            ax=ax,
            colorbar=False,
        )


        # Create legend handles manually
        from matplotlib.lines import Line2D
        unique_labels = df["labels"].unique()

        if self.cluster_summaries == {0: "senator_tweets", 1: "reddit_submissions", 2: "twitter_100M", 3: "webis_reddit"}:
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=lab_2_col[label] if label != -1 else "C0",
                       label=f'{self.cluster_summaries[label]}' if label != -1 else 'Noise',
                       markersize=8)
                for label in sorted(unique_labels)
            ]

            # Add the legend
            ax.legend(handles=legend_elements, loc='best', frameon=True)

        for label in unique_labels:
            if label == -1:
                continue
            summary = self.cluster_summaries.get(label, "")
            position = self.cluster_centers[label]
            t = ax.text(
                position[0],
                position[1],
                summary,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=4,
            )
            t.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=0, boxstyle='square,pad=0.1'))

        if self.cluster_entropy is not None:
            ax.set_title(f'{title} Cluster entropy: {self.cluster_entropy}')

        if self.aic is not None:
            ax.set_title(f'{title} AIC: {self.aic}')

        ax.set_axis_off()

        return ax


    def show_separate(self, projections, cluster_labels, save_path=None):
        """Show each dataset in a separate plot while maintaining joint UMAP computation."""
        self._show_separate_mpl(projections, cluster_labels)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Separate plot saved to: {save_path}")

    def _show_separate_mpl(self, projections, cluster_labels):
        # Calculate grid dimensions
        n_datasets = len(set(cluster_labels))
        if -1 in set(cluster_labels):  # Exclude noise label
            n_datasets -= 1

        n_rows = (n_datasets + 1) // 2  # Round up division
        n_cols = min(2, n_datasets)  # Use 2 columns or less

        # Create figure with appropriate size
        fig = plt.figure(figsize=(12, 6 * n_rows), dpi=300)

        # Create a DataFrame with all data
        df = pd.DataFrame(
            data={
                "X": projections[:, 0],
                "Y": projections[:, 1],
                "labels": cluster_labels,
                # "content_display": [
                #     textwrap.fill(txt[:1024], 64) for txt in self.texts
                # ],
            }
        )

        unique_labels = sorted(set(cluster_labels))
        if len(unique_labels) == 4:
            lab_2_col = dict(zip(unique_labels, ["red", "lime", "aqua", "magenta"]))
        else:
            lab_2_col = {l: f"C{(l % 9) + 1}" if l != -1 else "C0" for l in unique_labels}

        df["color"] = df["labels"].apply(lambda x: lab_2_col[x])

        if -1 in unique_labels:  # Remove noise label if present
            unique_labels.remove(-1)

        # only keep the 10 most common labels
        unique_labels = list(zip(*Counter(cluster_labels).most_common(n=10)))[0]

        for idx, label in enumerate(unique_labels, 1):
            ax = fig.add_subplot(n_rows, n_cols, idx)

            # Plot points for current dataset
            mask_current = df['labels'] == label
            current_points = df[mask_current]
            ax.scatter(
                current_points['X'],
                current_points['Y'],
                s=0.75,
                alpha=0.8,
                linewidth=0,
                # color=f'C{(label % 9) + 1}',
                color=lab_2_col[label],
                label=self.cluster_summaries[label]
            )

            # Plot other points in gray and with lower alpha
            mask_others = df['labels'] != label
            other_points = df[mask_others]
            ax.scatter(
                other_points['X'],
                other_points['Y'],
                s=0.75,
                alpha=0.1,
                linewidth=0,
                color='gray'
            )

            # Add cluster label
            center = self.cluster_centers[label]
            summary = self.cluster_summaries[label]
            t = ax.text(
                center[0],
                center[1],
                summary,
                horizontalalignment='center',
                verticalalignment='center',
                fontsize=8,
            )
            t.set_bbox(dict(facecolor='white', alpha=0.9, linewidth=0, boxstyle='square,pad=0.1'))

            if self.cluster_entropy is not None:
                ax.set_title(f'Dataset: {summary}, Cluster entropy: {self.cluster_entropy}')
            else:
                ax.set_title(f'Dataset: {summary}')
            ax.set_axis_off()

        plt.tight_layout()

    def show_gmm_comparison(self, gmm_results, save_path):
        """Visualize entropy comparisons between datasets."""
        import matplotlib.pyplot as plt
        import pandas as pd

        # Convert results to DataFrame for easier plotting
        df_gmm_res = pd.DataFrame(gmm_results).T

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df_gmm_res.plot(kind='bar', ax=ax)

        # Get the data range
        y_min = df_gmm_res.values.min()
        y_max = df_gmm_res.values.max()
        y_range = y_max - y_min

        # Set y-axis limits with padding
        padding = 0.05 * y_range  # 5% padding
        ax.set_ylim(y_min - padding, y_max + padding)


        plt.title('Gmm Comparison Across Datasets')
        plt.xlabel('Dataset')
        plt.ylabel('Gmm fit loss')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def compute_entropies(self, embeddings, projections):
        print("Computing entropies")
        """Compute various entropy measures for each dataset in the clustering."""

        # Initialize results dictionary
        entropy_results = {}

        unique_labels = sorted(set(cluster_labels))
        if -1 in unique_labels:  # Remove noise label if present
            unique_labels.remove(-1)

        for label in unique_labels:
            # Get embeddings for current dataset
            mask_current = np.array(cluster_labels) == label
            current_embeddings = embeddings[mask_current]
            current_projections = projections[mask_current]

            dataset_entropies = {}

            # # 1. Embedding Space Entropy
            # print("Embedding space")
            # # Compute pairwise distances in embedding space
            # distances_embed = pdist(current_embeddings, metric='cosine')
            # # Normalize distances to create probability distribution
            # prob_dist_embed = distances_embed / distances_embed.sum()
            # embed_entropy = entropy(prob_dist_embed)
            # dataset_entropies['embedding_space_entropy'] = float(embed_entropy)

            # # 2. UMAP Projection Entropy
            # print("UMAP space")
            # # Compute pairwise distances in UMAP space
            distances_umap = pdist(current_projections, metric='euclidean')
            # prob_dist_umap = distances_umap / distances_umap.sum()
            # umap_entropy = entropy(prob_dist_umap)
            # dataset_entropies['umap_projection_entropy'] = float(umap_entropy)

            # # 3. Local Density Entropy
            # print("Local Density")
            # # Compute local density using Gaussian kernel
            # sigma = 0.1  # bandwidth parameter
            # dist_matrix = squareform(distances_umap)
            # densities = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2)).mean(axis=1)
            # prob_dist_density = densities / densities.sum()
            # density_entropy = entropy(prob_dist_density)
            # dataset_entropies['local_density_entropy'] = float(density_entropy)
            #
            # 4. Nearest Neighbor Entropy
            print("NN")
            k = min(20, len(current_projections) - 1)  # number of neighbors
            # Get k nearest neighbors for each point
            dist_matrix = squareform(distances_umap)
            neighbor_distances = np.sort(dist_matrix, axis=1)[:, 1:k + 1]  # exclude self
            # Compute entropy of neighbor distance distribution
            prob_dist_nn = neighbor_distances.flatten()
            prob_dist_nn = prob_dist_nn / prob_dist_nn.sum()
            nn_entropy = entropy(prob_dist_nn)
            dataset_entropies['nearest_neighbor_entropy'] = float(nn_entropy)

            # Store results for current dataset
            entropy_results[self.cluster_summaries[label]] = dataset_entropies

        return entropy_results

    def show_entropy_comparison(self, entropy_results, save_path):
        """Visualize entropy comparisons between datasets."""
        import matplotlib.pyplot as plt
        import pandas as pd

        # Convert results to DataFrame for easier plotting
        df_entropy = pd.DataFrame(entropy_results).T

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        df_entropy.plot(kind='bar', ax=ax)

        # Get the data range
        y_min = df_entropy.values.min()
        y_max = df_entropy.values.max()
        y_range = y_max - y_min

        # Set y-axis limits with padding
        padding = 0.05 * y_range  # 5% padding
        ax.set_ylim(y_min - padding, y_max + padding)

        plt.title('Entropy Measures Comparison Across Datasets')
        plt.xlabel('Dataset')
        plt.ylabel('Entropy Value')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def print_entropy_analysis(self, entropy_results):
        """Print detailed analysis of entropy results."""
        print("Entropy Analysis Report")
        print("=" * 50)

        measures = list(next(iter(entropy_results.values())).keys())
        datasets = list(entropy_results.keys())

        # Print summary for each measure
        for measure in measures:
            print(f"\n{measure.replace('_', ' ').title()}:")
            print("-" * 30)

            values = {dataset: results[measure]
                      for dataset, results in entropy_results.items()}

            # Sort datasets by this measure
            sorted_datasets = sorted(values.items(), key=lambda x: x[1], reverse=True)

            # Print ranking
            for i, (dataset, value) in enumerate(sorted_datasets, 1):
                print(f"{i}. {dataset}: {value:.4f}")

            # Calculate and print statistics
            values_array = np.array(list(values.values()))
            print(f"\nMean: {values_array.mean():.4f}")
            print(f"Std: {values_array.std():.4f}")
            print(f"Range: {values_array.max() - values_array.min():.4f}")

from datasets import load_from_disk, concatenate_datasets, DatasetDict
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
        "webis",
        "webis_ow_cluster",
        "webis_na_cluster",
        "webis_sa_cluster",
        "webis_j_cluster",
        "webis_h_cluster",
        "senator_submissions_merged",
    ]
    parser = argparse.ArgumentParser(description="Set dataset name from command-line arguments.")
    parser.add_argument("--name", type=str, default="joint", choices=allowed_names)
    args = parser.parse_args()
    name = args.name

    per_d_sample = 90_000
    # per_d_sample = 9_000

    if name == "joint":
        labels = [0] * per_d_sample + [1] * per_d_sample + [2] * per_d_sample + [3] * per_d_sample
        cluster_summaries = {0: "senator_tweets", 1: "reddit_submissions", 2: "twitter_100M", 3: "webis_reddit"}
        ds_ = [
            load_from_disk("data/senator_tweets/prepared-senator-tweets-qualities"),
            load_from_disk("data/reddit_submissions/prepared-reddit-submissions-dataset-with-qualities-200-minus-20-plus"),
            load_from_disk("data/twitter_100m/prepared-100m-tweets-english-qualities-cl-before-gpt3-200-minus-20-plus"),
            load_from_disk("data/webis/prepared-cleaned-200-minus-20-plus-corpus-webis-tldr-17")
        ]
    else:
        name_2_path = {
            "senators_tweets": "data/senator_tweets/prepared-senator-tweets-qualities",
            "reddit_submissions": "data/reddit_submissions/prepared-reddit-submissions-dataset-with-qualities-200-minus-20-plus",
            "100m_tweets": "data/twitter_100m/prepared-100m-tweets-english-qualities-cl-before-gpt3-200-minus-20-plus",
            "webis": "data/webis/prepared-cleaned-200-minus-20-plus-corpus-webis-tldr-17",
            "webis_ow_cluster": "data/webis/prepared-OW-cluster-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17",
            "webis_na_cluster": "data/webis/prepared-NA-cluster-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17",
            "webis_sa_cluster": "data/webis/prepared-SA-cluster-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17",
            "webis_j_cluster": "data/webis/prepared-J-cluster-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17",
            "webis_h_cluster": "data/webis/prepared-H-cluster-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17",
            "senator_submissions_merged": "./data/senator_submissions_merged/prepared-senator-submissions-merged"
        }
        ds_ = [load_from_disk(name_2_path[name])]
        print("Loading dataset from : ", name_2_path[name])
        # cluster_summaries = {0: name}
        # labels = labels[:len(ds_)*per_d_sample]
        labels = None; cluster_summaries = None  # cluster

    ds = []
    full_ds = []
    for d in ds_:
        if type(d) == DatasetDict:
            assert d.keys() == {'train', 'test'}
            d = concatenate_datasets([d['train'], d['test']])

        cols = d.column_names
        cols.remove("text")
        if 'subreddit' in cols:
            cols.remove("subreddit")
        d = d.remove_columns(cols)
        full_ds.append(d)
        ds.append(d.shuffle().select(range(per_d_sample)))

    d = concatenate_datasets(ds)
    full_d = concatenate_datasets(full_ds)
    print("Dataset:", full_d)

    # folder = f"./viz_results/clustering/stella_5/{name}/"
    folder = f"./viz_results/clustering/stella_6_proj/{name}/"

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
        # (0.3, 100),
        (0.2, 50),
        # (0.2, 5),
    ]
    cluster_method = "dbscan"

    for eps, min_samples in cluster_params:
        for i, proj_par in enumerate(project_params_list):

            cc = ClusterClassifier(
                embed_device="cuda", multigpu=True,
                cluster_method=cluster_method, dbscan_eps=eps, dbscan_min_samples=min_samples,
            )

            texts = d['text']
            _, projections, cluster_labels = cc.fit(
                texts,
                load_embeddings_folder=None if i == 0 else folder, save_embeddings_folder=folder,
                # load_embeddings_folder=folder.replace("_proj", ""), save_embeddings_folder=folder,
                projection="umap", projection_params=proj_par,
                save_umap_pickle_path=folder + f"umap/umap_kn_{proj_par['n_neighbors']}_md_{proj_par['min_dist']}/{cluster_method}/eps_{eps}_min_samples_{min_samples}.pickle",
                # classifier_params={"n_neighbors": 10},
                save_classifier_dir=folder + f"classifier/umap_kn_{proj_par['n_neighbors']}_md_{proj_par['min_dist']}/{cluster_method}/eps_{eps}_min_samples_{min_samples}/",
                labels=labels, cluster_summaries=cluster_summaries,
            )
            results_folder = folder + f"results/umap_kn_{proj_par['n_neighbors']}_md_{proj_par['min_dist']}/{cluster_method}/eps_{eps}_min_samples_{min_samples}/"
            save_results(results_folder, texts, projections, cluster_labels)

            clustering_results = cc.show_cluster_histograms(save_path=folder + f"histogram/umap_kn_{proj_par['n_neighbors']}_md_{proj_par['min_dist']}/{cluster_method}/eps_{eps}_min_samples_{min_samples}.png", title=name)

            cc.show_elbow(
                projections=projections,
                save_path=folder+f"elbow/umap_kn_{proj_par['n_neighbors']}_md_{proj_par['min_dist']}/{cluster_method}/eps_{eps}_min_samples_{min_samples}.png",
                title=name
            )
            cc.show(
                projections=projections,
                save_path=folder+f"umap_kn_{proj_par['n_neighbors']}_md_{proj_par['min_dist']}/{cluster_method}/eps_{eps}_min_samples_{min_samples}.png",
                title=name,
                cluster_labels=cluster_labels
            )

            # now for the full dataset
            ##########################
            full_dataset_folder = folder + "full_dataset/"
            all_texts = full_d['text']
            _, projections, cluster_labels = cc.transform(
                all_texts,
                load_embeddings_folder=None if i == 0 else full_dataset_folder, save_embeddings_folder=full_dataset_folder,
                # load_embeddings_folder=full_dataset_folder.replace("_proj", ""), save_embeddings_folder=full_dataset_folder,
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
            save_results(results_folder, all_texts, projections, cluster_labels)
