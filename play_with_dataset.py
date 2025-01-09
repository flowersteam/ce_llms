import datasets
from pathlib import Path
import numpy as np
from collections import Counter

import glob
# from eval_utils import *

from dataset_utils import *
# from visualization_utils import *

from datasets import load_dataset

# dataset_name = "webis_reddit"
dataset_name = "reddit_submissions"
# dataset_name = "100m_tweets"
# dataset_name = "senator_tweets"

std = load_human_dataset(
    dataset_name=dataset_name,
    split="all",
    load_n=500,
    dataset_type="standard"
)
hq = load_human_dataset(
    dataset_name=dataset_name,
    split="all",
    load_n=500,
    dataset_type="hq"
)
mq = load_human_dataset(
    dataset_name=dataset_name,
    split="all",
    load_n=500,
    dataset_type="mq"
)
ld = load_human_dataset(
    dataset_name=dataset_name,
    split="all",
    load_n=500,
    dataset_type="ld"
)
from eval_utils import StellaEmbedder, compute_cos_diveristy, llama_quality
stella_embedder = StellaEmbedder()

dataset_names = ["std", "hq", "mq", "ld"]
datasets = [std, hq, mq, ld]

divs = []
qs = []

for d_name, d in zip(dataset_names, datasets):
    d = stella_embedder.add_embeddings(d, batch_size=1024)
    embs = np.array(d[f'stella_embeddings'])
    divs.append(compute_cos_diveristy(embs))

    if "llama_quality" in d.column_names:
        qs.append(np.mean(d['llama_quality']))
    else:
        # qs.append(np.nan)
        qs.append(np.mean(llama_quality(d['text'])))

for d_name, d, q in zip(dataset_names, divs, qs):
    print(f"{d_name} - {d} - {q}")


exit()
