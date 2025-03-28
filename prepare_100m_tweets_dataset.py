import datasets
from datasets import load_dataset
import re
from datetime import datetime
from eval_utils import llama_quality, llama_is_english
from dataset_utils import overwrite_to_disk
def remove_links(batch):
    return {"text": [re.sub(r'http\S+', '', t).rstrip() for t in batch['text']]}

def remove_trailling_hashtags(batch):
    return {"text": [re.sub(r"(?:\s*#\w+)+$", "", t) for t in batch['text']]}


def filter_posts_by_size(dataset, n_min, n_max):

    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    dataset = dataset.map(
        lambda batch: {"n_tokens": [len(tokenizer.encode(t)) for t in batch['text']]},
        batched=True, desc="Adding number of tokens"
    )

    dataset = dataset.filter(
        lambda batch: [n_min < n_t < n_max for n_t in batch['n_tokens']],
        batched=True, desc=f"Filtering too long (>{n_max}) and too shorts (<{n_min}) posts."
    )

    return dataset

def is_before_chatgpt_release(date_str):
    chatgpt_release_date = datetime(2022, 11, 30)
    input_date = datetime.fromisoformat(date_str).replace(tzinfo=None)
    return input_date < chatgpt_release_date

def is_before_gpt3_release(date_str):
    gpt3_release_date = datetime(2020, 6, 1)
    input_date = datetime.fromisoformat(date_str).replace(tzinfo=None)
    return input_date < gpt3_release_date

n_min = 20
n_max = 200

# dataset = load_dataset("enryu43/twitter100m_tweets", split='train')
# dataset = dataset.rename_column("tweet", "text")
# dataset = dataset.map(remove_links, batched=True, desc="Removing links", load_from_cache_file=True, num_proc=10)
# dataset = dataset.map(remove_trailling_hashtags, batched=True, desc="Removing links", load_from_cache_file=True, num_proc=10)
#
# dataset = filter_posts_by_size(dataset, n_min=n_min, n_max=n_max)
#
#
# file_path = f"./data/twitter_100m/prepared-100m-tweets-{n_max}-minus-{n_min}-plus"
# dataset.save_to_disk(file_path)
# print(f"Saved to: {file_path}")
#
# dataset_before_gpt3 = dataset.filter(
#     lambda batch: [is_before_gpt3_release(dt) for dt in batch['date']],
#     batched=True, desc=f"Filtering before GPT-3", num_proc=20
# )
# file_path = f"./data/twitter_100m/prepared-100m-tweets-before-gpt3-{n_max}-minus-{n_min}-plus"
# dataset_before_gpt3.save_to_disk(file_path)
# print(f"Saved to: {file_path}")
#
# dataset_before_chatgpt = dataset.filter(
#     lambda batch: [is_before_chatgpt_release(dt) for dt in batch['date']],
#     batched=True, desc=f"Filtering before ChatGPT", num_proc=32
# )
# file_path = f"./data/twitter_100m/prepared-100m-tweets-before-chatgpt-{n_max}-minus-{n_min}-plus"
# dataset_before_chatgpt.save_to_disk(file_path)
# print(f"Saved to: {file_path}")


# 55.1M
# dataset = datasets.load_from_disk(f"./data/twitter_100m/prepared-100m-tweets-{n_max}-minus-{n_min}-plus")
# 40.1M
# dataset_before_gpt3 = dataset.load_from_disk(f"./data/twitter_100m/prepared-100m-tweets-before-chatgpt-{n_max}-minus-{n_min}-plus")
# 16.8M
# dataset_before_gpt3 = datasets.load_from_disk(f"./data/twitter_100m/prepared-100m-tweets-before-gpt3-{n_max}-minus-{n_min}-plus")
#
# dataset_before_gpt3 = dataset_before_gpt3.shuffle().select(range(4_000_000))
#
# dataset_english = dataset_before_gpt3.filter(
#     lambda batch: llama_is_english(batch['text']),
#     batched=True, desc=f"Filtering english text", num_proc=32, batch_size=50
# )
#
# file_path = f"./data/twitter_100m/prepared-100m-tweets-english-before-gpt3-{n_max}-minus-{n_min}-plus"
# dataset_english.save_to_disk(file_path)
# print(f"Saved to: {file_path}")


# dataset_english = datasets.load_from_disk(f"./data/twitter_100m/prepared-100m-tweets-english-before-gpt3-{n_max}-minus-{n_min}-plus")
#
# dataset_english = dataset_english.map(
#     lambda examples: {"llama_quality": llama_quality(examples["text"])},
#     batched=True, desc="Computing quality", batch_size=10, num_proc=30
# )
#
# file_path = f"./data/twitter_100m/prepared-100m-tweets-english-qualities-before-gpt3-{n_max}-minus-{n_min}-plus"
# dataset_english.save_to_disk(file_path)
# print(f"Saved to: {file_path}")


# # step: 3
# dataset = datasets.load_from_disk(f"./data/twitter_100m/prepared-100m-tweets-english-qualities-before-gpt3-{n_max}-minus-{n_min}-plus")
#
# dataset_hq = dataset.filter(lambda ex: ex['llama_quality'] == 2)
# print(dataset_hq)
# file_path = f"./data/twitter_100m/prepared-100m-tweets-english-high-quality-before-gpt3-{n_max}-minus-{n_min}-plus"
# dataset_hq.save_to_disk(file_path)
# print(f"Saved to: {file_path}")
#
# dataset_mq = dataset.filter(lambda ex: ex['llama_quality'] == 1)
# print(dataset_mq)
# file_path = f"./data/twitter_100m/prepared-100m-tweets-english-mid-quality-before-gpt3-{n_max}-minus-{n_min}-plus"
# dataset_mq.save_to_disk(file_path)
# print(f"Saved to: {file_path}")

# # std
# file_path = f"./data/twitter_100m/prepared-100m-tweets-english-qualities-before-gpt3-{n_max}-minus-{n_min}-plus"
# std = datasets.load_from_disk(file_path)
# std = std.filter(lambda batch: [t.count("#") < 3 and t.count("@") < 3 for t in batch['text']], batched=True)
# file_path = f"./data/twitter_100m/prepared-100m-tweets-english-qualities-cl-before-gpt3-{n_max}-minus-{n_min}-plus"
# std.save_to_disk(file_path)
#
# # hq
# file_path = f"./data/twitter_100m/prepared-100m-tweets-english-high-quality-before-gpt3-{n_max}-minus-{n_min}-plus"
# hq = datasets.load_from_disk(file_path)
# hq = hq.filter(lambda batch: [t.count("#") < 3 and t.count("@") < 3 for t in batch['text']], batched=True)
# file_path = f"./data/twitter_100m/prepared-100m-tweets-english-high-quality-cl-before-gpt3-{n_max}-minus-{n_min}-plus"
# hq.save_to_disk(file_path)
#
# # lq
# file_path = f"./data/twitter_100m/prepared-100m-tweets-english-mid-quality-before-gpt3-{n_max}-minus-{n_min}-plus"
# mq = datasets.load_from_disk(file_path)
# mq = mq.filter(lambda batch: [t.count("#") < 3 and t.count("@") < 3 for t in batch['text']], batched=True)
# file_path = f"./data/twitter_100m/prepared-100m-tweets-english-mid-quality-cl-before-gpt3-{n_max}-minus-{n_min}-plus"
# mq.save_to_disk(file_path)


# Focused
import torch
import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#
# def create_focused_dataset(dataset, figure_path=None, subset_size=100_000):
#
#     def largest_proper_divisor(n):
#         for i in range(n // 2, 0, -1):
#             if n % i == 0:
#                 return i
#
#     dataset_len = len(dataset)
#     batch_size = largest_proper_divisor(dataset_len)  # divisor
#     print(f"Batch size: {batch_size}")
#     assert dataset_len % batch_size == 0
#
#     centers = []
#     sum = []
#     for i in tqdm(range(dataset_len // batch_size), desc="Finding the centers"):
#         embs_batch = torch.tensor(dataset.select(range(i, i + batch_size))['stella_embeddings'])
#         centers.append(embs_batch.mean(dim=0))
#         sum.append(embs_batch.sum(dim=0))
#
#     center = torch.vstack(centers).mean(dim=0)
#
#     print("computing similarities")
#     cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#     similarities = []
#     for i in tqdm(range(dataset_len // batch_size), desc="Computing similarities"):
#         embs_batch = torch.tensor(dataset.select(range(i, i + batch_size))['stella_embeddings'])
#         similarities_batch = cos(center, embs_batch)
#         similarities.extend(similarities_batch)
#
#     similarities = torch.tensor(similarities)
#
#     focused_indices = similarities.topk(subset_size).indices
#     focus_dataset = dataset.select(focused_indices)
#
#     if figure_path:
#         print("plotting")
#
#         # take 1/100 sample from each dataset
#         embs_draw = np.array(dataset.select(range(len(dataset) // 100))['stella_embeddings'], dtype=np.float16)
#         embs_focus = np.array(focus_dataset.select(range(len(focus_dataset) // 100))['stella_embeddings'],
#                               dtype=np.float16)
#
#         X = np.vstack([embs_draw, embs_focus])
#
#         ss_X = StandardScaler().fit_transform(X)
#         pca_X = PCA(n_components=2).fit_transform(ss_X)
#
#         c = [0] * len(embs_draw) + [1] * len(embs_focus)
#         plt.gcf().clear()
#         plt.scatter(pca_X[:, 0], pca_X[:, 1], c=c, label="Data points")
#         plt.savefig(figure_path)
#         print(f"Saved to: {figure_path}")
#
#     return focus_dataset

###################
# add embeddings
###################
# if __name__ == "__main__":
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     from eval_utils import StellaEmbedder
#     file_path = f"data/twitter_100m/prepared-100m-tweets-english-qualities-cl-before-gpt3-200-minus-20-plus"
#     dataset = load_from_disk(file_path)
#     print(dataset)
#     stella_embedder = StellaEmbedder(multigpu=True)
#     dataset = stella_embedder.add_embeddings_multigpu(dataset, batch_size=2048)
#     overwrite_to_disk(dataset, file_path)
# exit()


###################
# add qualities
###################
# from eval_utils import llama_quality_scale
# file_path = f"data/twitter_100m/prepared-100m-tweets-english-qualities-cl-before-gpt3-200-minus-20-plus"
# dataset = load_from_disk(file_path)
#
#
# dataset = dataset.map(
#     lambda examples: {"llama_quality_scale": llama_quality_scale(examples["text"])},
#     batched=True, desc="Computing quality scale", batch_size=10, num_proc=60, load_from_cache_file=False
# )
# overwrite_to_disk(dataset, file_path)

###################
# Quality datasets
###################
# file_path = f"data/twitter_100m/prepared-100m-tweets-english-qualities-cl-before-gpt3-200-minus-20-plus"
# dataset = datasets.load_from_disk(file_path)
# dataset = dataset.filter(lambda ex: ex['llama_quality_scale'] is not None, num_proc=64, load_from_cache_file=False)
#
# qualities = np.array(dataset['llama_quality_scale'])
# print("Separating")
# for q in [20, 40, 60, 80]:
#     d = dataset.select(np.where(qualities == q)[0])
#     print("Q:", q)

####################
# Length datasets
####################
file_path = f"data/twitter_100m/prepared-100m-tweets-english-qualities-cl-before-gpt3-200-minus-20-plus"
ds = []
for q_str in ["_llama_scale_40", "_llama_scale_60", "_llama_scale_80"]:  # no Q20 avoid rock bottom
    ds.append(datasets.load_from_disk(file_path+q_str))
dataset = datasets.concatenate_datasets(ds)

dataset = dataset.map(
    lambda examples: {"text_len": [len(t) for t in examples["text"]]},
    batched=True, desc="Adding len", batch_size=10, num_proc=60, load_from_cache_file=False
)
lengths = np.array(dataset['text_len'])

sort_indices = np.argsort(lengths)
chunk_size = len(sort_indices) // 3

short_d = dataset.select(sort_indices[:chunk_size])
medium_d = dataset.select(sort_indices[chunk_size:2 * chunk_size])
long_d = dataset.select(sort_indices[2 * chunk_size:])

print("Short: ", np.mean(short_d['text_len']))
print("Medium: ", np.mean(medium_d['text_len']))
print("Long: ", np.mean(long_d['text_len']))

short_d.save_to_disk(file_path + "_short")
medium_d.save_to_disk(file_path + "_medium")
long_d.save_to_disk(file_path + "_long")
exit()

# print("Sorting")
# sort_indices = np.argsort(dataset['llama_quality_scale'])
# tercile_size = int(np.floor(len(dataset) // 3))
#
# print("Creating datasets")
# dataset_lq = dataset.select(sort_indices[:tercile_size]).shuffle()
# print("LQ len: {}, Q: {}".format(len(dataset_lq['llama_quality_scale']), np.mean(dataset_lq['llama_quality_scale'])))
#
# dataset_mq = dataset.select(sort_indices[tercile_size: tercile_size * 2]).shuffle()
# print("MQ len: {}, Q: {}".format(len(dataset_mq['llama_quality_scale']), np.mean(dataset_mq['llama_quality_scale'])))
#
# dataset_hq = dataset.select(sort_indices[tercile_size * 2: tercile_size * 3]).shuffle()
# print("HQ len: {}, Q: {}".format(len(dataset_hq['llama_quality_scale']), np.mean(dataset_hq['llama_quality_scale'])))
#
# dataset_lq.save_to_disk(file_path + "_llama_scale_lq")
# dataset_mq.save_to_disk(file_path + "_llama_scale_mq")
# dataset_hq.save_to_disk(file_path + "_llama_scale_hq")
exit()




# focused_dataset = create_focused_dataset(dataset, subset_size=100_000, figure_path="viz_results/clustering/100M_tweets_focused.png")
# file_path = f"./data/twitter_100m/prepared-100m-tweets-english-focused-qualities-cl-before-gpt3-{n_max}-minus-{n_min}-plus"
# focused_dataset.save_to_disk(file_path)
