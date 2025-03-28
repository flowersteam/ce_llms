import datasets
import re
import numpy as np
import json
from sklearn.cluster import KMeans, MiniBatchKMeans


# import logging
# from sentence_transformers import LoggingHandler, SentenceTransformer
#
# logging.basicConfig(
#     format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
# )


# # CLEAN BAD SUBREDDITS
############################
# file_path = '/home/flowers-user/Documents/projects/SocialLLM/corpus-webis-tldr-17.json'
#
# with open('contexts/subreddits_save.json') as f:
#     subreddits_safe = json.load(f)
#
# tokenizer = tiktoken.get_encoding("cl100k_base")
#
#
# dataset = []
# with open(file_path, 'r') as file:
#     for line_i, line in enumerate(file):
#         if line_i % 50_000 == 0:
#             print("Line i: ", line_i)
#
#         entry = json.loads(line)
#
#         if "subreddit" not in entry:
#             continue
#
#         if subreddits_safe.get(entry['subreddit'], True):
#             entry["n_tokens"] = len(tokenizer.encode(entry["content"]))
#             dataset.append(entry)
#
# # save
# clear_file_path = 'data/webis_reddit/clear-corpus-webis-tldr-17.json'
# print("Clear dataset len:", len(dataset))
#
# with open(clear_file_path, 'w') as outfile:
#     for entry_i, entry in enumerate(dataset):
#         if entry_i % 100_000 == 0:
#             print("Entry i: ", entry_i)
#
#         json.dump(entry, outfile)
#         outfile.write('\n')
#
# print(f"Saved to {clear_file_path}")


# # FILTER LONG POSTS
############################


def prepare_dataset(dataset):
    def merge_title_and_text(title, text):
        post = ""

        if title is not None:
            post += title

        if title is not None and text is not None:
            post += "\n\n"

        if text is not None:
            post += text

        return post.strip()

    dataset = dataset.map(
        lambda examples: {
            "text": [merge_title_and_text(ti, te) for ti, te in zip(examples["title"], examples["normalizedBody"])]
        },
        remove_columns=["title", "body", "normalizedBody", "n_tokens", "content", "content_len", "summary", "summary_len"], # there was for body only
        batched=True, desc="Merging title and body"
    )
    dataset = dataset.map(lambda examples: {"text_len": [len(t) for t in examples["text"]]}, batched=True, desc="Adding text_len")

    # # clean dataset from links
    # dataset = dataset.map(lambda examples: {"text": [re.sub(r'http\S+', '', t).rstrip() for t in examples['text']]}, batched=True, desc="Removing links")
    assert dataset['text'] == [t.rstrip() for t in dataset['text']]

    return dataset



def remove_tldrs(dataset):
    # remove tldrs and everything after
    import re

    def find_tldr(text):
        match = re.search(r'(?:stl|tl|tld|tll)[\s~.|"‘&’\';:_/\-\\,]*(?:sdr|dr)', text, re.IGNORECASE)
        return match.group(0) if match else None


    def remove_tldr(batch):
        tldr_matches = [find_tldr(t) for t in batch['text']]
        return {"text": [t.split(tldr_m)[0] if tldr_m else t for tldr_m, t in zip(tldr_matches, batch['text'])]}


    dataset = dataset.map(remove_tldr, batched=True, desc="Removing tldrs.")
    return dataset


def filter_posts_by_size(dataset, n_min, n_max):

    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    dataset = dataset.map(
        lambda batch: {"n_tokens": [len(tokenizer.encode(t)) for t in batch['text']]},
        batched=True, desc="Adding number of tokens"
    )

    dataset = dataset.filter(
        lambda batch: [n_min < n_t < n_max for n_t in batch['n_tokens']],
        batched=True, desc="Filtering too long (>{n_max}) and too shorts (<{n_min}) posts."
    )

    return dataset


def get_unique_indices(d):
    unique_text_indices = {}

    for i, text in enumerate(d['text']):
        if text not in unique_text_indices:
            unique_text_indices[text] = i  # Store the index of the first occurrence

    unique_indices = list(unique_text_indices.values())
    return unique_indices

if __name__ == '__main__':
    # # file_path = 'data/webis_reddit/clear-corpus-webis-tldr-17.json'
    # file_path = f'data/webis_reddit/350-minus-20-plus-clear-corpus-webis-tldr-17.json'
    # dataset_hf = datasets.load_dataset("json", data_files=[file_path])
    #
    # dataset_hf = prepare_dataset(dataset_hf)
    # dataset_hf = remove_tldrs(dataset_hf)
    #
    n_min = 20
    n_max = 200
    # dataset_hf = filter_posts_by_size(dataset_hf, n_min=n_min, n_max=n_max)
    #
    # # deduplicate
    # # unique_indices = np.unique(dataset_hf['train']['text'], return_index=True)[1]
    #
    # unique_indices = get_unique_indices(dataset_hf['train'])
    # assert len(unique_indices) == len(set(dataset_hf['train']['text']))
    #
    # dataset_hf['train'] = dataset_hf['train'].select(unique_indices)
    #
    # # split and save
    # split_dataset = dataset_hf['train'].train_test_split(test_size=0.9, shuffle=True, seed=42)
    # split_dataset.save_to_disk(f"./data/webis/prepared-no-tldr-{n_max}-minus-{n_min}-plus-clear-corpus-webis-tldr-17")


    # add toxicity estimates
    # from eval_utils import get_toxicity_batch
    # dataset = split_dataset.map(
    #     lambda examples: {"toxicity": get_toxicity_batch(examples["text"], batch_size=len(examples["text"]))},
    #     batched=True, desc="Computing toxicity", batch_size=1000
    # )
    # dataset.save_to_disk(f"./data/webis/prepared-tox-no-tldr-{n_max}-minus-{n_min}-plus-clear-corpus-webis-tldr-17")

    from datasets import load_from_disk
    # from eval_utils import llama_quality
    #
    # file_path = f"./data/webis/prepared-no-tldr-{n_max}-minus-{n_min}-plus-clear-corpus-webis-tldr-17"
    # split_dataset = load_from_disk(file_path)
    #
    # dataset = split_dataset.map(
    #     lambda examples: {"llama_quality": llama_quality(examples["text"])},
    #     batched=True, desc="Computing quality", batch_size=10, num_proc=30
    # )
    # dataset.save_to_disk(f"./data/webis/prepared-quality-no-tldr-{n_max}-minus-{n_min}-plus-clear-corpus-webis-tldr-17")


    # file_path = f"./data/webis/prepared-quality-no-tldr-{n_max}-minus-{n_min}-plus-clear-corpus-webis-tldr-17"
    # split_dataset = load_from_disk(file_path)
    # from IPython import embed; embed();


    # split_dataset_hq = split_dataset.filter(lambda ex: ex['llama_quality'] == 2)
    # file_path = f"./data/webis/prepared-high-quality-no-tldr-{n_max}-minus-{n_min}-plus-clear-corpus-webis-tldr-17"
    # split_dataset_hq.save_to_disk(file_path)

    # split_dataset_mq = split_dataset.filter(lambda ex: ex['llama_quality'] == 1)
    # file_path = f"./data/webis/prepared-mid-quality-no-tldr-{n_max}-minus-{n_min}-plus-clear-corpus-webis-tldr-17"
    # print(split_dataset_mq)
    # split_dataset_mq.save_to_disk(file_path)

    # file_path = "./data/webis/prepared-quality-no-tldr-200-minus-20-plus-clear-corpus-webis-tldr-17"
    # split_dataset = load_from_disk(file_path)
    #
    # from eval_utils import StellaEmbedder
    # stella_embedder = StellaEmbedder(multigpu=True)
    # split_dataset['test'] = stella_embedder.add_embeddings_multigpu(split_dataset['test'], batch_size=2048)
    # split_dataset['train'] = stella_embedder.add_embeddings_multigpu(split_dataset['train'], batch_size=2048)
    #

    # file_path = "./data/webis/prepared-cleaned-200-minus-20-plus-corpus-webis-tldr-17"
    # split_dataset = load_from_disk(file_path)
    # split_dataset = filter_posts_by_size(split_dataset, n_min=n_min, n_max=n_max)
    # split_dataset = split_dataset.map(lambda examples: {"text_len": [len(t) for t in examples["text"]]}, batched=True, desc="Adding text_len")

    #
    # ###################
    # # add embeddings
    # ###################
    # import logging
    # logging.basicConfig(level=logging.INFO)
    # from eval_utils import StellaEmbedder
    # from dataset_utils import overwrite_to_disk
    # file_path = "./data/webis/prepared-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17"
    # split_dataset = load_from_disk(file_path)
    # print(split_dataset)
    # stella_embedder = StellaEmbedder(multigpu=True)
    # split_dataset['test'] = stella_embedder.add_embeddings_multigpu(split_dataset['test'], batch_size=2048)
    # split_dataset['train'] = stella_embedder.add_embeddings_multigpu(split_dataset['train'], batch_size=2048)
    # overwrite_to_disk(split_dataset, file_path)
    # exit()

    ###################
    # Add qualities
    ###################
    # file_path = "./data/webis/prepared-cleaned-200-minus-20-plus-corpus-webis-tldr-17"
    # split_dataset = load_from_disk(file_path)
    # print(split_dataset)
    #
    # from eval_utils import llama_quality_scale
    # split_dataset = split_dataset.map(
    #     lambda examples: {"llama_quality_scale": llama_quality_scale(examples["text"])},
    #     batched=True, desc="Computing quality scale", batch_size=10, num_proc=60, load_from_cache_file=False
    # )
    #
    # file_path = "./data/webis/prepared-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17"
    # split_dataset.save_to_disk(file_path)

    ####################
    # Quality datasets
    ####################
    # file_path = "./data/webis/prepared-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17"
    # split_dataset = datasets.load_from_disk(file_path)
    #
    # dataset = datasets.concatenate_datasets([split_dataset['train'], split_dataset['test']])
    # dataset = dataset.filter(lambda ex: ex['llama_quality_scale'] is not None, num_proc=64, load_from_cache_file=False)
    #
    # qualities = np.array(dataset['llama_quality_scale'])
    # print("Separating")
    # for q in [20, 40, 60, 80]:
    #     d = dataset.select(np.where(qualities == q)[0])
    #     print("Q:", q)
    #     print("Size: ", len(d))
    #     # d.save_to_disk(file_path + f"_llama_scale_{q}")

    ####################
    # Length datasets
    ####################
    file_path = "./data/webis/prepared-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17"
    split_dataset = datasets.load_from_disk(file_path)
    dataset = datasets.concatenate_datasets([split_dataset['train'], split_dataset['test']])

    lengths = np.array(dataset['text_len'])

    sort_indices = np.argsort(lengths)
    chunk_size = len(sort_indices) // 3

    short_d = dataset.select(sort_indices[:chunk_size])
    medium_d = dataset.select(sort_indices[chunk_size:2*chunk_size])
    long_d = dataset.select(sort_indices[2*chunk_size:])

    print("Short: ", np.mean(short_d['text_len']))
    print("Medium: ", np.mean(medium_d['text_len']))
    print("Long: ", np.mean(long_d['text_len']))

    short_d.save_to_disk(file_path + "_short")
    medium_d.save_to_disk(file_path + "_medium")
    long_d.save_to_disk(file_path + "_long")

