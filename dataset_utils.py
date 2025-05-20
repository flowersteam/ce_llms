import json
import random
import os
import shutil
import re
from datasets import load_dataset, ClassLabel, Dataset, load_from_disk, concatenate_datasets
from datasets.dataset_dict import DatasetDict
import csv
from collections import Counter

from tqdm import trange

try:
    import pandas as pd
except:
    ...

try:
    from nltk import sent_tokenize, word_tokenize
except:
    ...

import inspect
import sys
import hashlib
import numpy as np

datasets_to_merge = {
    "wikipedia": f"./data/wikipedia/wikipedia_dataset_with_qualities",
    "webis_reddit": "./data/webis/webis_dataset_with_qualities",
    "100m_tweets": "data/twitter_100m/100m_tweets_dataset_with_qualities"
}
datasets_clusters_to_merge = {
    "wikipedia": "./data/wikipedia/selected_clusters_indices_to_path.json",
    "webis_reddit": "./data/webis/selected_clusters_indices_to_path.json",
    "100m_tweets": "./data/twitter_100m/selected_clusters_indices_to_path.json"
}


def load_merged_participants_dataset(path_pattern, all_participants, size=None, partition_instruction=None):
    """
    path_pattern : 'all_parts' is replaced by part_X for part_X in all_participants
    """

    # either we only have one participant or there is a pattern for matching
    assert len(all_participants) > 1 or "all_parts" in path_pattern

    ds = []
    for part in all_participants:
        d = load_from_disk(path_pattern.replace("all_parts", part))

        if partition_instruction is not None:
            d = d.filter(lambda x: x['instruction'] == partition_instruction)

        d = d.add_column('source', [part]*len(d))
        ds.append(d)

    if size is not None:
        per_part_size = int(np.ceil(size / len(all_participants)))
        ds = [d.select(range(per_part_size)) for d in ds]

    # in case it's not divisible take less from the last participant
    merged_d = concatenate_datasets(ds)

    if size is not None:
        merged_d = merged_d.shuffle().select(range(size))

    return merged_d


def overwrite_to_disk(d, save_path):
    print(f"Saving path: {save_path}")
    if os.path.exists(save_path):
        assert not save_path.endswith("/")
        d.save_to_disk(save_path + "_temp")
        shutil.rmtree(save_path)
        os.rename(save_path + "_temp", save_path)
    else:
        d.save_to_disk(save_path)


def get_current_file_hash():
    current_file_text = inspect.getsource(sys.modules[__name__])
    file_hash = int(hashlib.sha256(current_file_text.encode('utf-8')).hexdigest(), 16)
    return file_hash


def save_texts_to_csv(texts, filename):

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text'])  # Write the header
        for text in texts:
            writer.writerow([text])  # Write each text as a row


def load_texts_from_csv(filename):
    return list(pd.read_csv(filename, keep_default_na=False)['text'])

def get_twitter_instructions(n, prompt = 'neutral'):
    if prompt == 'political':
        return [f"Generate a twitter post about a political topic."] * n
    else:
        return [f"Generate a twitter post."] * n

def get_merged_instructions(n):
    return [f"Generate a post."] * n

# def create_twitter_instructions(batch):
#     return {"instruction": [f"Generate a twitter post."] * len(batch['text'])}
#     # return {"instruction": [f"Generate a post for the r/{sub} subreddit." for sub in batch['subreddit']]}

def load_senator_tweets_dataset_old(load_n=None, load_frac=1.0, lean=None, seed=1, split='train', dataset_type="standard"):

    dataset_name = "m-newhauser/senator-tweets"
    print(f"Loading twitter dataset: {dataset_name}")

    # Prepare data
    dataset = load_from_disk(dataset_name)
    if split == "all":
        if type(dataset) == DatasetDict:
            assert dataset.keys() == {'train', 'test'}
            dataset = concatenate_datasets([dataset['train'], dataset['test']])

    else:
        dataset = dataset[split]

    # clean dataset
    dataset = dataset.map(remove_links, batched=True, desc="Removing links", load_from_cache_file=False)
    dataset = dataset.filter(lambda examples: [len(word_tokenize(t)) > 10 for t in examples['text']], batched=True, load_from_cache_file=False)
    dataset = dataset.remove_columns(["embeddings"])

    print(f"Dataset loaded. Size: {len(dataset)}")

    if load_n is not None or load_frac != 1.0:
        if load_n is None:
            # load frac != 1.0
            load_n = int(len(dataset)*load_frac)

        dataset = dataset.shuffle(seed=seed)
        sample = dataset.select(range(load_n))
    else:
        # full dataset
        sample = dataset

    print("Creating instructions for sample")
    sample = sample.add_column("instruction", get_twitter_instructions(len(sample)))
    # sample = sample.map(create_twitter_instructions, batched=True, desc="Creating instructions for sample", load_from_cache_file=False, batch_size=len(sample))

    return sample, None, None


def get_wikipedia_instructions(n):
    return [f"Generate an opening wikipedia paragraph."] * n


def load_wikipedia_dataset(load_n=None, load_frac=1.0, lean=None, seed=1, split='train', dataset_type="standard"):

    dataset_path = f"./data/wikipedia/wikipedia_dataset_with_qualities"

    if dataset_type == "Q20":
        dataset_path += "_llama_scale_20"
    elif dataset_type == "Q40":
        dataset_path += "_llama_scale_40"
    elif dataset_type == "Q60":
        dataset_path += "_llama_scale_60"
    elif dataset_type == "Q80":
        dataset_path += "_llama_scale_80"

    # clusters
    elif re.match(r"cluster_\d+", dataset_type):
        cluster_index = int(dataset_type.split("_")[-1])
        cluster_index_to_path_dict_path = "./data/wikipedia/selected_clusters_indices_to_path.json"

        with open(cluster_index_to_path_dict_path, "r") as f:
            cluster_index_to_path_dict = json.load(f)
        dataset_path = cluster_index_to_path_dict[str(cluster_index)]
    elif dataset_type == "standard":
        ...
    else:
        raise ValueError(f"Type {dataset_type} not recognized.")

    print(f"Loading wikipedia dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)

    print(f"Dataset loaded. Size: {len(dataset)}")

    if load_n is not None or load_frac != 1.0:
        if load_n is None:
            # load frac != 1.0
            load_n = int(len(dataset)*load_frac)

        dataset = dataset.shuffle(seed=seed)
        sample = dataset.select(range(load_n))
    else:
        # full dataset
        sample = dataset.shuffle(seed=seed)

    print("Creating instructions for sample")
    sample = sample.add_column("instruction", get_wikipedia_instructions(len(sample)))

    return sample, None, None

def load_reddit_submissions_dataset(load_n=None, load_frac=1.0, lean=None, seed=1, split='train', dataset_type="standard"):

    dataset_path = f"./data/reddit_submissions/prepared-reddit-submissions-dataset-with-qualities-200-minus-20-plus"
    # dataset_path = "data/reddit_submissions/reddit_submissions_dataset_with_qualities" # new

    if dataset_type == "Q20":
        dataset_path += "_llama_scale_20"
    elif dataset_type == "Q40":
        dataset_path += "_llama_scale_40"
    elif dataset_type == "Q60":
        dataset_path += "_llama_scale_60"
    elif dataset_type == "Q80":
        dataset_path += "_llama_scale_80"
    # clusters
    # if dataset type is cluster_[0-9]+
    elif re.match(r"cluster_\d+", dataset_type):
        cluster_index = int(dataset_type.split("_")[-1])
        cluster_index_to_path_dict_path = "./data/reddit_submissions/selected_clusters_indices_to_path.json"

        with open(cluster_index_to_path_dict_path, "r") as f:
            cluster_index_to_path_dict = json.load(f)
        dataset_path = cluster_index_to_path_dict[str(cluster_index)]
    elif dataset_type == "standard":
        ...
    else:
        raise ValueError(f"Type {dataset_type} not recognized.")

    print(f"Loading reddit submissions dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)

    print(f"Dataset loaded. Size: {len(dataset)}")

    if load_n is not None or load_frac != 1.0:
        if load_n is None:
            # load frac != 1.0
            load_n = int(len(dataset)*load_frac)

        dataset = dataset.shuffle(seed=seed)
        sample = dataset.select(range(load_n))
    else:
        # full dataset
        sample = dataset.shuffle(seed=seed)

    print("Creating instructions for sample")
    sample = sample.add_column("instruction", get_reddit_instructions(len(sample)))
    # sample = sample.map(create_reddit_instructions, batched=True, desc="Creating instructions for sample", load_from_cache_file=False, batch_size=len(sample))

    return sample, None, None

# before GPT-3
def load_100m_tweets_dataset(load_n=None, load_frac=1.0, lean=None, seed=1, split='train', dataset_type="standard"):

    # dataset_path = "data/twitter_100m/prepared-100m-tweets-english-qualities-cl-before-gpt3-200-minus-20-plus"
    dataset_path = "data/twitter_100m/100m_tweets_dataset_with_qualities"  # new

    if dataset_type == "Q20":
        dataset_path += "_llama_scale_20"
    elif dataset_type == "Q40":
        dataset_path += "_llama_scale_40"
    elif dataset_type == "Q60":
        dataset_path += "_llama_scale_60"
    elif dataset_type == "Q80":
        dataset_path += "_llama_scale_80"
    elif dataset_type == "short":
        dataset_path += "_short"
    elif dataset_type == "medium":
        dataset_path += "_medium"
    elif dataset_type == "long":
        dataset_path += "_long"
    # clusters
    # if dataset type is cluster_[0-9]+
    elif re.match(r"cluster_\d+", dataset_type):
        cluster_index = int(dataset_type.split("_")[-1])
        cluster_index_to_path_dict_path = "./data/twitter_100m/selected_clusters_indices_to_path.json"

        with open(cluster_index_to_path_dict_path, "r") as f:
            cluster_index_to_path_dict = json.load(f)
        dataset_path = cluster_index_to_path_dict[str(cluster_index)]
    elif dataset_type == "standard":
        ...
    else:
        raise ValueError(f"Type {dataset_type} not recognized.")

    print(f"Loading 100m tweets dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)

    print(f"Dataset loaded. Size: {len(dataset)}")

    if load_n is not None or load_frac != 1.0:
        if load_n is None:
            # load frac != 1.0
            load_n = int(len(dataset)*load_frac)

        dataset = dataset.shuffle(seed=seed)
        sample = dataset.select(range(load_n))
    else:
        # full dataset
        sample = dataset.shuffle(seed=seed)

    print("Creating instructions for sample")
    sample = sample.add_column("instruction", get_twitter_instructions(len(sample)))
    # sample = sample.map(create_twitter_instructions, batched=True, desc="Creating instructions for sample", load_from_cache_file=False, batch_size=len(sample))

    return sample, None, None


def get_reddit_instructions(n):
    return [f"Generate a reddit post."] * n

# def create_reddit_instructions(batch):
#     return {"instruction": [f"Generate a reddit post."] * len(batch)}
#     # return {"instruction": [f"Generate a post for the r/{sub} subreddit." for sub in batch['subreddit']]}


def load_clear_webis_reddit_dataset(load_n=None, load_frac=1.0, lean=None, seed=1, split='train', dataset_type="standard"):

    # dataset_path = "./data/webis/prepared-quality-cleaned-200-minus-20-plus-corpus-webis-tldr-17"
    dataset_path = "./data/webis/webis_dataset_with_qualities"  # new

    if dataset_type == "Q20":
        dataset_path += "_llama_scale_20"
    elif dataset_type == "Q40":
        dataset_path += "_llama_scale_40"
    elif dataset_type == "Q60":
        dataset_path += "_llama_scale_60"
    elif dataset_type == "Q80":
        dataset_path += "_llama_scale_80"
    # len
    elif dataset_type == "short":
        dataset_path += "_short"
    elif dataset_type == "medium":
        dataset_path += "_medium"
    elif dataset_type == "long":
        dataset_path += "_long"
    # cluster
    # if dataset type is cluster_[0-9]+
    elif re.match(r"cluster_\d+", dataset_type):
        cluster_index = int(dataset_type.split("_")[-1])
        cluster_index_to_path_dict_path = "./data/webis/selected_clusters_indices_to_path.json"

        with open(cluster_index_to_path_dict_path, "r") as f:
            cluster_index_to_path_dict = json.load(f)
        dataset_path = cluster_index_to_path_dict[str(cluster_index)]
    elif dataset_type == "standard":
        ...
    else:
        raise ValueError(f"Type {dataset_type} not recognized.")

    print(f"Loading webis reddit (type {dataset_type}) dataset from {dataset_path}")

    # Prepare data
    dataset = load_from_disk(dataset_path)
    if split == "all":
        if type(dataset) == DatasetDict:
            assert dataset.keys() == {'train', 'test'}
            dataset = concatenate_datasets([dataset['train'], dataset['test']])

    else:
        dataset = dataset[split]

    print(f"Dataset loaded. Size: {len(dataset)}")

    if load_n is not None or load_frac != 1.0:

        if load_n is None:
            # load frac != 1.0
            load_n = int(len(dataset)*load_frac)

        dataset = dataset.shuffle(seed=seed)

        sample = dataset.select(range(load_n))

    else:
        # full dataset
        sample = dataset.shuffle(seed=seed)

    print("Creating instructions for sample")
    sample = sample.add_column("instruction", get_reddit_instructions(len(sample)))
    # sample = sample.map(create_reddit_instructions, batched=True, desc="Creating instructions for sample", load_from_cache_file=False, batch_size=10000, input_columns="id")

    # rstrip
    assert sample['text'] == sample.map(lambda examples: {"text": [t.rstrip() for t in examples["text"]]}, batched=True, desc="rstrip", num_proc=64)['text']

    return sample, None, None


def load_dataset_from_texts_from_csv(filename):
    df = pd.read_csv(filename, keep_default_na=False)
    return Dataset.from_pandas(df)


def remove_links(batch):
    return {"text": [re.sub(r'http\S+', '', t).rstrip() for t in batch['text']]}

def remove_trailling_hashtags(batch):
    return {"text": [re.sub(r"(?:\s*#\w+)+$", "", t) for t in batch['text']]}


def get_merged_random_instructions(n):
    dataset_names = datasets_to_merge.keys()

    n_per_dataset = int(np.ceil(n / len(dataset_names)))

    per_dataset_instructions = [get_instructions(dataset_name, n=n_per_dataset) for dataset_name in dataset_names]
    # assert per dataset instruction length is the same
    assert all([len(instructions) == n_per_dataset for instructions in per_dataset_instructions])

    # interleave instructions
    interleaved_instructions = [val for tup in zip(*per_dataset_instructions) for val in tup]
    interleaved_instructions = interleaved_instructions[:n]

    return interleaved_instructions


def get_instructions(dataset_name, n=250, prompt='neutral'):
    if dataset_name is None:
        raise ValueError("dataset_name is not provided")

    elif dataset_name == "webis_reddit":
        instructions = get_reddit_instructions(n=n)

    elif dataset_name == "senator_tweets":
        instructions = get_twitter_instructions(n=n, prompt=prompt)

    elif dataset_name == "100m_tweets":
        instructions = get_twitter_instructions(n=n)

    elif dataset_name == "reddit_submissions":
        instructions = get_reddit_instructions(n=n)

    elif dataset_name == "wikipedia":
        instructions = get_wikipedia_instructions(n=n)

    elif dataset_name == "merged":
        instructions = get_merged_random_instructions(n=n)

    else:
        raise NotImplementedError(f"Undefined dataset {dataset_name}.")

    return instructions


def load_human_dataset(dataset_name=None, **kwargs):
    if dataset_name is None:
        raise ValueError("dataset_name is not provided")

    if dataset_name == "twitter":
        raise DeprecationWarning()
        human_dataset, _, _ = load_twitter_dataset(**kwargs)
        human_dataset = human_dataset.map(remove_links, batched=True, desc="Removing links", load_from_cache_file=False)

    elif dataset_name == "100m_tweets":
        human_dataset, _, _ = load_100m_tweets_dataset(**kwargs)

    elif dataset_name == "reddit_submissions":
        human_dataset, _, _ = load_reddit_submissions_dataset(**kwargs)

    elif dataset_name == "wikipedia":
        human_dataset, _, _ = load_wikipedia_dataset(**kwargs)

    elif dataset_name == "senator_tweets":
        human_dataset, _, _ = load_senator_tweets_dataset(**kwargs)

    elif dataset_name == "webis_reddit":
        human_dataset, _, _ = load_clear_webis_reddit_dataset(**kwargs)

    elif dataset_name == "merged":
        human_dataset, _, _ = load_merged_dataset(**kwargs)

    else:
        raise NotImplementedError(f"Undefined dataset {dataset_name}.")

    return human_dataset


def load_merged_dataset(load_n=None, load_frac=1.0, lean=None, seed=1, split='train', dataset_type="standard"):

    if load_frac != 1.0:
        raise ValueError("load_frac != 1.0 not supported for merged datasets")

    datasets = {}
    print(f"Merging datasets: {datasets_to_merge.values()}")
    for dataset_name in datasets_to_merge.keys():

        if re.match(r"cluster_\d+", dataset_type):
            cluster_index = int(dataset_type.split("_")[-1])
            cluster_index_to_path_dict_path = datasets_clusters_to_merge[dataset_name]

            with open(cluster_index_to_path_dict_path, "r") as f:
                cluster_index_to_path_dict = json.load(f)
            dataset_path = cluster_index_to_path_dict[str(cluster_index)]

        else:
            dataset_path = datasets_to_merge[dataset_name]

        d = load_from_disk(dataset_path)
        datasets[dataset_name] = d
        print(f"Dataset {dataset_name} loaded. Size: {len(dataset_name)}")


    # load stratified
    if load_n is not None:
        load_n_per_d = int(np.ceil(load_n / len(datasets)))
    else:
        load_n_per_d = min([len(d) for d in datasets.values()])
        load_n = 3*load_n_per_d

    for dataset_name, d in datasets.items():
        d = d.shuffle(seed=seed)
        d = d.select(range(load_n_per_d))
        d = d.add_column("instruction", get_instructions(dataset_name, len(d)))

        datasets[dataset_name] = d
        print("Creating instructions for sample")

    # select features intersection
    common_features = set.intersection(*[set(d.features.keys()) for d in datasets.values()])

    common_features.remove('id')  # can be different type in different datasets, and we don't need it

    dataset = concatenate_datasets([d.select_columns(common_features) for d in datasets.values()])
    dataset = dataset.shuffle(seed=seed)
    sample = dataset.select(range(load_n))

    return sample, None, None


# before GPT-3
def load_senator_tweets_dataset(load_n=None, load_frac=1.0, lean=None, seed=1, split='train', dataset_type="standard"):

    dataset_path = "data/senator_tweets/prepared-senator-tweets-qualities"
    if dataset_type == '50l50r':
         dataset_path = "/lustre/fsn1/projects/rech/imi/uov75at/data/senator_tweets/prepared-50-50-polarization-political-dataset/"
    elif dataset_type == "25l75r":
        dataset_path = "/lustre/fsn1/projects/rech/imi/uov75at/data/senator_tweets/prepared-25-75-polarization-political-dataset/"
    elif dataset_type == "75l25r":
        dataset_path = "/lustre/fsn1/projects/rech/imi/uov75at/data/senator_tweets/prepared-75-25-polarization-political-dataset/"
    elif dataset_type == "0l100r":
        dataset_path = "/lustre/fsn1/projects/rech/imi/uov75at/data/senator_tweets/prepared-0-100-polarization-political-dataset/"
    elif dataset_type == "100l0r":
        dataset_path = "/lustre/fsn1/projects/rech/imi/uov75at/data/senator_tweets/prepared-100-0-polarization-political-dataset/"
    elif dataset_type == "Q51":
        dataset_path += "_llama_scale_51"
    elif dataset_type == "Q80":
        dataset_path += "_llama_scale_80"
    elif dataset_type == "short":
        dataset_path += "_short"
    elif dataset_type == "long":
        dataset_path += "_long"
    elif dataset_type == "standard":
        ...
    else:
        raise ValueError(f"Type {dataset_type} not recognized.")

    print(f"Loading senator tweets dataset from {dataset_path}")
    if split == "all":
        dataset = load_from_disk(dataset_path)
        if type(dataset) == DatasetDict:
            assert dataset.keys() == {'train', 'test'}
            dataset = concatenate_datasets([dataset['train'], dataset['test']])
    elif split == "none":
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_from_disk(dataset_path)[split]

    print(f"Dataset loaded. Size: {len(dataset)}")

    if load_n is not None or load_frac != 1.0:
        if load_n is None:
            # load frac != 1.0
            load_n = int(len(dataset)*load_frac)

        dataset = dataset.shuffle(seed=seed)
        sample = dataset.select(range(load_n))
    else:
        # full dataset
        sample = dataset.shuffle(seed=seed)

    print("Creating instructions for sample")
    sample = sample.add_column("instruction", get_twitter_instructions(len(sample)))
    # sample = sample.map(create_twitter_instructions, batched=True, desc="Creating instructions for sample", load_from_cache_file=False, batch_size=len(sample))

    return sample, None, None
