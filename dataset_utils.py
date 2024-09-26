import json
import os
import re
from datasets import load_dataset, ClassLabel, Dataset, load_from_disk
from nltk import sent_tokenize, word_tokenize
import csv

from tqdm import trange

try:
    import pandas as pd
except:
    ...

# nltk.download("punkt")

import torch
from transformers import BertTokenizer, BertModel
from functools import lru_cache

import inspect
import sys
import hashlib

def get_current_file_hash():
    current_file_text = inspect.getsource(sys.modules[__name__])
    file_hash = int(hashlib.sha256(current_file_text.encode('utf-8')).hexdigest(), 16)
    return file_hash


@lru_cache(maxsize=None)
def get_instructions():
    instructions = []
    for prefix in ["Generate", "Write"]:
        for m in [
            "post", "comment", "viewpoint", "impression", "attitude",
            "tweet", "remark", "opinion", "sentiment", "idea",
            "statement", "view", "reaction", "thought", "judgement"
        ]:
            for type in [
                "{} a political {}.",
                "{} a {} regarding politics.",
                "{} a {} about politics."
            ]:
                instructions.append(type.format(prefix, m))

    return instructions


def save_texts_to_csv(texts, filename):

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text'])  # Write the header
        for text in texts:
            writer.writerow([text])  # Write each text as a row


def load_news_dataset(cache_dir=None, load_n=None, load_frac=1.0, lean=None, seed=1):
    dataset_folder = "./data/news_dataset/"
    print(f"Loading news dataset from {dataset_folder}")

    feat_sentiment = ClassLabel(num_classes = 2, names=["Liberal", "Conservative"])

    # dataset_savepath = os.path.join(
    #     cache_dir, "dataset_cache", f"news_load_n_{load_n}_lean_{lean}_seed_{seed}_{get_current_file_hash()}.save"
    # )

    dataset_savepath = os.path.join(
        "data", "dataset_cache", f"news_load_n_{load_n}_lean_{lean}_seed_{seed}.save"
    )

    if os.path.exists(dataset_savepath):
        print(f"Loading dataset from cache in: {dataset_savepath}")
        dataset = load_from_disk(dataset_savepath)
    
    else:
        print(f"path {dataset_savepath} does not exist")
        print("Loading dataset from files")

        bias_2_lean = {
            'left': 'Conservative',
             'right': 'Liberal',
        }
            
        files = [f for f in os.listdir(dataset_folder) if f.endswith(".json")]

        # Initialize an empty list to store data
        data = {"text": [], "Political Lean": []}

        for filename in trange(len(files)):

            file_path = os.path.join(dataset_folder, files[filename])
            with open(file_path, "r") as f:
                entry = json.load(f)
                content = entry['content']
                bias = entry['bias_text']
                if bias in ['left', 'right']:
                    data["text"].append(content)
                    data["Political Lean"].append(bias_2_lean[bias])

        # Create the dataset
        dataset = Dataset.from_dict(data)
        print (f"Dataset size: {len(dataset)}")


        if lean:
            print('Filtering')
            dataset = dataset.filter(lambda examples: [p == lean for p in examples['Political Lean']], batched=True)

        if load_n is not None:

            load_frac = load_n / len(dataset)
            dataset = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=seed, stratify_by_column="Political Lean")['train']

        elif load_frac != 1.0:
            dataset = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=seed, stratify_by_column="Political Lean")['train']

        # save dataset
        print (f"Dataset size: {len(dataset)}")

        dataset.save_to_disk(dataset_savepath)

    labels = dataset['Political Lean']

    return dataset, labels, feat_sentiment


def load_reddit_dataset(cache_dir=None, load_n=None, load_frac=1.0, lean=None, seed=1):
    dataset_path = "./data/reddit_lib_con/file_name.csv"
    print(f"Loading reddit dataset from {dataset_path}")

    feat_sentiment = ClassLabel(num_classes=2, names=["Liberal", "Conservative"])


    dataset_savepath = os.path.join(
        cache_dir, "dataset_cache", f"reddit_load_n_{load_n}_lean_{lean}_seed_{seed}_{get_current_file_hash()}.save"
    )

    if os.path.exists(dataset_savepath):
        print(f"Loading dataset from cache in: {dataset_savepath}")
        dataset = load_from_disk(dataset_savepath)

    else:

        dataset = load_dataset("csv", data_files=dataset_path, cache_dir=cache_dir)['train']
        print(set(dataset["Subreddit"]))
        # dataset = dataset.filter(lambda e: e["Subreddit"] in ["conservatives", "democrats"])
        # dataset = dataset.filter(lambda e: e["Subreddit"] in ["feminisms", "conservatives"])
        # dataset = dataset.filter(lambda e: e["Score"] > 50)

        if lean:
            dataset = dataset.filter(lambda e: e["Political Lean"] == lean)

        def merge_title_and_text(title, text):
            post = ""

            if title is not None:
                # post += "Title: "+title
                post += title

            if title is not None and text is not None:
                post += "\n\n"

            if text is not None:
                post += text

            return post.strip()

        dataset = dataset.map(
            lambda examples: {
                "text": [
                    merge_title_and_text(ti, te) for ti, te in zip(examples["Title"], examples["Text"])
                ]
            },
            batched=True
        )

        # clean dataset
        dataset = dataset.map(remove_links, batched=True)
        dataset = dataset.filter(lambda examples: [len(word_tokenize(t)) > 4 for t in examples['text']], batched=True)

        # stratify
        # dataset = dataset.cast_column("Political Lean", feat_sentiment)

        if load_n is not None:
            load_frac = load_n / len(dataset)
            dataset = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=seed, stratify_by_column="Political Lean")['train']
        elif load_frac != 1.0:
            dataset = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=seed, stratify_by_column="Political Lean")['train']

        # dataset = dataset.select(range(1000))

        dataset = dataset.remove_columns(["Text", "Title"])

        # save dataset
        print (f"Saving dataset to {dataset_savepath}")
        dataset.save_to_disk(dataset_savepath)

    labels = dataset['Political Lean']

    return dataset, labels, feat_sentiment


def load_dataset_from_csv(filename):

    print(f"Loading dataset from {filename}")
    df = pd.read_csv(filename, keep_default_na=False)
    return Dataset.from_pandas(df)


def remove_links(batch):
    return {"text": [
        re.sub(r'http\S+', '', t).rstrip() for t in batch['text']
    ]}

def load_human_dataset(dataset_name=None, **kwargs):
    if dataset_name is None:
        raise ValueError("dataset_name is not provided")

    if dataset_name == "twitter":
        human_dataset, _, _ = load_twitter_dataset(**kwargs)
        human_dataset = human_dataset.map(remove_links, batched=True)

    elif dataset_name == "reddit":
        human_dataset, _, _ = load_reddit_dataset(**kwargs)

    elif dataset_name == "news":
        human_dataset, _, _ = load_news_dataset(**kwargs)

    else:
        raise NotImplementedError(f"Undefined dataset {dataset_name}.")

    return human_dataset




def load_twitter_dataset(cache_dir='data', load_n=None, load_frac=1.0, lean=None, seed=1):
    dataset_name = "m-newhauser/senator-tweets"
    print(f"Loading dataset {dataset_name}")

    feat_sentiment = ClassLabel(num_classes=2, names=["Liberal", "Conservative"])

    # dataset_savepath = os.path.join(
    #     cache_dir, "dataset_cache",
    #     f"twitter_load_n_{load_n}_lean_{lean}_{get_current_file_hash()}.save"
    # )
    dataset_savepath = os.path.join(
        cache_dir, "dataset_cache",
        f"twitter_lean_{lean}.save"
    )

    if os.path.exists(dataset_savepath):
        print(f"Loading dataset from cache in: {dataset_savepath}")
        dataset = load_from_disk(dataset_savepath)

    else:
        print("Loading dataset from Huggingface")
        party_2_lean = {
            "Democrat": "Liberal",
            "Republican": "Conservative"
        }
        # Prepare data
        dataset = load_dataset(dataset_name, cache_dir=cache_dir, split='train')

        if lean is not None:
            dataset = dataset.filter(lambda examples: [party_2_lean.get(p, p) == lean for p in examples['party']], batched=True)

        # clean dataset
        dataset = dataset.map(remove_links, batched=True)
        dataset = dataset.filter(lambda examples: [len(word_tokenize(t)) > 10 for t in examples['text']], batched=True)

        dataset = dataset.map(
            lambda examples: {
                "Political Lean": [party_2_lean[p] for p in examples["party"]]
            }, batched=True
        )

        # dataset = dataset.cast_column("Political Lean", feat_sentiment)

        # save dataset
        dataset.save_to_disk(dataset_savepath)
        print(f"Dataset saved to: {dataset_savepath}")

    if load_n is not None:
        load_frac = load_n / len(dataset)
        dataset = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=seed, stratify_by_column="Political Lean")['train']
        # dataset_ = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=seed, stratify_by_column="Political Lean")['test']

    elif load_frac != 1.0:
        dataset = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=seed, stratify_by_column="Political Lean")['train']

    dataset = dataset.remove_columns(["embeddings"])

    labels = dataset['Political Lean']

    return dataset, labels, feat_sentiment


class BertEmbedder:

    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.bert_model = BertModel.from_pretrained("bert-base-uncased").eval().to(self.device)

    def add_bert_embeddings(self, dataset):

        def bert_embed_text(examples):
            encoded_input = self.bert_tokenizer(examples['text'], return_tensors='pt', padding=True)
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            output = self.bert_model(**encoded_input)
            embeddigs = output['last_hidden_state'][:, 0, :]  # we take the representation of the [CLS] token
            if self.device == torch.device("mps"):
                torch.mps.empty_cache()
            elif self.device == torch.device("cuda"):
                torch.cuda.empty_cache()
            return {"embeddings": list(embeddigs)}

        return dataset.map(bert_embed_text, batched=True, batch_size=128)
