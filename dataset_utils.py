import json
import random
import os
import re
from datasets import load_dataset, ClassLabel, Dataset, load_from_disk
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

# https://www.statista.com/statistics/1416594/divisive-issues-political-ideology-us/
polarising_questions = [
    "human activity is the main cause of global warming",
    "you worry a great deal of a fair amount about global warming",
    "same-sex marriage should be legally valid",
    "the government should ensure that everyone has healthcare",
    "gun laws should be stricter",
    "marijuana should be legal",
    "immigration is good for the country",
    "protecting the environment has priority over energy development",
    "government should do more to solve the nation's problems",
    "abortion should be legal under any circumstance",
    "you sympathize more with Israelis or Palestinians",
    "you favor death penalty in cases of murder",
    "you have great a great deal or quite a lot of confidence in the police",
    "the federal government has too much power",
    "immigration should be decreased"
]

# https://www.pewresearch.org/politics/2020/02/13/as-economic-concerns-recede-environmental-protection-rises-on-the-publics-policy-agenda/
polarising_topics = [
    "terrorism",
    "economy",
    "health care costs",
    "education",
    "environment",
    "social security",
    "poor and needy",
    "crime",
    "immigration",
    "budget deficit",
    "climate change",
    "drug addiction",
    "infrastructure",
    "jobs",
    "military",
    "gun policy",
    "race relations",
    "global trade",
]
@lru_cache(maxsize=None)
def get_twitter_instructions_generation():
    instructions = []
    for prefix in ["Generate", "Write"]:
        for m in [
            "post", "comment", "viewpoint", "impression",
            "tweet", "remark",  "sentiment",
            "statement", "view", "reaction", "thought", "judgement",
        ]:
            for type in [
                "{} a political {} about if {}.",
                "{} a {} regarding if {}.",
                "{} a {} about if {}."
            ]:
                for question in polarising_questions:
                    instructions.append(type.format(prefix, m, question))

            for type in [
                "{} a political {} about {}.",
                "{} a {} regarding {}.",
                "{} a {} about {}."
            ]:
                for topic in polarising_topics:
                    instructions.append(type.format(prefix, m, topic))

            for type in [
                "{} a political {}.",
                "{} a {}.",
                "{} a {}."
            ]:
                instructions.append(type.format(prefix, m))

        for m in ["attitude", "idea", "opinion"]:
            for type in [
                "{} a political {} about if {}.",
                "{} an {} regarding if {}.",
                "{} an {} about if {}."
            ]:
                for question in polarising_questions:
                    instructions.append(type.format(prefix, m, question))

            for type in [
                "{} a political {} about {}.",
                "{} an {} regarding {}.",
                "{} an {} about {}."
            ]:
                for topic in polarising_topics:
                    instructions.append(type.format(prefix, m, topic))

            for type in [
                "{} a political {}.",
                "{} an {}.",
                "{} an {}."
            ]:
                instructions.append(type.format(prefix, m))

    return instructions


@lru_cache(maxsize=None)
def get_twitter_instructions_train():
    instructions = []
    for prefix in ["Generate", "Write"]:
        for m in [
            "post", "comment", "viewpoint", "impression",
            "tweet", "remark",  "sentiment",
            "statement", "view", "reaction", "thought", "judgement",
        ]:
            for type in [
                "{} a political {}.",
                "{} a {}.",
                "{} a {}."
            ]:
                instructions.append(type.format(prefix, m))

        for m in ["attitude", "idea", "opinion"]:

            for type in [
                "{} a political {}.",
                "{} an {}.",
                "{} an {}."
            ]:
                instructions.append(type.format(prefix, m))

    return instructions


def save_texts_to_csv(texts, filename):

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text'])  # Write the header
        for text in texts:
            writer.writerow([text])  # Write each text as a row


def load_texts_from_csv(filename):
    return list(pd.read_csv(filename, keep_default_na=False)['text'])


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


def load_reddit_dataset_2(cache_dir='data/dataset_cache', load_n=None, load_frac=1.0, lean=None, seed=1):


    dataset_savepath = os.path.join(
        cache_dir, "dataset_cache", f"reddit2_load_n_{load_n}_lean_{lean}_seed_{seed}_{get_current_file_hash()}.save"
    )

    feat_sentiment = ClassLabel(num_classes=2, names=["Liberal", "Conservative"])

    if os.path.exists(dataset_savepath):
        print(f"Loading dataset from cache in: {dataset_savepath}")
        dataset = load_from_disk(dataset_savepath)
    
    else:
        dataset_folder = "./data/reddit_dataset_2/"
        print(f"Loading reddit dataset from {dataset_folder}")
        dataset_liberal = load_dataset_from_csv(os.path.join(dataset_folder, "Liberal.json"))
        dataset_conservative = load_dataset_from_csv(os.path.join(dataset_folder, "Conservative.json"))

        ##Add political lean column
        dataset_liberal = dataset_liberal.map(lambda x: {"Political Lean": 'Liberal'}, batched=True)
        dataset_conservative = dataset_conservative.map(lambda x: {"Political Lean": 'Conservative'}, batched=True)

        dataset = dataset_liberal.concatenate(dataset_conservative)

        if lean:
            dataset = dataset.filter(lambda examples: [p == lean for p in examples['Political Lean']], batched=True)

        if load_n is not None:
            
            load_frac = load_n / len(dataset)
            dataset = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=seed, stratify_by_column="Political Lean")['train']
    
        elif load_frac != 1.0:
            dataset = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=seed, stratify_by_column="Political Lean")['train']

        # save dataset
        print (f"Saving dataset to {dataset_savepath}")
        dataset.save_to_disk(dataset_savepath)

    labels = dataset['Political Lean']

    return dataset, labels, feat_sentiment

    


def load_reddit_dataset(cache_dir='data/dataset_cache', load_n=None, load_frac=1.0, lean=None, seed=1):
    dataset_path = "data/reddit_lib_con/file_name.csv"
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
            batched=True, desc="Merging title and body"
        )

        # clean dataset
        dataset = dataset.map(remove_links, batched=True, desc="Removing links")
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


def create_reddit_instructions(batch):
    # currenlty it's deterministic
    # if you want it to be stohastic also add seed to load_instructions function
    return {"instruction": [f"Generate a post for the r/{sub} subreddit." for sub in batch['subreddit']]}


webis_reddit_dataset_path = "./data/webis/prepared-no-tldr-200-minus-20-plus-clear-corpus-webis-tldr-17"

def load_clear_webis_reddit_instructions(split='train', cache_dir=None):
    # deterministic -> no seed as argument

    if cache_dir:
        os.makedirs(os.path.join(cache_dir, "dataset_cache"), exist_ok=True)
        instructions_cache_savepath = os.path.join(
            cache_dir, "dataset_cache", f"webis_reddit_{get_current_file_hash()}_{split}_instructions.save"
        )

    if cache_dir and os.path.exists(instructions_cache_savepath):
        print(f"Loading instructions from cache in: {instructions_cache_savepath}")
        instructions = load_texts_from_csv(instructions_cache_savepath)

    else:
        dataset = load_from_disk(webis_reddit_dataset_path)[split]
        print("Creating instructions")
        dataset = dataset.map(create_reddit_instructions, batched=True, desc="Creating instructions")

        instructions = dataset['instruction']

        if cache_dir:
            save_texts_to_csv(instructions, instructions_cache_savepath)
            print(f"Instructions cached to {instructions_cache_savepath}.")

    return instructions


def load_clear_webis_reddit_dataset(load_n=None, load_frac=1.0, lean=None, seed=1, split='train', roof_prob=None):

    print(f"Loading webis reddit dataset from {webis_reddit_dataset_path}")

    print(f"Loading full dataset from: {webis_reddit_dataset_path}")
    dataset = load_from_disk(webis_reddit_dataset_path)[split]
    print(f"Dataset loaded. Size: {len(dataset)}")

    if load_n is not None or load_frac != 1.0:

        if load_n is None:
            # load frac != 1.0
            load_n = int(len(dataset)*load_frac)

        dataset = dataset.shuffle(seed=seed)

        if roof_prob:

            # sample by capping the roof_prob per subreddit to 3%
            subreddit_indices = dataset.to_pandas().groupby("subreddit").indices

            # todo: add seed
            unique_subreddits, capped_probs = get_capped_probs(dataset['subreddit'], roof_prob=roof_prob)
            sampled_subreddits = random.choices(unique_subreddits, weights=capped_probs, k=load_n)

            # sample uniformly inside each sampled subreddit
            sampled_post_indices = [random.choice(subreddit_indices[sub]) for sub in sampled_subreddits]

            sample = dataset.select(sampled_post_indices)

            # todo: why is does this slightly above roof_prob
            # Counter(sampled_subreddits).most_common()[0][1] / len(sampled_subreddits)
            # Counter(sample['subreddit']).most_common()[0][1] / load_n

        else:
            sample = dataset.select(range(load_n))

    else:
        # full dataset
        sample = dataset

    print("Creating instructions for sample")
    sample = sample.map(create_reddit_instructions, batched=True, desc="Creating instructions for sample")

    return sample, None, None,


def load_dataset_from_texts_from_csv(filename):
    df = pd.read_csv(filename, keep_default_na=False)
    return Dataset.from_pandas(df)


def remove_links(batch):
    return {"text": [
        re.sub(r'http\S+', '', t).rstrip() for t in batch['text']
    ]}


def get_instructions(dataset_name, cache_dir=None, split="train"):
    if dataset_name is None:
        raise ValueError("dataset_name is not provided")

    elif dataset_name == "webis_reddit":
        instructions = load_clear_webis_reddit_instructions(split=split, cache_dir=cache_dir)

    else:
        raise NotImplementedError(f"Undefined dataset {dataset_name}.")

    return instructions


def get_capped_probs(elements, roof_prob):
    """
    For a list with repeating elements, give the probability for each unique element so that none goes over roof_prob
    e.g. [1, 1, 1, 1, 2, 3, 4, 5, 6, 7], 0.2 -> [1,2,3,4,5,6,7], [0.2, 0.1333, 0.1333, 0.1333, 0.1333,0.1333,0.1333]
    :param elements:
    :param roof_prob:
    :return:
    """

    elements_counter = Counter(elements)
    is_above = {e: cnt / len(elements) > roof_prob for e, cnt in elements_counter.items()}

    # what is the probability of all the elements with frequency <= roof_prob
    # (provided all elements with their frequencies > roof_prob have roof_prob)
    N_uniq_above = sum(is_above.values())  # num of elements that occur more than > roof_prob (unique)
    cum_prob_non_above = 1.0 - N_uniq_above * roof_prob

    # the total number of elements that do not occur more that roof_prob (not unique)
    N_total_non_above = sum([cnt for instr, cnt in elements_counter.items() if not is_above[instr]])

    unique_elements = list(elements_counter.keys())

    capped_probs = [
        roof_prob if is_above[uniq_instr] else
        (elements_counter[uniq_instr] / N_total_non_above) * cum_prob_non_above  # normalize the prob of non_above
        for uniq_instr in unique_elements]

    return unique_elements, capped_probs


def load_human_dataset(dataset_name=None, **kwargs):
    if dataset_name is None:
        raise ValueError("dataset_name is not provided")

    if dataset_name == "twitter":
        human_dataset, _, _ = load_twitter_dataset(**kwargs)
        human_dataset = human_dataset.map(remove_links, batched=True)

    elif dataset_name == "reddit":
        human_dataset, _, _ = load_reddit_dataset(**kwargs)

    elif dataset_name == "webis_reddit":
        human_dataset, _, _ = load_clear_webis_reddit_dataset(**kwargs)

    elif dataset_name == "news":
        human_dataset, _, _ = load_news_dataset(**kwargs)
    
    elif dataset_name == "reddit2":
        human_dataset, _, _ = load_reddit_dataset_2(**kwargs)

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
        dataset = dataset.map(remove_links, batched=True, desc="Removing links")
        dataset = dataset.filter(lambda examples: [len(word_tokenize(t)) > 10 for t in examples['text']], batched=True)

        dataset = dataset.map(
            lambda examples: {
                "Political Lean": [party_2_lean[p] for p in examples["party"]]
            }, batched=True, desc="Annotating party to pol. lean"
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
