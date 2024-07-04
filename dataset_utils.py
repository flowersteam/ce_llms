from datasets import load_dataset, ClassLabel, Dataset
from nltk import sent_tokenize, word_tokenize
import csv
import pandas as pd
# nltk.download("punkt")

import torch
from transformers import BertTokenizer, BertModel


def save_texts_to_csv(texts, filename):

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text'])  # Write the header
        for text in texts:
            writer.writerow([text])  # Write each text as a row


def load_reddit_dataset(cache_dir=None, load_n=None, load_frac=1.0, lean=None):
    dataset_path = "./data/reddit_lib_con/file_name.csv"
    print(f"Loading reddit dataset from {dataset_path}")
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

    dataset = dataset.filter(lambda examples: [len(word_tokenize(t)) > 4 for t in examples['text']], batched=True)

    # stratify
    feat_sentiment = ClassLabel(num_classes=2, names=["Liberal", "Conservative"])
    dataset = dataset.cast_column("Political Lean", feat_sentiment)

    if load_n is not None:
        load_frac = load_n / len(dataset)
        dataset = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=1, stratify_by_column="Political Lean")['train']
    elif load_frac != 1.0:
        dataset = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=1, stratify_by_column="Political Lean")['train']


    # dataset = dataset.select(range(1000))

    labels = dataset['Political Lean']

    dataset = dataset.remove_columns(["Text", "Title"])

    return dataset, labels, feat_sentiment


def load_dataset_from_csv(filename):
    print(f"Loading dataset from {filename}")
    df = pd.read_csv(filename, keep_default_na=False)
    return Dataset.from_pandas(df)


def load_twitter_dataset(cache_dir=None, load_n=None, load_frac=1.0, lean=None):
    dataset_name = "m-newhauser/senator-tweets"
    print(f"Loading dataset {dataset_name}")

    party_2_lean = {
        "Democrat": "Liberal",
        "Republican": "Conservative"
    }

    # Prepare data
    dataset = load_dataset(dataset_name, cache_dir=cache_dir, split='train')

    if lean is not None:
        dataset = dataset.filter(lambda examples: [party_2_lean.get(p, p) == lean for p in examples['party']], batched=True)

    dataset = dataset.filter(lambda examples: [len(word_tokenize(t)) > 10 for t in examples['text']], batched=True)

    dataset = dataset.map(
        lambda examples: {
            "Political Lean": [party_2_lean[p] for p in examples["party"]]
        },
        batched=True
    )

    feat_sentiment = ClassLabel(num_classes=2, names=["Liberal", "Conservative"])
    dataset = dataset.cast_column("Political Lean", feat_sentiment)

    if load_n is not None:
        load_frac = load_n / len(dataset)
        dataset = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=1, stratify_by_column="Political Lean")['train']

    elif load_frac != 1.0:
        dataset = dataset.train_test_split(test_size=1-load_frac, shuffle=True, seed=1, stratify_by_column="Political Lean")['train']

    labels = dataset['Political Lean']

    dataset = dataset.remove_columns(["embeddings"])

    return dataset, labels, feat_sentiment


class BertEmbedder:

    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.has_mps:
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
            return {"embeddings": list(embeddigs)}

        return dataset.map(bert_embed_text, batched=True, batch_size=256)
