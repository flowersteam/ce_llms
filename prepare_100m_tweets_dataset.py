import datasets
from datasets import load_dataset
import re
from datetime import datetime
from eval_utils import llama_quality, llama_is_english

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
dataset = datasets.load_from_disk(f"./data/twitter_100m/prepared-100m-tweets-english-qualities-before-gpt3-{n_max}-minus-{n_min}-plus")

dataset_hq = dataset.filter(lambda ex: ex['llama_quality'] == 2)
print(dataset_hq)
file_path = f"./data/twitter_100m/prepared-100m-tweets-english-high-quality-before-gpt3-{n_max}-minus-{n_min}-plus"
dataset_hq.save_to_disk(file_path)
print(f"Saved to: {file_path}")

dataset_mq = dataset.filter(lambda ex: ex['llama_quality'] == 1)
print(dataset_mq)
file_path = f"./data/twitter_100m/prepared-100m-tweets-english-mid-quality-before-gpt3-{n_max}-minus-{n_min}-plus"
dataset_mq.save_to_disk(file_path)
print(f"Saved to: {file_path}")
