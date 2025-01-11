from eval_utils import llama_quality
from dataset_utils import *
from datasets import load_dataset

dataset = load_dataset("m-newhauser/senator-tweets")

dataset = dataset.map(remove_links, batched=True, desc="Removing links", load_from_cache_file=False)
dataset = dataset.filter(lambda examples: [len(word_tokenize(t)) > 10 for t in examples['text']], batched=True,
                         load_from_cache_file=False)
dataset = dataset.remove_columns(["embeddings"])

dataset['train'] = dataset['train'].map(
    lambda examples: {"llama_quality": llama_quality(examples["text"])},
    batched=True, desc="Computing quality", batch_size=10, num_proc=30
)

dataset['test'] = dataset['test'].map(
    lambda examples: {"llama_quality": llama_quality(examples["text"])},
    batched=True, desc="Computing quality", batch_size=10, num_proc=30
)

dataset.save_to_disk(f"./data/senator_tweets/prepared-senator-tweets")

split_dataset_hq = dataset.filter(lambda ex: ex['llama_quality'] == 2)
split_dataset_hq.save_to_disk(f"./data/senator_tweets/prepared-high-quality-senator-tweets")

split_dataset_mq = dataset.filter(lambda ex: ex['llama_quality'] == 1)
split_dataset_mq.save_to_disk(f"./data/senator_tweets/prepared-mid-quality-senator-tweets")