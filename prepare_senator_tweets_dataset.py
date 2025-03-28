import datasets
import numpy as np

# from eval_utils import llama_quality
# from dataset_utils import *
from datasets import load_dataset

# dataset = load_dataset("m-newhauser/senator-tweets")
#
# dataset = dataset.map(remove_links, batched=True, desc="Removing links", load_from_cache_file=False)
# dataset = dataset.filter(lambda examples: [len(word_tokenize(t)) > 10 for t in examples['text']], batched=True,
#                          load_from_cache_file=False)
# dataset = dataset.remove_columns(["embeddings"])
#
# dataset['train'] = dataset['train'].map(
#     lambda examples: {"llama_quality": llama_quality(examples["text"])},
#     batched=True, desc="Computing quality", batch_size=10, num_proc=30
# )
#
# dataset['test'] = dataset['test'].map(
#     lambda examples: {"llama_quality": llama_quality(examples["text"])},
#     batched=True, desc="Computing quality", batch_size=10, num_proc=30
# )
#
# dataset.save_to_disk(f"./data/senator_tweets/prepared-senator-tweets")

###################
# add embeddings
###################
# multigpu sentence transformers must be in __main__
# if __name__ == "__main__":
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     from eval_utils import StellaEmbedder
#     from dataset_utils import overwrite_to_disk
#     file_path = "./data/senator_tweets/prepared-senator-tweets-qualities"
#     split_dataset = datasets.load_from_disk(file_path)
#     print(split_dataset)
#     stella_embedder = StellaEmbedder(multigpu=True)
#     split_dataset['test'] = stella_embedder.add_embeddings_multigpu(split_dataset['test'], batch_size=2048)
#     split_dataset['train'] = stella_embedder.add_embeddings_multigpu(split_dataset['train'], batch_size=2048)
#     overwrite_to_disk(split_dataset, file_path)
#     print(f"Saved to: {file_path}")
#     exit()

###################
# add quality_scale
###################
# from eval_utils import llama_quality_scale
# split_dataset = datasets.load_from_disk(f"./data/senator_tweets/prepared-senator-tweets")
# print(split_dataset)
#
# split_dataset = split_dataset.map(
#     lambda examples: {"llama_quality_scale": llama_quality_scale(examples["text"])},
#     batched=True, desc="Computing quality scale", batch_size=10, num_proc=60, load_from_cache_file=False
# )
# file_path = "./data/senator_tweets/prepared-senator-tweets-qualities"
# split_dataset.save_to_disk(file_path)


# ###################
# # Quality datasets
# ###################
# file_path = "./data/senator_tweets/prepared-senator-tweets-qualities"
# split_dataset = datasets.load_from_disk(file_path)
#
# dataset = datasets.concatenate_datasets([split_dataset['train'], split_dataset['test']])
# dataset = dataset.filter(lambda ex: ex['llama_quality_scale'] is not None, num_proc=64, load_from_cache_file=False)
#
# qualities = np.array(dataset['llama_quality_scale'])
# print("Separating")
#
# # 80
# d = dataset.select(np.where(qualities == 80)[0])
# print("Q:", 80)
# print("Size: ", len(d))
# d.save_to_disk(file_path + f"_llama_scale_80")
#
# # 60ish
# d = dataset.select(np.where((qualities == 60) | (qualities == 40))[0])
# q = int(np.mean(d['llama_quality_scale']))
# print("Q:", q)
# print("Size: ", len(d))
# d.save_to_disk(file_path + f"_llama_scale_{q}")

####################
# Length datasets
####################
file_path = "./data/senator_tweets/prepared-senator-tweets-qualities"
split_dataset = datasets.load_from_disk(file_path)
dataset = datasets.concatenate_datasets([split_dataset['train'], split_dataset['test']])

dataset = dataset.map(
    lambda examples: {"text_len": [len(t) for t in examples["text"]]},
    batched=True, desc="Adding len", batch_size=10, num_proc=60, load_from_cache_file=False
)
lengths = np.array(dataset['text_len'])

sort_indices = np.argsort(lengths)
# chunk_size = len(sort_indices) // 3
chunk_size = len(sort_indices) // 2

short_d = dataset.select(sort_indices[:chunk_size])
# medium_d = dataset.select(sort_indices[chunk_size:2 * chunk_size])
# long_d = dataset.select(sort_indices[2 * chunk_size:])
long_d = dataset.select(sort_indices[chunk_size:])

print("Short: ", np.mean(short_d['text_len']))
# print("Medium: ", np.mean(medium_d['text_len']))
print("Long: ", np.mean(long_d['text_len']))

short_d.save_to_disk(file_path + "_short")
# medium_d.save_to_disk(file_path + "_medium")
long_d.save_to_disk(file_path + "_long")

#
# dataset = datasets.concatenate_datasets([split_dataset['train'], split_dataset['test']])
# dataset = dataset.filter(lambda ex: ex['llama_quality_scale'] is not None, num_proc=64, load_from_cache_file=False)
#
# print("Sorting")
# sort_indices = np.argsort(dataset['llama_quality_scale'])
# half_size = int(np.floor(len(dataset) // 2))
#
# dataset_lq = dataset.select(sort_indices[:half_size]).shuffle()
# print("LQ len: {}, Q: {}".format(len(dataset_lq['llama_quality_scale']), np.mean(dataset_lq['llama_quality_scale'])))
#
# dataset_hq = dataset.select(sort_indices[half_size:]).shuffle()
# print("HQ len: {}, Q: {}".format(len(dataset_hq['llama_quality_scale']), np.mean(dataset_hq['llama_quality_scale'])))
#
# dataset_lq.save_to_disk(file_path+"_llama_scale_lq")
# dataset_hq.save_to_disk(file_path+"_llama_scale_hq")
# exit()
