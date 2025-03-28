import datasets
import numpy as np
from dataset_utils import *
n_min = 20
n_max = 200

def filter_posts_by_size(dataset, n_min, n_max):

    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    dataset = dataset.map(
        lambda batch: {"n_tokens": [len(tokenizer.encode(t)) for t in batch['text']]},
        batched=True, desc="Adding number of tokens", num_proc=1,
    )

    dataset = dataset.filter(
        lambda batch: [n_min < n_t < n_max for n_t in batch['n_tokens']],
        batched=True, desc=f"Filtering too long (>{n_max}) and too shorts (<{n_min}) posts.", num_proc=1
    )

    return dataset


def prepare_dataset(d, split_name):
    d = d.map(
        lambda titles, selftexts: {
            "text": [merge_title_and_text(ti, te) for ti, te in zip(titles, selftexts)],
            "subreddit": [split_name] * len(titles),
        },
        input_columns=["title", "selftext"],
        remove_columns=d.column_names,
        batched=True, desc="Merging title and body", num_proc=1
    )
    d = filter_posts_by_size(d, n_min=n_min, n_max=n_max)
    return d

def remove_tags(t):
    return t.replace("[deleted]", "").replace("[removed]", "").strip()


from datasets import load_dataset
split_names = [
    "tifu",
    "explainlikeimfive",
    "WritingPrompts",
    "changemyview",
    "LifeProTips",
    "todayilearned",
    "science",
    "askscience",
    "ifyoulikeblank",
    "Foodforthought",
    "IWantToLearn",
    "bestof",
    "IAmA",
    "socialskills",
    "relationship_advice",
    "philosophy",
    "YouShouldKnow",
    "history",
    "books",
    "Showerthoughts",
    "personalfinance",
    "buildapc",
    "EatCheapAndHealthy",
    "boardgames",
    "malefashionadvice",
    "femalefashionadvice",
    "scifi",
    "Fantasy",
    "Games",
    "bodyweightfitness",
    "SkincareAddiction",
    "podcasts",
    "suggestmeabook",
    "AskHistorians",
    "gaming",
    "DIY",
    "mildlyinteresting",
    "sports",
    "space",
    "gadgets",
    "Documentaries",
    "GetMotivated",
    "UpliftingNews",
    "technology",
    "Fitness",
    "travel",
    "lifehacks",
    "Damnthatsinteresting",
    "gardening",
    "programming"
]


def merge_title_and_text(title, text):
    post = ""

    if title is not None:
        post += title

    if title is not None and text is not None:
        post += "\n\n"

    if text is not None:
        post += text

    return post.strip()

# # step:1
# # per_sub_posts = 25_000
# # ds = []
# # for split_name in split_names:
# #     print(split_name)
# #
# #     for sample_size in [60_000, 100_000, np.inf]:
# #         print("---------------")
# #         split_d = datasets.load_dataset("HuggingFaceGECLM/REDDIT_submissions", split=f"{split_name}")
# #         split_d = split_d.shuffle()
# #         split_d = split_d.select(range(min(sample_size, len(split_d))))
# #         split_d_filtered = prepare_dataset(split_d, split_name)
# #
# #         if per_sub_posts < len(split_d_filtered):
# #             # enough datapoints
# #             break
# #
# #     split_d_filtered = split_d_filtered.shuffle().select(range(min(per_sub_posts, len(split_d_filtered))))
# #     ds.append(split_d_filtered)
# #
# # print("Concatenating")
# # dataset = datasets.concatenate_datasets(ds).shuffle()
# #
# # print("Saving")
# # file_path = f"./data/reddit_submissions/prepared-reddit-submissions-dataset-{n_max}-minus-{n_min}-plus"
# # dataset.save_to_disk(file_path)
# # print(f"Saved to: {file_path}")
# # exit()
#
# # step 2:
# dataset = datasets.load_from_disk(f"./data/reddit_submissions/prepared-reddit-submissions-dataset-{n_max}-minus-{n_min}-plus")
#
# from eval_utils import llama_quality
# dataset = dataset.map(
#     lambda examples: {"llama_quality": llama_quality(examples["text"])},
#     batched=True, desc="Computing quality", batch_size=10, num_proc=30
# )
# print("Saving")
# file_path = f"./data/reddit_submissions/prepared-reddit-submissions-dataset-with-qualities-{n_max}-minus-{n_min}-plus"
# dataset.save_to_disk(file_path)
# print(f"Saved to: {file_path}")
#


# clean [deleted][removed] tags

from eval_utils import llama_quality

# d_path = f"./data/reddit_submissions/prepared-reddit-submissions-dataset-with-qualities-{n_max}-minus-{n_min}-plus"
# dataset = datasets.load_from_disk(d_path)
# dataset = dataset.map(
#     lambda batch: {"text": [remove_tags(t) for t in batch['text']]},
#     batched=True, desc="Removing [deleted] and [removed] tags", num_proc=32,
# )
# dataset = dataset.map(
#     lambda examples: {"llama_quality": llama_quality(examples["text"])},
#     batched=True, desc="Computing quality", batch_size=10, num_proc=30
# )
# overwrite_to_disk(dataset, d_path)

###################
# add embeddings
###################
# if __name__ == "__main__":
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     from eval_utils import StellaEmbedder
#     file_path = f"./data/reddit_submissions/prepared-reddit-submissions-dataset-with-qualities-200-minus-20-plus"
#     d = load_from_disk(file_path)
#     print(d)
#     stella_embedder = StellaEmbedder(multigpu=True)
#     d = stella_embedder.add_embeddings_multigpu(d, batch_size=2048)
#     overwrite_to_disk(d, file_path)
#
# exit()


###################
# add qualities
###################
# from eval_utils import llama_quality_scale
# file_path = f"./data/reddit_submissions/prepared-reddit-submissions-dataset-with-qualities-200-minus-20-plus"
# split_dataset = datasets.load_from_disk(file_path)
#
# split_dataset = split_dataset.map(
#     lambda examples: {"llama_quality_scale": llama_quality_scale(examples["text"])},
#     batched=True, desc="Computing quality scale", batch_size=10, num_proc=60, load_from_cache_file=False
# )
#
# file_path = f"./data/reddit_submissions/prepared-reddit-submissions-dataset-with-qualities-200-minus-20-plus"
# overwrite_to_disk(split_dataset, file_path)


####################
# Quality datasets
####################
file_path = f"./data/reddit_submissions/prepared-reddit-submissions-dataset-with-qualities-200-minus-20-plus"
dataset = datasets.load_from_disk(file_path)
dataset = dataset.filter(lambda ex: ex['llama_quality_scale'] is not None, num_proc=64, load_from_cache_file=False)

qualities = np.array(dataset['llama_quality_scale'])
print("Separating")
for q in [20, 40, 60, 80]:
    d = dataset.select(np.where(qualities == q)[0])
    print("Q:", q)
    print("Size: ", len(d))
    # print("Avg toks", np.mean(d['n_tokens']))
    # d.save_to_disk(file_path + f"_llama_scale_{q}")

exit()

from IPython import embed; embed();

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


# # Step 4: high q and mid q
# dataset = datasets.load_from_disk(f"./data/reddit_submissions/prepared-reddit-submissions-dataset-with-qualities-{n_max}-minus-{n_min}-plus")
# dataset_hq = dataset.filter(lambda ex: ex['llama_quality'] == 2, num_proc=32, load_from_cache_file=False)
# file_path = f"./data/reddit_submissions/prepared-reddit-submissions-dataset-high-quality-{n_max}-minus-{n_min}-plus"
# dataset_hq.save_to_disk(file_path)
# print(f"Saved to: {file_path}")
#
#
# dataset_mq = dataset.filter(lambda ex: ex['llama_quality'] == 1, num_proc=32, load_from_cache_file=False)
# file_path = f"./data/reddit_submissions/prepared-reddit-submissions-dataset-mid-quality-{n_max}-minus-{n_min}-plus"
# dataset_mq.save_to_disk(file_path)
# print(f"Saved to: {file_path}")

# low diversity dataset
# d_path = f"./data/reddit_submissions/prepared-reddit-submissions-dataset-low-diversity-{n_max}-minus-{n_min}-plus"
# dataset = datasets.load_from_disk(d_path)
# # dataset = dataset.map(
# #     lambda batch: {"text": [remove_tags(t) for t in batch['text']]},
# #     batched=True, desc="Removing [deleted] and [removed] tags", num_proc=32,
# # )
# dataset = dataset.map(
#     lambda examples: {"llama_quality": llama_quality(examples["text"])},
#     batched=True, desc="Computing quality", batch_size=10, num_proc=30
# )
# overwrite_to_disk(dataset, d_path)