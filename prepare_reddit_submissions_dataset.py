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

# # step: 3 diversity: take half of subreddits
# ld_subreddit = "gaming"
# dataset_ld = datasets.load_dataset("HuggingFaceGECLM/REDDIT_submissions", split=ld_subreddit)
# dataset_ld = prepare_dataset(dataset_ld, split_name=ld_subreddit)
# dataset_ld = dataset_ld.shuffle()
#
# file_path = f"./data/reddit_submissions/prepared-reddit-submissions-dataset-low-diversity-{n_max}-minus-{n_min}-plus"
# dataset_ld.save_to_disk(file_path)
# print(f"Saved to: {file_path}")


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

print("prepare reddit test")
import time
time.sleep(60)
print("prepare reddit test done")
