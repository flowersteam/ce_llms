import datasets
import re
import json

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

    # dataset was not prepared
    assert "title" in list(dataset_hf['train'].features)
    assert "body" in list(dataset_hf['train'].features)
    assert "text" not in list(dataset_hf['train'].features)

    dataset = dataset.map(
        lambda examples: {
            "text": [merge_title_and_text(ti, te) for ti, te in zip(examples["title"], examples["normalizedBody"])]
        },
        remove_columns=["title", "body", "normalizedBody", "n_tokens", "content", "content_len", "summary", "summary_len"], # there was for body only
        batched=True, desc="Merging title and body"
    )
    dataset = dataset.map(lambda examples: {"text_len": [len(t) for t in examples["text"]]}, batched=True, desc="Adding text_len")

    # clean dataset from links
    dataset = dataset.map(lambda examples: {"text": [re.sub(r'http\S+', '', t).rstrip() for t in examples['text']]}, batched=True, desc="Removing links")

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


# file_path = 'data/webis_reddit/clear-corpus-webis-tldr-17.json'
file_path = f'data/webis_reddit/350-minus-20-plus-clear-corpus-webis-tldr-17.json'
dataset_hf = datasets.load_dataset("json", data_files=[file_path])

dataset_hf = prepare_dataset(dataset_hf)
dataset_hf = remove_tldrs(dataset_hf)

n_min = 20
n_max = 200
dataset_hf = filter_posts_by_size(dataset_hf, n_min=n_min, n_max=n_max)


split_dataset = dataset_hf['train'].train_test_split(test_size=0.9, shuffle=True, seed=42)
split_dataset.save_to_disk(f"./data/webis/prepared-no-tldr-{n_max}-minus-{n_min}-plus-clear-corpus-webis-tldr-17")