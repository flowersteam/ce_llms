import os
import warnings
from collections import Counter
from eval_utils import calculate_ttr

import torch.cuda
import termcolor

hostname = os.uname()[1]
if hostname == "PTB-09003439":
    hf_cache_dir = "/home/flowers-user/.cache/huggingface"
else:
    hf_cache_dir = "/gpfsscratch/rech/imi/utu57ed/.cache/huggingface"
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

os.environ['HF_HOME'] = hf_cache_dir

import time
import numpy as np

from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset

from dataset_utils import *


def secs_2_hms(s):
    minutes, seconds = divmod(s, 60)
    hours, minutes = divmod(minutes, 60)
    return hours, minutes, seconds


# calculate metrics for FULL datasets


# Load FRACTIONS of datasets
dataset, labels, feat_sentiment = load_reddit_dataset(cache_dir=hf_cache_dir, load_frac=0.3)
# dataset, labels, feat_sentiment = load_twitter_dataset(cache_dir=hf_cache_dir, load_frac=0.01)
# dataset, labels, feat_sentiment = load_twitter_dataset(cache_dir=hf_cache_dir, load_frac=1)

print(f"Dataset size: {len(dataset)}")
for l, n in Counter(labels).items():
    print(f"\t{feat_sentiment.int2str(l)} - {n}")

dataset = dataset.map(lambda examples: {"ttr": [calculate_ttr(tx) for tx in examples['text']]}, batched=True)


# estimate ttr for lists of 128 tweets
joint_ttr = []
for i in range(100):
    rand_inds = np.random.randint(0, len(dataset), 128)
    t = calculate_ttr(" ".join(dataset[rand_inds]['text']))
    joint_ttr.append(t)

joint_ttr = np.mean(joint_ttr)

print("Dataset stats")
print("N:", len(dataset))
print("Joint TTR:", joint_ttr)
print("Mean TTR:", np.mean(dataset['ttr']))
print("Mean TTR:", np.min(dataset['ttr']))
exit()


# Prepare the Pipeline

# model = "gpt"
model = "zsc"

metric = "post"

if model == "gpt":
    from openai import OpenAI, RateLimitError
    import tiktoken
    import json

    from tenacity import (
        retry,
        stop_after_attempt,
        wait_random_exponential,
    )  # for exponential backoff


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def completions_with_backoff(**kwargs):
        return client.chat.completions.create(**kwargs)

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    if metric == "pol_or":
        # classification of POLITICAL ORIENTATION
        candidate_label_2_label_map = {
            # "democrat": "Liberal",
            # "republican": "Conservative",
            "liberal": "Liberal",
            "conservative": "Conservative",
        }
        candidate_labels = list(candidate_label_2_label_map.keys())
        def create_prompt(candidate_labels):
            prompt = f"Classify the political orientation of the author of the following text, i.e. was this text written by a {' or a '.join(candidate_labels)}?\n\n{text}"
            prompt += f"\n\nReply with a json with the following format: " \
                       r"{'prediction': '"+"/".join(candidate_labels)+r"'}"

            return prompt
    elif metric == "grammar":
        # classification of GRAMMAR
        candidate_labels = ["true", "false"]
        candidate_label_2_label_map = dict(zip(candidate_labels, candidate_labels))

        def create_prompt(candidate_labels):
            prompt = f"You are presented with a text and you have to rate if this text is grammatically correct or not. Here is the text:\n'''\n{text}\n'''"
            prompt += f"\n\nReply with a json with the following format: " \
                      r"{'prediction': '" + "/".join(candidate_labels) + r"'}"

            return prompt

    elif metric == "post":
        candidate_labels = ["true", "false"]
        candidate_label_2_label_map = dict(zip(candidate_labels, candidate_labels))

        def create_prompt(candidate_labels):
            prompt = f"You are presented with a text and you have to rate if this text a tweet or not. Here is the text:\n'''\n{text}\n'''"
            prompt += f"\n\nReply with a json with the following format: " \
                      r"{'prediction': '" + "/".join(candidate_labels) + r"'}"

            return prompt


    print("Evaluating")
    corrs = []
    n_cls_zero = []

    start_time = time.time()
    batch_start_time = start_time

    # model_name = "gpt-4-0125-preview"
    model_name = "gpt-3.5-turbo-0125"

    print("Model name: ", model_name)

    if model_name == "gpt-3.5-turbo-0125":
        price_1M = 0.5
    elif model_name == "gpt-4-0125-preview":
        price_1M = 10
    else:
        raise ValueError(f"Unknown token price for model {model_name}.")

    total_tokens = 0
    tokenizer = tiktoken.get_encoding("cl100k_base")
    n_invalid_resp = 0

    for i, (text, lab) in enumerate(zip(dataset["text"], dataset["Political Lean"])):

        content = create_prompt(candidate_labels)

        num_tokens = len(tokenizer.encode(content))
        total_tokens += num_tokens

        if i == 0:
            print(f"Content:\n{content}")

        response = completions_with_backoff(
            model=model_name,
            temperature=0.00001,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
        )
        try:
            pred_ = json.loads(response.choices[0].message.content).get('prediction', None)
        except:
            pred_ = "Bad Json"

        if pred_ in candidate_label_2_label_map:
            pred = candidate_label_2_label_map[pred_]
            correct = pred == feat_sentiment.int2str(lab)

        else:
            print(f"Invalid pred={pred_} in response.")
            n_invalid_resp += 1
            correct = False

        corrs.append(correct)

        cls_zero = pred_ == candidate_labels[0]
        n_cls_zero.append(cls_zero)

        log_step = 1

        if i % log_step == 0 and i > 0:
            batch_end_time = time.time()

            print(f"I {i}/{len(dataset)}")
            print("\tCurrent Acc: ", np.mean(corrs))
            print(f"\tCurrent % {candidate_labels[0]}: ", np.mean(n_cls_zero))

            log_time = batch_end_time - batch_start_time
            batch_start_time = batch_end_time
            example_time = log_time / log_step

            print("\ts/example:", example_time)

            avg_example_time = (batch_end_time - start_time) / (i+1)

            time_passed = batch_end_time - start_time
            hours, minutes, seconds = secs_2_hms(time_passed)
            print("\tTime passed: %d:%02d:%02d" % (hours, minutes, seconds))

            # ETA
            n_left = len(dataset) - i
            eta = n_left * avg_example_time

            hours, minutes, seconds = secs_2_hms(eta)
            print("\tETA: %d:%02d:%02d" % (hours, minutes, seconds))

            pc_invalid = n_invalid_resp/(i+1)
            print(f"\tN invalid: {n_invalid_resp}")
            print(f"\t% invalid: {pc_invalid:.2f}")

            print(f"\tCurrent input tokens: {total_tokens} ({(total_tokens/1_000_000)*price_1M:.2f})$")

            n_tokens_per_example = total_tokens/(i+1)
            total_tokens_estimate = n_tokens_per_example*len(dataset)
            print(f"\tTotal input tokens estimate: {total_tokens_estimate:.0f} ({(total_tokens_estimate/1_000_000)*price_1M:.2f})$")


elif model == "zsc":

    if metric == "pol_or":
        hypothesis_template = "This is a post from a {}."
        candidate_label_2_label_map = {
            "democrat": "Liberal",
            "republican": "Conservative",
        }
        candidate_labels = list(candidate_label_2_label_map.keys())

    elif metric == "grammar":
        hypothesis_template = "This text is grammatically {}."
        candidate_labels = ["correct", "incorrect"]
        candidate_label_2_label_map = dict(zip(candidate_labels, candidate_labels))

    elif metric == "post":
        hypothesis_template = "This text {}."
        # candidate_labels = ["is a tweet", "is not a tweet"]
        candidate_labels = ["contains a tweet", "does not contain a tweet"]
        candidate_label_2_label_map = dict(zip(candidate_labels, candidate_labels))

    # hypothesis_template = "This text has a {} sentiment."
    # candidate_labels = ["positive", "negative"]
    # candidate_labels = ["Liberal", "Conservative"]
    # candidate_label_2_label_map = dict(zip(candidate_labels, candidate_labels))
    # hypothesis_template = "This is a post from a {}."
    # candidate_label_2_label_map = None

    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(model="facebook/bart-large-mnli", task="zero-shot-classification", batch_size=batch_size, device=device)

    print("Evaluating")
    corrs = []
    n_cls_zero = []

    start_time = time.time()
    batch_start_time = start_time

    if not set(map(feat_sentiment.int2str, labels)) == set(map(candidate_label_2_label_map.get, candidate_labels)):
        warnings.warn(f"Candidate labels {candidate_labels} do not match ground truth labels.")

    print("Hypothesis template: ", hypothesis_template)
    print("Candidate label:", candidate_labels)

    results = pipe(KeyDataset(dataset, "text"), candidate_labels=candidate_labels, hypothesis_template=hypothesis_template, batch_size=batch_size, device=device)

    log_step = 2*batch_size

    for i, (r, lab) in enumerate(zip(results, labels)):

        pred_ = r['labels'][np.argmax(r['scores'])]

        pred = candidate_label_2_label_map[pred_]

        correct = pred == feat_sentiment.int2str(lab)
        corrs.append(correct)

        cls_zero = pred_ == candidate_labels[0]
        n_cls_zero.append(cls_zero)

        if i % log_step == 0 and i > 0:

            print(f"I {i}/{len(dataset)}")
            print("\tCurrent Acc: ", np.mean(corrs))
            print(f"\tCurrent % {candidate_labels[0]}: ", np.mean(n_cls_zero))
            batch_end_time = time.time()
            log_time = batch_end_time - batch_start_time
            batch_start_time = batch_end_time
            example_time = log_time / log_step

            print("\ts/example:", example_time)
            print("\ts/batch:", example_time*batch_size)

            avg_example_time = (batch_end_time - start_time) / (i+1)

            time_passed = batch_end_time - start_time
            hours, minutes, seconds = secs_2_hms(time_passed)
            print("\tTime passed: %d:%02d:%02d" % (hours, minutes, seconds))
            # ETA
            n_left = len(dataset) - i
            eta = n_left * avg_example_time

            hours, minutes, seconds = secs_2_hms(eta)
            print("\tETA: %d:%02d:%02d" % (hours, minutes, seconds))

else:
    raise NotImplementedError(f"Unimplemented model {model}.")

print("-- Final results --")
print("Accuracy:", np.mean(corrs))
print(f"{candidate_labels[0]}: ", np.mean(n_cls_zero))

end_time = time.time()
elapsed_time = end_time-start_time
print("Time:", elapsed_time)