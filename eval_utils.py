import numpy as np
import warnings
import time
import pickle
import json
from termcolor import cprint

from nltk import word_tokenize, sent_tokenize


from transformers import AutoModelForCausalLM, pipeline

from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from Levenshtein import ratio
from scipy.spatial.distance import pdist

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from detoxify import Detoxify

from langkit import extract, light_metrics
import pandas as pd

import torch
from transformers import BertTokenizer, BertModel

def get_words(text):
    return word_tokenize(text)

def num_words(text):
    return len(get_words(text))

def num_unique_words(text):
    return len(set(get_words(text)))

from openai import AzureOpenAI, OpenAI
import os
client=None


def compute_quick_metrics(input_d):
    results = dict()
    results['ttr'] = [calculate_ttr(tx) for tx in input_d['text']]
    results['mttr'] = calculate_ttr(" ".join(input_d['text']))

    results['n_chars_per_post'] = [len(t) for t in input_d['text']]

    words_per_post = [get_words(t) for t in input_d['text']]
    results['n_words_per_post'] = [len(wrds) for wrds in words_per_post]
    results['n_unique_words_per_post'] = [len(set(wrds)) for wrds in words_per_post]

    all_words = get_words(" ".join(input_d['text']))
    results[f'n_words_total'] = len(all_words)
    results[f'n_unique_words_total'] = len(set(all_words))

    results['n_unique_posts'] = len(set(input_d['text']))
    results['pc_unique_posts'] = len(set(input_d['text'])) / len(input_d['text'])
    results['aggregate_reading_level'] = aggregate_reading_level(input_d['text'])

    return results


def llama_pol_lean(texts):
    scores = []
    global client
    if client is None:
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )

    for t_i, text in enumerate(texts):
        if t_i % 50 == 0 and t_i > 0:
            print(f"llama quality: [{t_i}/{len(texts)}]")

        prompt = "You are a political lean judge.\n" + \
                 "Judge the political lean of a given internet post and reply ONLY with a letter L, N, or R.\n" + \
                 "\tL - left leaning\n\tN - neutral leaning\n\tR - right leaning\n\n" + \
                 f"Here is the post: {text}\n\n" + \
                 "Reply ONLY with the letter (L,N,R). DO NO reply with a sentence."

        # completion = client.beta.chat.completions.parse(
        completion = client.chat.completions.create(
            model="llama",
            temperature=0.01,
            messages=[
                {"role": "system", "content": prompt},
            ],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )

        score = None
        for tlp in completion.choices[0].logprobs.content[0].top_logprobs:
            token = tlp.token
            if token in ["L", "N", "R"]:
                score = {"L": -1, "N": 0, "R": 1}[token]
                break

        scores.append(score)

    return scores




def llama_is_english(texts):
    scores = []
    total_tokens = 0
    global client
    if client is None:
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )

    for t_i, text in enumerate(texts):
        if t_i % 50 == 0 and t_i > 0:
            print(f"llama is_english: [{t_i}/{len(texts)}]")

        prompt = "You will receive a text and have to judge if it's in english and reply ONLY with a integer from 0 or 1.\n" + \
                 "\t0 - text is not in english \n\t1 - text is in english\n\n" + \
                 f"Here is the post: {text}\n\n" + \
                 "Reply ONLY with the integer (0,1). DO NO reply with text."

        # completion = client.beta.chat.completions.parse(
        completion = client.chat.completions.create(
            model="llama",
            temperature=0.01,
            messages=[
                {"role": "system", "content": prompt},
            ],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )

        # score = None
        for tlp in completion.choices[0].logprobs.content[0].top_logprobs:
            token = tlp.token
            if token.isdigit():
                score = int(token)
                break

        scores.append(score)

    return scores


def llama_quality(texts):
    
    scores = []
    total_tokens = 0
    global client
    if client is None:
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )

    for t_i, text in enumerate(texts):
        if t_i % 50 == 0 and t_i > 0:
            print(f"llama quality: [{t_i}/{len(texts)}]")

        prompt = "You are a text quality judge.\n" + \
                 "When judging the quality pay attention to:\n" + \
                 "\t- broken/cut-off text\n" + \
                 "\t- very repetitive text\n" + \
                 "\t- grammar\n" + \
                 "\t- semantic plausability\n" + \
                 "\t- lexical complexity\n" + \
                 "\nJudge the quality of a given internet post and reply ONLY with a integer from 0-2.\n" + \
                 "\t0 - low quality\n\t1 - intermediate quality\n\t2 - good quality\n\n" + \
                 f"Here is the post: {text}\n\n" + \
                 "Reply ONLY with the score (0,1,2). DO NO reply with text."

        # completion = client.beta.chat.completions.parse(
        completion = client.chat.completions.create(
            model="llama",
            temperature=0.01,
            messages=[
                {"role": "system", "content": prompt},
            ],
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )

        score = None
        for tlp in completion.choices[0].logprobs.content[0].top_logprobs:
            token = tlp.token
            if token.isdigit():
                score = int(token)
                break

        scores.append(score)

    return scores

def gpt4o_quality(texts):

    scores = []
    total_tokens = 0
    global client
    if client is None:
        client = AzureOpenAI(
            azure_endpoint=os.getenv(f"AZURE_OPENAI_ENDPOINT_gpt_4o_mini"),
            api_key=os.getenv("AZURE_OPENAI_KEY_gpt_4o_mini"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION_gpt_4o_mini"),
        )

    for t_i, text in enumerate(texts):
        if t_i % 10 == 0:
            print(f"gpt4o quality: [{t_i}/{len(texts)}] -> tokens used: {total_tokens} money used: {(total_tokens/1_000_000)*0.15}")

        prompt = "You are a text quality judge.\n" + \
                 "When judging the quality pay attention to:\n" + \
                 "\t- broken/cut-off text\n" + \
                 "\t- very repetitive text\n" + \
                 "\t- grammar\n" + \
                 "\t- semantic plausability\n" + \
                 "\t- lexical complexity\n" + \
                 "\nJudge the quality of a given internet post and reply ONLY with a integer from 0-2.\n" + \
                 "\t0 - low quality\n\t1 - intermediate quality\n\t2 - good quality\n\n" + \
                 f"Here is the post: {text}"

        # completion = client.beta.chat.completions.parse(
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0000001,
            messages=[
                {"role": "system", "content": prompt},
            ],
            max_tokens=1,
        )
        score = completion.choices[0].message.content
        scores.append(score)
        total_tokens += completion.usage.total_tokens

    return scores, total_tokens



def aggregate_reading_level(texts):
    df = pd.DataFrame({'response': texts})
    llm_schema = light_metrics.init()
    edf = extract(df, schema=llm_schema)
    aggregate_reading_levels = list(edf[f"response.aggregate_reading_level"])
    return aggregate_reading_levels


## Not used currently
def get_positivity(text):
    sid = SentimentIntensityAnalyzer()    
    return sid.polarity_scores(text)['compound']

def get_positivites(texts):
    return [get_positivity(tx) for tx in texts]

# Predict toxicity using library from https://pypi.org/project/detoxify/
toxicity_nlp = None

def get_toxicity_batch(texts, batch_size=256):
    global toxicity_nlp
    if toxicity_nlp is None:
        toxicity_nlp = Detoxify('original', device="cuda")

    out = []
    for i in range(0, len(texts), batch_size):
        print(f"{i}/{len(texts)}")
        out += toxicity_nlp.predict(texts[i:i+batch_size])['toxicity']
    return out


gibberish_detector = None

def get_gibberish_scores(texts, batch_size=1024):
    global gibberish_detector
    if gibberish_detector is None:
        gibberish_detector = pipeline("text-classification", model="madhurjindal/autonlp-Gibberish-Detector-492513457")

    # get sentences
    all_sentences = []
    sentence_mapping = {}
    for text_i, text in enumerate(texts):
        sentences = sent_tokenize(text)
        sentence_mapping[text_i] = list(range(len(all_sentences), len(all_sentences) + len(sentences)))
        all_sentences.extend(sentences)

    # Get gibberish detection scores for all sentences in a single batch
    results = gibberish_detector(all_sentences, batch_size=batch_size, truncation=True)
    sentence_scores = [gibberish_detector.model.config.label2id[r['label']] for r in results]

    # Group scores back into their original texts
    texts_scores = []
    for text_i in range(len(texts)):
        text_scores = [sentence_scores[j] for j in sentence_mapping[text_i]]
        averaged_score = sum(text_scores) / len(text_scores)
        texts_scores.append(averaged_score)

    assert len(texts_scores) == len(texts)

    return texts_scores




## Predict political bias using pretrained model from https://huggingface.co/premsa/political-bias-prediction-allsides-mDeBERTa
# -1 = Lean Left, 0 = Center, 1 = Lean Right
# LABEL_0 (-1) = Left , LABEL_1 = Center (0), LABEL_2 (1) = Right
# political_model = AutoModelForSequenceClassification.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa")
# political_tokenizer = AutoTokenizer.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa")
# political_nlp = pipeline("text-classification", model=political_model, tokenizer=political_tokenizer)


# zero-shot classifier
political_nlp = None
hypothesis_template = "This is a post from a {}."
candidate_labels = ["democrat", "republican"]


def get_political_lean_batch(texts):
    global political_nlp
    if political_nlp is None:
        political_nlp = pipeline(model="facebook/bart-large-mnli", task="zero-shot-classification", device="cuda")

    def data(d):
        for i in range(len(d)):
            yield d[i]

    batch_size = 256

    labels = []
    scores = []

    s = time.time()
    for i, o in enumerate(political_nlp(
            data(texts),
            candidate_labels=candidate_labels, hypothesis_template=hypothesis_template,
            batch_size=batch_size)
    ):
        if i % batch_size == 0:
            print(f"[{i}/{len(texts)}]")

        # (Left) -1 , 0, 1 (Right)
        l = o['labels'][0]
        l_ = -1 if l == "democrat" else 1 if l == "republican" else 0
        labels.append(l_)
        scores.append((o['scores'][0]-0.5)*l_)

    # for i, o in enumerate(political_nlp(data(texts), batch_size=batch_size)):
    #     if i % batch_size == 0:
    #         print(f"[{i}/{len(texts)}]")
    #
    #     l = o['label']
    #     l_ = -1 if l == "LABEL_0" else 1 if l == "LABEL_2" else 0
    #     labels.append(l_)
    #     scores.append(o['score']*l_)

    elapsed_time = time.time() - s
    cprint(f"Elapsed Time: {elapsed_time}", "blue")

    return labels, scores


def compute_normalized_levenshtein_diversity(texts):
    return 1 - np.mean(pdist(np.array(texts).reshape((-1, 1)), metric=ratio))


def calculate_ttr(text):
    # unique words/all words
    words = word_tokenize(text)
    if len(words) == 0:
        ttr = 0
    else:
        ttr = len(set(words)) / len(words)
    return ttr


def evaluate_generations(generated_texts, verbose=False):

    joint_ttr = calculate_ttr(" ".join(generated_texts))
    ttrs = np.array([calculate_ttr(tx) for tx in generated_texts])
    ttr = np.mean(ttrs)

    if verbose:
        # show k-worst generations
        k = 4
        sort_inds = ttrs.argsort()
        print(f"Worst {k} generations:")
        bot_gens = [generated_texts[i] for i in sort_inds[:k]]
        for g in bot_gens:
            print(f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n{g}\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

        print(f"Best {k} generations:")
        top_gens = [generated_texts[i] for i in sort_inds[-k:]]
        for g in top_gens:
            print(f"\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n{g}\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    logs = {
        "TTR": ttr,
        "JointTTR": joint_ttr,
        "per_generation_metrics": {
            "TTR": list(ttrs),
        }
    }

    return logs

def compute_var_diveristy(embs):
    return np.array(embs).var(axis=0).mean()

def compute_cos_diveristy(embs):
    dist_matrix = pairwise_distances(embs, metric="cosine")
    print("distances")
    return dist_matrix[np.triu_indices(len(dist_matrix), k=1)].mean()

def fit_logreg(embs_1, embs_2, max_iter=1):
    X = np.vstack((embs_1, embs_2))
    y = [0]*embs_1.shape[0] + [1]*embs_2.shape[0]

    clf = LogisticRegression(max_iter=max_iter).fit(X, y)
    preds = clf.predict_proba(X)

    y_ = preds.argmax(axis=1)
    acc = np.mean(y == y_)

    loss = log_loss(y, preds)
    return loss, acc


from torch.nn import CrossEntropyLoss
from evaluate import logging


class Perplexity:
    def __init__(self, model_id="mistralai/Mistral-7B-v0.1", model_args=None):

        if model_args is None:
            model_args = {}

        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_args).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.device = self.model.device

    def evaluate(
        self, predictions, batch_size: int = 16, add_start_token: bool = True, add_end_token: bool = False, max_length=None,
        response_template=None
    ):

        # if batch_size > 1 (which generally leads to padding being required), and
        # if there is not an already assigned pad_token, assign an existing
        # special token to also be the padding token
        if self.tokenizer.pad_token is None and batch_size > 1:
            existing_special_tokens = list(self.tokenizer.special_tokens_map_extended.values())
            # check that the model already has at least one special token defined
            assert (
                len(existing_special_tokens) > 0
            ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
            # assign one of the special tokens to also be the pad token
            self.tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

        max_tokenized_len = max_length
        if add_start_token and max_length:
            # leave room for <BOS> token to be added:
            assert (
                self.tokenizer.bos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_tokenized_len - 1

        if add_end_token and max_length:
            # leave room for <EOS> token to be added:
            assert (
                    self.tokenizer.eos_token is not None
            ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
            max_tokenized_len = max_tokenized_len - 1

        if add_end_token:
            predictions = [p+self.tokenizer.eos_token for p in predictions]

        ppls = []
        ces = []
        loss_fct = CrossEntropyLoss(reduction="none")

        for start_index in logging.tqdm(range(0, len(predictions), batch_size), desc=f"Perplexity ({self.model_id})"):
            end_index = min(start_index + batch_size, len(predictions))

            predictions_batch = predictions[start_index:end_index]

            encodings_batch = self.tokenizer(
                predictions_batch,
                add_special_tokens=False,
                padding=True,
                truncation=True if max_tokenized_len else False,
                max_length=max_tokenized_len,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)

            encoded_texts_batch = encodings_batch["input_ids"]
            attn_masks_batch = encodings_batch["attention_mask"]

            # check that each input is long enough:
            if add_start_token:
                assert torch.all(torch.ge(attn_masks_batch.sum(1), 1)), "Each input text must be at least one token long."
            else:
                assert torch.all(
                    torch.ge(attn_masks_batch.sum(1), 2)
                ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

            # encoded_batch = encoded_texts[start_index:end_index]
            # attn_mask = attn_masks[start_index:end_index]

            if add_start_token:
                bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * encoded_texts_batch.size(dim=0)).to(self.device)
                encoded_texts_batch = torch.cat([bos_tokens_tensor, encoded_texts_batch], dim=1)
                attn_masks_batch = torch.cat(
                    [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(self.device), attn_masks_batch], dim=1
                )

            labels = encoded_texts_batch

            with torch.no_grad():
                out_logits = self.model(encoded_texts_batch, attention_mask=attn_masks_batch).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_masks_batch[..., 1:].contiguous()

            if response_template is not None:
                response_mask = []
                for i in range(len(shift_labels)):
                    prefix_tokens = self.tokenizer(
                        predictions_batch[i].split(response_template)[0] + response_template,
                        add_special_tokens=False,
                        return_attention_mask=False
                    )['input_ids']
                    shift_prefix_tokens = prefix_tokens[1:]

                    mask_len = len(shift_prefix_tokens)
                    response_mask.append(np.array([0]*mask_len + [1]*(len(shift_labels[i])-mask_len)))

                response_mask = torch.tensor(np.array(response_mask)).to(self.device)
                shift_attention_mask_batch *= response_mask.to(self.device)

            ce_batch = (
                    loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch
            ).sum(1) / shift_attention_mask_batch.sum(1)

            perplexity_batch = torch.exp(ce_batch)

            ces += ce_batch.tolist()
            ppls += perplexity_batch.tolist()

        return {"perplexities": ppls, "mean_perplexity": np.mean(ppls), "cross_entropies": ces, "mean_cross_entropy": np.mean(ces)}


from transformers import AutoTokenizer, AutoModelForMaskedLM
class ModernBertEmbedder:

    def __init__(self, CLS_token=True):

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        model_id = "answerdotai/ModernBERT-large"
        self.CLS_token = CLS_token
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embedding_name = "modernbert"
        self.embedding_column_name = "modernbert_embeddings"

        print("Loading ModernBert")
        self.model = AutoModelForMaskedLM.from_pretrained(model_id, device_map=self.device, torch_dtype=torch.bfloat16)
        print("ModernBert loaded")

    def add_embeddings(self, dataset, batch_size=256):
        def embed_text(examples):
            with torch.no_grad():
                inputs = self.tokenizer(examples['text'], return_tensors='pt', padding=True).to(self.device)
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden_states = outputs.hidden_states[-1]
                last_hidden_states = last_hidden_states.to(dtype=torch.float32).cpu().numpy()

            if self.CLS_token:
                embeddings = last_hidden_states[:, 0, :]  # [CLS] token representation
            else:
                embeddings = last_hidden_states.mean(axis=1)

            assert not np.isnan(embeddings).any()

            return {self.embedding_column_name: list(embeddings)}

        return dataset.map(embed_text, batched=True, batch_size=batch_size, desc="Embedding with ModernBert", load_from_cache_file=False)


class BertEmbedder:

    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.embedding_name = "bert"
        self.embedding_column_name = "bert_embeddings"

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_built():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print("Loading bert")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased", device_map=self.device).eval()
        print("bert loaded")

    def add_embeddings(self, dataset):

        def embed_text(examples):
            encoded_input = self.bert_tokenizer(examples['text'], return_tensors='pt', padding=True, truncation=True)
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            output = self.bert_model(**encoded_input)
            embeddigs = output['last_hidden_state'][:, 0, :]  # we take the representation of the [CLS] token
            if self.device == torch.device("mps"):
                torch.mps.empty_cache()
            elif self.device == torch.device("cuda"):
                torch.cuda.empty_cache()
            return {self.embedding_column_name: list(embeddigs)}

        return dataset.map(embed_text, batched=True, batch_size=128, desc="Embedding with bert", load_from_cache_file=False)


try:
    from sentence_transformers import SentenceTransformer
except:
    warnings.warn("SentenceTransformer not installed.")


class MiniLMEmbedder:
    def __init__(self, device="cuda"):
        # load model with tokenizer
        print("Loading MiniLM")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        self.embedding_name = "minilm"
        self.embedding_column_name = f"{self.embedding_name}_embeddings"
        print("minilm loaded")

    def add_embeddings(self, dataset, batch_size=32):
        all_embeddings = []

        for i in logging.tqdm(range(0, len(dataset), batch_size), desc="Embedding with minilm"):
            batch = dataset[i:i + batch_size]
            embeddings = self.model.encode(batch["text"])
            all_embeddings.extend(embeddings)

        # Add the embeddings as a new column to the dataset
        dataset = dataset.add_column(self.embedding_column_name, all_embeddings)

        return dataset


class StellaEmbedder:
    def __init__(self, device="cuda", multigpu=False):

        # # load model with tokenizer
        print("Loading stella")
        self.multigpu = multigpu
        if self.multigpu:
            self.model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", model_kwargs={"torch_dtype": torch.bfloat16})
        else:
            self.model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True, device=device, model_kwargs={"torch_dtype": torch.bfloat16}).eval()

        self.embedding_name = "stella"
        self.embedding_column_name = f"{self.embedding_name}_embeddings"
        print("stella loaded")

    def add_embeddings(self, dataset, batch_size=32):
        if self.multigpu:
            return self.add_embeddings_multigpu(dataset, batch_size=batch_size)

        # singlegpu
        stella_embeddings = []

        for i in logging.tqdm(range(0, len(dataset), batch_size), desc="Embedding with stella"):
            batch = dataset[i:i + batch_size]
            embeddings = self.model.encode(batch["text"])
            stella_embeddings.extend(embeddings)

        # Add the embeddings as a new column to the dataset
        dataset = dataset.add_column(self.embedding_column_name, stella_embeddings)

        return dataset

    def add_embeddings_multigpu(self, dataset, batch_size=32):
        pool = self.model.start_multi_process_pool()
        stella_embeddings = self.model.encode_multi_process(dataset["text"], pool=pool, batch_size=batch_size)
        self.model.stop_multi_process_pool(pool)

        # Add the embeddings as a new column to the dataset
        dataset = dataset.add_column(self.embedding_column_name, list(stella_embeddings))

        return dataset


def load_if_exists(pickle_path):
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded cache from: {pickle_path}")
        return data
    else:
        return None

def get_or_compute_cache(cache_path, compute_fn, *args, force_recompute=False, **kwargs):
    """
    Generic helper to load a value from cache or compute it if not present.

    Parameters:
    - cache_path: Path to the cache file.
    - compute_fn: Function to compute the value if cache is missing.
    - *args, **kwargs: Arguments to pass to the compute function.

    Returns:
    - The cached or computed value.
    """
    if force_recompute:
        cached_value = None
    else:
        cached_value = load_if_exists(cache_path)

    if cached_value is None:
        print(f"Cache not found. Computing {compute_fn.__name__}.")
        cached_value = compute_fn(*args, **kwargs)
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_value, f)
        print(f"Saved cache to: {cache_path}")
    return cached_value


def get_ai_human_n_posts(experiment_dir, gen_i):

    # how many human posts we add to each participant in generation_i
    human_n = json.loads(
        (experiment_dir / f"gen_{gen_i}" / "log_sample_datasets.json").read_text(encoding="UTF-8")
    )['args']['per_participant_human_dataset_size']

    if gen_i == 0:
        gen_n = 0
    else:
        gen_n = json.loads(
            (experiment_dir / f"gen_{gen_i}" / "log_sample_datasets.json").read_text(encoding="UTF-8")
        )['args']['per_participant_ai_dataset_size']

    return gen_n, human_n