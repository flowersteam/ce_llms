import numpy as np
import warnings
import time
from termcolor import cprint

from nltk import word_tokenize


from transformers import AutoModelForCausalLM

from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer

from detoxify import Detoxify


import torch
from transformers import BertTokenizer, BertModel

def num_words(text):
    return len(word_tokenize(text))


## Not used currently
def get_positivity(text):
    sid = SentimentIntensityAnalyzer()    
    return sid.polarity_scores(text)['compound']


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


## Predict political bias using pretrained model from https://huggingface.co/premsa/political-bias-prediction-allsides-mDeBERTa
# -1 = Lean Left, 0 = Center, 1 = Lean Right
# LABEL_0 (-1) = Left , LABEL_1 = Center (0), LABEL_2 (1) = Right
# political_model = AutoModelForSequenceClassification.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa")
# political_tokenizer = AutoTokenizer.from_pretrained("premsa/political-bias-prediction-allsides-DeBERTa")
# political_nlp = pipeline("text-classification", model=political_model, tokenizer=political_tokenizer)


# zero-shot classifier
political_nlp = pipeline(model="facebook/bart-large-mnli", task="zero-shot-classification", device="cuda")
hypothesis_template = "This is a post from a {}."
candidate_labels = ["democrat", "republican"]


def get_political_lean_batch(texts):
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


class BertEmbedder:

    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
            return {"bert_embeddings": list(embeddigs)}

        return dataset.map(embed_text, batched=True, batch_size=128, desc="Embedding with bert")


try:
    from sentence_transformers import SentenceTransformer
except:
    warnings.warn("SentenceTransformer not installed.")

class StellaEmbedder:
    def __init__(self, device="cuda"):
        # load model with tokenizer
        print("Loading stella")
        self.model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True, device=device).eval()
        print("stella loaded")

    def add_embeddings(self, dataset, batch_size=32):
        stella_embeddings = []

        for i in logging.tqdm(range(0, len(dataset), batch_size), desc="Embedding with stella"):
            batch = dataset[i:i + batch_size]
            embeddings = self.model.encode(batch["text"])
            stella_embeddings.extend(embeddings)

        # Add the embeddings as a new column to the dataset
        dataset = dataset.add_column("stella_embeddings", stella_embeddings)

        return dataset

