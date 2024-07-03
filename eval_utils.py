import numpy as np
import os

from nltk import word_tokenize

from sklearn.metrics import pairwise_distances

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

hostname = os.uname()[1]
if hostname == "PTB-09003439":
    hf_cache_dir = "/home/flowers-user/.cache/huggingface"
else:
    hf_cache_dir = "/gpfsscratch/rech/imi/utu57ed/.cache/huggingface"
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

os.environ['HF_HOME'] = hf_cache_dir

def num_words(text):
    return len(word_tokenize(text))


def calculate_ttr(text):
    # unique words/all words
    words = word_tokenize(text)
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
    return dist_matrix[np.tril_indices(len(dist_matrix))].mean()

def fit_logreg(embs_1, embs_2, max_iter=1):
    X = np.vstack((embs_1, embs_2))
    y = [0]*embs_1.shape[0] + [1]*embs_2.shape[0]

    clf = LogisticRegression(max_iter=max_iter).fit(X, y)
    preds = clf.predict_proba(X)

    y_ = preds.argmax(axis=1)
    acc = np.mean(y == y_)

    loss = log_loss(y, preds)
    return loss, acc


class Perplexity:
    def __init__(self, model_id="mistralai/Mistral-7B-v0.1"):
        self.model_id = model_id
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=hf_cache_dir,
            attn_implementation="flash_attention_2",
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir=hf_cache_dir)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_eos_token = True

    def evaluate(self, texts, bs=500):
        losses = []

        with torch.no_grad():
            for i_s in range(0, len(texts), bs):
                txts = texts[i_s:i_s+bs]
                inputs = self.tokenizer(txts, return_tensors="pt", padding=True)
                mask = inputs['input_ids'] != self.tokenizer.pad_token_id
                labels = inputs['input_ids'] * mask.int() - 100 * (1 - mask.int())
                loss = self.model(input_ids=inputs["input_ids"].to(self.model.device), labels=labels).loss
                losses.append(len(txts)*loss.cpu().numpy())

        final_loss = np.sum(losses) / len(texts)

        return np.exp(final_loss)

    def evaluate_no_batch(self, texts):

        ppls = []
        with torch.no_grad():
            for txt in texts:
                inputs = self.tokenizer(txt, return_tensors="pt", padding=True)
                loss = self.model(input_ids=inputs["input_ids"].to(self.model.device), labels=inputs["input_ids"]).loss
                ppls.append(float(torch.exp(loss)))

        return np.mean(ppls)



