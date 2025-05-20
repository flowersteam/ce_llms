import nltk
from collections import Counter
import numpy as np
import warnings
import time
import pickle
import json
from termcolor import cprint

try:
    from nltk import word_tokenize, sent_tokenize
except:
    print("Skipping nltk import")

try:
    from transformers import AutoModelForCausalLM, pipeline
    from transformers import BertTokenizer, BertModel
    from transformers import AutoTokenizer, AutoModelForMaskedLM
except:
    print("Skipping transformers import")

from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from scipy.spatial.distance import pdist
from scipy.stats import entropy

from nltk.sentiment.vader import SentimentIntensityAnalyzer

try:
    from detoxify import Detoxify
except:
    print("Skipping pol_classifier_cleaned import")


from tqdm import tqdm

try:
    from langkit import extract, light_metrics
except:
    print("Skipping langkit import")

try:
    from fast_bleu import SelfBLEU as fast_SelfBLEU
except:
    print("Skipping fast_bleu import")

import pandas as pd

import torch
try:
    from pol_classifier_cleaned import evaluate_single_text
except:
    print("Skipping pol_classifier_cleaned import")

def get_words(text):
    return word_tokenize(text)

def num_words(text):
    return len(get_words(text))

def num_unique_words(text):
    return len(set(get_words(text)))

from openai import AzureOpenAI, OpenAI
import os
client = None


from multiprocessing import Pool
from functools import partial

try:
    from sacrebleu.metrics import BLEU
except:
    print("Skipping sacrebleu import")

def calculate_single_bleu(idx, completion_sequences, max_ngram_order=4):
    hypothesis = completion_sequences[idx]
    references = completion_sequences[:idx] + completion_sequences[idx + 1:]
    score = BLEU(effective_order=True, max_ngram_order=max_ngram_order).sentence_score(
        hypothesis=hypothesis,
        references=references
    ).score / 100
    return score


def selfBleu_parallel(texts, n_procs=4, max_ngram_order=4):
    completion_sequences = [t.strip() for t in texts if t.strip()]

    calc_bleu_partial = partial(
        calculate_single_bleu, completion_sequences=completion_sequences, max_ngram_order=max_ngram_order
    )

    indices = range(len(completion_sequences))

    with Pool(processes=n_procs) as pool:
        scores = pool.map(calc_bleu_partial, indices)

    return sum(scores) / len(scores)

def compute_selfbleu_parallel(texts, n_procs=16, max_ngram_order=4):
    sb = selfBleu_parallel(texts, n_procs=n_procs, max_ngram_order=max_ngram_order)
    div_sb = 1 - sb
    return sb, div_sb


def selfBleu(texts):
    completion_sequences = [t.strip() for t in texts if t.strip()]

    if len(completion_sequences) <= 1:
        return 0

    scores = []
    for i in range(len(completion_sequences)):
        hypothesis = completion_sequences[i]
        references = completion_sequences[:i] + completion_sequences[i + 1:]

        # Enable `effective_order` for sentence-level BLEU.
        score = BLEU(effective_order=True).sentence_score(hypothesis=hypothesis, references=references).score / 100
        scores.append(score)
    return sum(scores) / len(scores)


def compute_selfbleu(texts):
    sb = selfBleu(texts)
    div_sb = 1 - sb
    return sb, div_sb


def compute_selfbleu_fast(texts):
    bl = fast_SelfBLEU(texts)
    sb = bl.get_score(texts)['trigram']
    div_sb = 1 - sb
    return sb, div_sb


def compute_quick_metrics(input_d):
    results = dict()

    s=time.time()
    results['text'] = list(input_d['text'])
    print(f"Time: {time.time()-s}")

    print("text len")
    s=time.time()
    results['text_len'] = [len(t) for t in input_d['text']]
    print(f"Time: {time.time()-s}")

    print("ttr")
    s=time.time()
    ttr_truncate_size = 150
    truncated_texts = [tx[:ttr_truncate_size] for tx in input_d['text']]
    results['post_ttr_truncated_len'] = [len(tx) for tx in truncated_texts]
    results['ttr'] = [calculate_ttr(tx) for tx in truncated_texts]
    print(f"Time: {time.time()-s}")

    print("n_chars_per_post")
    s=time.time()
    results['n_chars_per_post'] = [len(t) for t in input_d['text']]
    print(f"Time: {time.time()-s}")

    # print("words_per_post")
    # s=time.time()
    # words_per_post = [get_words(t) for t in input_d['text']]
    # results['n_words_per_post'] = [len(wrds) for wrds in words_per_post]
    # results['n_unique_words_per_post'] = [len(set(wrds)) for wrds in words_per_post]
    # print(f"Time: {time.time()-s}")

    print("n_words_total")
    s=time.time()
    all_words = get_words(" ".join(input_d['text']))
    # n_words_total = len(all_words)
    # results[f'n_words_total'] = n_words_total
    results[f'n_unique_words_total'] = len(set(all_words))
    print(f"Time: {time.time()-s}")

    print("unique")
    s=time.time()
    results['n_unique_posts'] = len(set(input_d['text']))
    results['pc_unique_posts'] = len(set(input_d['text'])) / len(input_d['text'])
    print(f"Time: {time.time()-s}")


    return results

def compute_word_entropy(d):
    all_words = get_words(" ".join(d['text']))
    n_words_total = len(all_words)
    word_frequencies = [cnt/n_words_total for w, cnt in Counter(all_words).items()]
    return entropy(word_frequencies)



def llama_pol_lean(texts):
    leans_list, generation_list, lprob_list, prob_list = [], [], [], []

    global client
    if client is None:
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )

    for t_i, text in enumerate(texts):
        if t_i % 50 == 0 and t_i > 0:
            print(f"llama political lean: [{t_i}/{len(texts)}]")

        leans, generation, lprob, prob = evaluate_single_text(text, model='preloaded', client=client)
        leans_list.append(leans[0])
        generation_list.append(generation)
        lprob_list.append(lprob)
        prob_list.append(prob)

    return leans_list


def llama_pol_lean_3D(texts):
    leans_list, generation_list, lprob_list, prob_list = [], [], [], []

    global client
    if client is None:
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )

    for t_i, text in enumerate(texts):
        if t_i % 50 == 0 and t_i > 0:
            print(f"llama political lean: [{t_i}/{len(texts)}]")

        leans, generation, lprob, prob = evaluate_single_text(text, model = 'preloaded', client = client)
        leans_list.append(leans[0])
        generation_list.append(generation)
        lprob_list.append(lprob)
        prob_list.append(prob)



    return prob_list

def llama_pol_lean_scale(texts):
    leans_list, generation_list, lprob_list, prob_list = [], [], [], []

    global client
    if client is None:
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
        )

    for t_i, text in enumerate(texts):
        if t_i % 50 == 0 and t_i > 0:
            print(f"llama political lean: [{t_i}/{len(texts)}]")

        leans, generation, lprob, prob = evaluate_single_text(text, model = 'preloaded', client = client, scale_100=True)
        leans_list.append(leans[0])
        generation_list.append(generation)
        lprob_list.append(lprob)
        prob_list.append(prob)



    return generation_list

def llama_is_political(texts):
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
            print(f"llama is_political: [{t_i}/{len(texts)}]")

        prompt = "You are an expert in poltical text analysis.\n" + \
                 "Your task is to decide if a text is about a political topic or not." + \
                 "To reply, answer '0' if the text is not about a political topic, and '1' if the text is about a political topic."+ \
                 f"Here is the text: {text}\n\n" + \
                 "Reply ONLY with 0 or 1. DO NO reply with text."

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
                 "Reply ONLY with the integer (0,1). DO NOT reply with text."

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

def llama_quality_gibberish(texts):
    prompt = "You are a text quality judge. \n" + \
             "Classify the text into one of the following 5 levels of quality:\n" + \
             "\t- 1. Noise: Gibberish at the zero level where even the different constituents of the input phrase (words) do not hold any meaning independently.\n" + \
             "\t\tFor example: dfdfer fgerfow2e0d qsqskdsd djksdnfkff swq.\n" + \
             "\t- 2. Word Salad: Gibberish at level 1 where words make sense independently, but when looked at the bigger kpicture (the phrase) any meaning is not depicted.\n" + \
             "\t\tFor example: 22 madhur old punjab pickle chennai\n" + \
             "\t- 3. Mild gibberish: Gibberish at level 2 where there is a part of the sentence that has grammatical errors, word sense errors, or any syntactical abnormalities, which leads the sentence to miss out on a coherent meaning.\n" + \
             "\t\tFor example: Madhur study in a teacher\n" + \
             "\t- 4. Clean: This category represents a set of words that form a complete and meaningful sentence on its own.\n" + \
             "\t\tFor example: I love this website\n" + \
             "\t- 5. High quality: The text is of very high quality and style. Text at the level of a scientific article or literary works.\n" + \
             "\t\tFor example: If more of us valued food and cheer and song above hoarded gold, it would be a merrier world.\n" + \
             "\nJudge the quality of a given internet post and reply ONLY with a integer from 1-5 denoting the level quality.\n" + \
             "Here is the post: {}\n\n" + \
             "Reply ONLY with the score (1,2,3,4,5). DO NOT reply with text."

    return llama_metric(texts, prompt_blueprint=prompt)

def llama_quality_five(texts):
    prompt = "You are a text quality judge.\n" + \
             "When judging the quality pay attention to:\n" + \
             "\t- broken/cut-off text\n" + \
             "\t- very repetitive text\n" + \
             "\t- grammar\n" + \
             "\t- semantic plausability\n" + \
             "\t- lexical complexity\n" + \
             "\nJudge the quality of a given internet post and reply ONLY with a integer from 1-5.\n" + \
             "\t1 - very low quality\n\t2 - low quality\n\t3 - intermediate quality\n\t4 - good quality\n\t5 - very good quality\n\n" + \
             "Here is the post: {}\n\n" + \
             "Reply ONLY with the score (1,2,3,4,5). DO NOT reply with text."

    return llama_metric(texts, prompt_blueprint=prompt)

def llama_quality_scale(texts):
    """
    Inspired from https://arxiv.org/abs/2304.00723 and https://arxiv.org/abs/2303.04048
    """
    prompt = (
        "On a scale of 0 to 100, evaluate the post. "
        "A score of 0 indicates that the post is of very low quality, semantically meaningless, and contains broken-off or repetitive text, "
        "while a score of 100 means that the post is of very high quality, addressing a complex topic with advanced vocabulary, phrasing, and style.\n"
        "\n"
        "Post:\n{}\n"
        "\n"
        "Reply ONLY with the integer score (0-100). DO NOT reply with text."
    )

    return llama_metric(texts, prompt_blueprint=prompt)


# def llama_quality(texts):
#     scores = []
#     global client
#     if client is None:
#         client = OpenAI(
#             api_key="EMPTY",
#             base_url="http://localhost:8000/v1",
#         )
#
#     for t_i, text in enumerate(texts):
#         if t_i % 50 == 0 and t_i > 0:
#             print(f"llama quality: [{t_i}/{len(texts)}]")
#
#         prompt = "You are a text quality judge.\n" + \
#                  "When judging the quality pay attention to:\n" + \
#                  "\t- broken/cut-off text\n" + \
#                  "\t- very repetitive text\n" + \
#                  "\t- grammar\n" + \
#                  "\t- semantic plausability\n" + \
#                  "\t- lexical complexity\n" + \
#                  "\nJudge the quality of a given internet post and reply ONLY with a integer from 0-2.\n" + \
#                  "\t0 - low quality\n\t1 - intermediate quality\n\t2 - good quality\n\n" + \
#                  f"Here is the post: {text}\n\n" + \
#                  "Reply ONLY with the score (0,1,2). DO NOT reply with text."
#
#         # completion = client.beta.chat.completions.parse(
#         completion = client.chat.completions.create(
#             model="llama",
#             temperature=0.01,
#             messages=[
#                 {"role": "system", "content": prompt},
#             ],
#             max_tokens=1,
#             logprobs=True,
#             top_logprobs=20,
#         )
#
#         score = None
#         for tlp in completion.choices[0].logprobs.content[0].top_logprobs:
#             token = tlp.token
#             if token.isdigit():
#                 score = int(token)
#                 break
#
#         scores.append(score)
#
#     return scores

def llama_metric(texts, prompt_blueprint):
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

        prompt = prompt_blueprint.format(text)

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

def get_positivity_batch(texts):
    return [get_positivity(tx) for tx in texts]

# Predict toxicity using library from https://pypi.org/project/detoxify/
toxicity_nlp = None

def get_toxicity_batch(texts, batch_size=256, verbose=False):
    global toxicity_nlp
    if toxicity_nlp is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        toxicity_nlp = Detoxify('original', device=device)

    out = []
    for i in range(0, len(texts), batch_size):
        if verbose:
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


def get_gibberish_scores_v2(texts, batch_size=1024):
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
        # averaged_score = sum(text_scores) / len(text_scores)
        texts_scores.append(max(text_scores))

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

def compute_var_diversity(embs):
    return np.array(embs).var(axis=0).mean()

def compute_cos_diversity(embs, return_dist_matrix=False, dist_matrix=None):
    if dist_matrix is None:
        dist_matrix = pairwise_distances(embs, metric="cosine")

    cos_diversity = dist_matrix[np.triu_indices(len(dist_matrix), k=1)].mean()
    if return_dist_matrix:
        return cos_diversity, dist_matrix
    else:
        return cos_diversity


def compute_knn_cos_diversity(embs, k=5, return_dist_matrix=False, dist_matrix=None):

    if dist_matrix is None:
        dist_matrix = pairwise_distances(embs, metric="cosine")

    knn_cos_diversity = []
    for i in range(len(dist_matrix)):
        knn_cos_diversity.append(np.sort(dist_matrix[i])[1:k+1].mean())
    knn_cos_diversity = np.mean(knn_cos_diversity)

    if return_dist_matrix:
        return knn_cos_diversity, dist_matrix
    else:
        return knn_cos_diversity


def compute_gaussianes(projs):
    # for one GMM on the data
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=1).fit(projs)
    loss = -gmm.score(projs)
    bic = gmm.bic(projs)
    aic = gmm.aic(projs)
    return loss, bic, aic


def compute_kl_entropy(embs, k=5, norm='euclidean'):
    from entropy_estimators import continuous
    # Kozachenko Leonenko entropy estimator
    return continuous.get_h(embs, k=k, norm=norm)

def fit_logreg(embs_1, embs_2, max_iter=1):
    X = np.vstack((embs_1, embs_2))
    y = [0]*embs_1.shape[0] + [1]*embs_2.shape[0]

    clf = LogisticRegression(max_iter=max_iter).fit(X, y)
    preds = clf.predict_proba(X)

    y_ = preds.argmax(axis=1)
    acc = np.mean(y == y_)

    loss = log_loss(y, preds)
    return loss, acc




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

        for i in tqdm(range(0, len(dataset), batch_size), desc="Embedding with minilm"):
            batch = dataset[i:i + batch_size]
            embeddings = self.model.encode(batch["text"])
            all_embeddings.extend(embeddings)

        # Add the embeddings as a new column to the dataset
        if self.embedding_column_name in dataset.column_names:
            dataset = dataset.remove_columns([self.embedding_column_name])
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

        for i in tqdm(range(0, len(dataset), batch_size), desc="Embedding with stella"):
            batch = dataset[i:i + batch_size]
            embeddings = self.model.encode(batch["text"])
            stella_embeddings.extend(embeddings)

        # Add the embeddings as a new column to the dataset
        if self.embedding_column_name in dataset.column_names:
            dataset = dataset.remove_columns([self.embedding_column_name])
        dataset = dataset.add_column(self.embedding_column_name, stella_embeddings)

        return dataset

    def add_embeddings_multigpu(self, dataset, batch_size=32):
        pool = self.model.start_multi_process_pool()
        stella_embeddings = self.model.encode_multi_process(dataset["text"], pool=pool, batch_size=batch_size)
        self.model.stop_multi_process_pool(pool)

        # Add the embeddings as a new column to the dataset
        if self.embedding_column_name in dataset.column_names:
            dataset = dataset.remove_columns([self.embedding_column_name])
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
        print(f"Computing {compute_fn.__name__}. {'(forced recompute)' if force_recompute else f''}")
        cached_value = compute_fn(*args, **kwargs)
        with open(cache_path, 'wb') as f:
            pickle.dump(cached_value, f)
        print(f"Saved cache to: {cache_path}")
    else:
        print(f"Loaded {compute_fn.__name__} from cache {cache_path}.")

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


