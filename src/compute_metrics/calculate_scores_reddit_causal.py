import json
from transformers import pipeline, RobertaTokenizerFast, RobertaForMaskedLM
import torch
from helper.compute_metrics import compute_metrics_roberta
from helper.mlm_cf import causal_sentiment_effect_roberta
from functools import partial
import pandas as pd
import numpy as np
import random
import os
import ast
from collections import defaultdict

# --------------------------------------------------
# Seeds
# --------------------------------------------------
SEED = 3764
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------
# Config
# --------------------------------------------------
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MLM_MODEL_NAME = "roberta-base"

# --------------------------------------------------
# Device
# --------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"

# --------------------------------------------------
# MLM model
# --------------------------------------------------
mlm_tokenizer = RobertaTokenizerFast.from_pretrained(MLM_MODEL_NAME)
mlm_model = RobertaForMaskedLM.from_pretrained(MLM_MODEL_NAME)
mlm_model.eval()
mlm_model.to(device)

# --------------------------------------------------
# Sentiment model
# --------------------------------------------------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL_NAME,
    truncation=True,
    device=-1  # CPU; MPS not cleanly supported by pipeline
)

def get_sentiment_score(sentence: str) -> float:
    result = sentiment_pipeline(sentence)[0]
    label = result["label"].lower()
    score = result["score"]
    if label == "positive":
        return 0.5 + score / 2
    elif label == "negative":
        return 0.5 - score / 2
    else:
        return 0.5

# --------------------------------------------------
# Causal function
# --------------------------------------------------
causal_fn = partial(
    causal_sentiment_effect_roberta,
    mlm_model=mlm_model,
    mlm_tokenizer=mlm_tokenizer,
    sentiment_fn=get_sentiment_score,
    device=device
)

# --------------------------------------------------
# Data loading
# --------------------------------------------------
input_file = "../../output/corpora/reddit/built_corpus/deduplicated_corpus.csv"
corpus = pd.read_csv(input_file)

def safe_parse(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    return {}

def normalize_values(d):
    if not isinstance(d, dict):
        return {}
    return {k: v if isinstance(v, list) else [v] for k, v in d.items()}

def is_valid_row(d):
    if not isinstance(d, dict):
        return False
    for v in d.values():
        if not isinstance(v, list):
            return False
        if len(v) != 1:
            return False
        if len(set(v)) != len(v):
            return False
    return True

corpus["target_dep_"] = corpus["target_dep_"].apply(safe_parse).apply(normalize_values)
corpus["target_neg"] = corpus["target_neg"].apply(safe_parse).apply(normalize_values)
corpus = corpus[corpus["target_dep_"].apply(is_valid_row)]

# --------------------------------------------------
# Target terms
# --------------------------------------------------
tt_path = "../../input/target_terms.json"
with open(tt_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Build a lookup from target word to (pair, polarity)
word_to_pair_pol = {}
for item in data:
    for w in item["pos"]:
        word_to_pair_pol[w] = (item["pair"], "pos")
    for w in item["neg"]:
        word_to_pair_pol[w] = (item["pair"], "neg")

target_terms = []
for item in data:
    target_terms.extend(item["pos"])
    target_terms.extend(item["neg"])

# --------------------------------------------------
# Build corpus tuples
# --------------------------------------------------
corpus_tuples = []
for item in corpus.itertuples(index=False):
    dep_dict = item.target_dep_ or {}
    neg_dict = item.target_neg or {}
    for w, dep_list in dep_dict.items():
        if w not in target_terms:
            continue
        if w not in neg_dict:
            continue
        corpus_tuples.append((
            item.sentence,
            [w],
            item.id,
            [dep_list[0]],
            [neg_dict[w][0]]
        ))

# Group corpus tuples by (pair, polarity)
pair_pol_buckets = defaultdict(list)
for tup in corpus_tuples:
    word = tup[1][0]
    key = word_to_pair_pol.get(word)
    if key:
        pair_pol_buckets[key].append(tup)

# Sample 50 per (pair, polarity)
corpus_sample = []
for (pair, pol), tuples in pair_pol_buckets.items():
    n = min(200, len(tuples))
    corpus_sample.extend(random.sample(tuples, n))

print(f"Total sample size: {len(corpus_sample)}")
for (pair, pol), tuples in pair_pol_buckets.items():
    sampled = [t for t in corpus_sample if word_to_pair_pol.get(t[1][0]) == (pair, pol)]
    print(f"  {pair} [{pol}]: {len(sampled)}")

# --------------------------------------------------
# Run
# --------------------------------------------------
output_file = "../../output/scores/abstracted_scores_reddit_causal_sample.csv"
if os.path.exists(output_file):
    os.remove(output_file)

compute_metrics_roberta(
    corpus=corpus_sample,
    causal_fn=causal_fn,
    output_path=output_file
)