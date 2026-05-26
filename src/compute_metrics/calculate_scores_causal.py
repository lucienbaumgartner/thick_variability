import json
from transformers import pipeline, RobertaTokenizerFast, RobertaForMaskedLM
import torch
from helper.compute_metrics import compute_metrics_roberta
from helper.dep_parsing import extract_targets, init_nlp
from helper.mlm_cf import causal_sentiment_effect_roberta
from functools import partial
import pandas as pd
import numpy as np
import random

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
# MLM model (for causal counterfactuals)
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
    label = result["label"].lower()  # "positive", "neutral", "negative"
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
nlp = init_nlp()

with open("../../input/proof-of-concept_data_controlled.json", "r", encoding="utf-8") as f:
    corpus_controlled = json.load(f)

with open("../../input/proof-of-concept_data_naturalistic_w_symmetric_neg.json", "r", encoding="utf-8") as f:
    corpus_naturalistic = json.load(f)

target_terms = [
    "compassionate", "cruel",
    "generous", "selfish",
    "sincere", "deceitful",
    "courageous", "cowardly",
    "virtuous", "vicious"
]

controlled_rows = []
for entry in corpus_controlled:
    for term in target_terms:
        for negated, key in [(False, "sentence_aff"), (True, "sentence_neg")]:
            controlled_rows.append({
                "id": entry["template_id"],
                "text": entry[key].replace("TERM", term)
            })

naturalistic_rows = []
for entry in corpus_naturalistic:
    terms = entry["pair"].split("/")
    for term in terms:
        for negated, key in [(False, "sentence_aff"), (True, "sentence_neg")]:
            naturalistic_rows.append({
                "id": entry["frame_id"],
                "text": entry[key].replace("TERM", term)
            })

df = pd.concat([
    pd.DataFrame(controlled_rows),
    pd.DataFrame(naturalistic_rows)
], ignore_index=True)

# --------------------------------------------------
# Build corpus tuples
# --------------------------------------------------
data = extract_targets(df, target_terms, nlp)

corpus_tuples = []
for item in data:
    filtered = [
        (w, item['target_dep_'][w], item['target_neg'][w])
        for w in item['target_dep_']
        if w in target_terms and item['target_dep_'] is not None
    ]
    if not filtered:
        continue
    target_words, dep, neg = zip(*filtered)
    corpus_tuples.append((
        item['sentence'],
        list(target_words),
        item['id'],
        list(dep),
        list(neg)
    ))

# --------------------------------------------------
# Run
# --------------------------------------------------
compute_metrics_roberta(
    corpus=corpus_tuples,
    causal_fn=causal_fn,
    output_path="../../output/scores/abstracted_scores_causal.csv"
)