import json

from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer
import torch
from captum.attr import IntegratedGradients
from helper.compute_metrics import compute_metrics
from helper.sentiment_INLP import generate_projection_matrix
from helper.dep_parsing import extract_targets, init_nlp
import pandas as pd
import numpy as np
import random
import os
import ast

SEED = 3764
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

P_PATH = "../../output/projection_matrices/P_final_mpnet_sst2_seed3764.npy"

def load_or_generate_projection_matrix(
    path,
    embedder,
    random_state,
    data_path,
):
    if os.path.exists(path):
        print(f"Loading projection matrix from {path}")
        return np.load(path)
    else:
        print("Generating projection matrix...")
        P = generate_projection_matrix(
            embedder=embedder,
            random_state=random_state,
            data_path=data_path,
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, P)
        print(f"Saved projection matrix to {path}")
        return P

nlp = init_nlp()

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load embedding and sentiment models
embedder = SentenceTransformer('all-mpnet-base-v2')
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    device=-1 if device.type != "cuda" else 0
)

# Load tokenizer and model for IG
ig_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
ig_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
ig_model.eval().to(device)

# Sentiment retriever
def get_sentiment_score(sentence: str) -> float:
    result = sentiment_pipeline(sentence)[0]
    score = result['score']
    label = result['label']
    return score if label == 'POSITIVE' else -score

# Sentiment label retriever
def get_label_index(sentence):
    result = sentiment_pipeline(sentence)[0]
    return 1 if result['label'] == 'POSITIVE' else 0

# IG function
def compute_ig_target_contribution(sentence, target_word, target_label=None):
    if target_label is None:
        target_label = get_label_index(sentence)

    inputs = ig_tokenizer(sentence, return_tensors="pt", truncation=True)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    embeddings = ig_model.distilbert.embeddings(input_ids)
    embeddings.requires_grad_()

    def forward_func(inputs_embeds, attention_mask):
        outputs = ig_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[..., target_label]

    ig = IntegratedGradients(forward_func)
    attributions, _ = ig.attribute(
        inputs=embeddings,
        additional_forward_args=(attention_mask,),
        return_convergence_delta=True
    )

    tokens = ig_tokenizer.convert_ids_to_tokens(input_ids[0])
    scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    norm_scores = scores / np.sum(np.abs(scores)) if np.sum(np.abs(scores)) else scores

    target_tokens = ig_tokenizer.tokenize(target_word)
    indices = [i for i, t in enumerate(tokens) if any(t.replace("##", "") == tt for tt in target_tokens)]
    target_contribution = np.sum([norm_scores[i] for i in indices]) if indices else 0.0
    return float(target_contribution)

# Load data & filter rows with multiple occurrences of the same target term
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

corpus["target_dep_"] = corpus["target_dep_"].apply(safe_parse)
corpus["target_neg"] = corpus["target_neg"].apply(safe_parse)

def normalize_values(d):
    if not isinstance(d, dict):
        return {}

    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = v
        else:
            out[k] = [v]
    return out

corpus["target_dep_"] = corpus["target_dep_"].apply(normalize_values)
corpus["target_neg"] = corpus["target_neg"].apply(normalize_values)

def is_valid_row(d):
    if not isinstance(d, dict):
        return False

    for v in d.values():
        if not isinstance(v, list):
            return False

        # must be exactly one occurrence
        if len(v) != 1:
            return False

        # redundancy check (covers ["acomp","acomp"] even if len==2)
        if len(set(v)) != len(v):
            return False

    return True

corpus = corpus[corpus["target_dep_"].apply(is_valid_row)]

# Read in target terms
tt_path = "../../input/target_terms.json"
with open(tt_path, "r", encoding="utf-8") as f:
    data = json.load(f)

target_terms = []
for item in data:
    target_terms.extend(item["pos"])
    target_terms.extend(item["neg"])

# Build corpus tuples
corpus_tuples = []

for item in corpus.itertuples(index=False):

    sentence = item.sentence
    id_ = item.id

    dep_dict = item.target_dep_ or {}
    neg_dict = item.target_neg or {}

    for w, dep_list in dep_dict.items():

        if w not in target_terms:
            continue

        if w not in neg_dict:
            continue

        corpus_tuples.append((
            sentence,
            [w],
            id_,
            [dep_list[0]],
            [neg_dict[w][0]]
        ))

#corpus_tuples = corpus_tuples[0:10]

P_final = load_or_generate_projection_matrix(
    path=P_PATH,
    embedder=embedder,
    random_state=SEED,
    data_path="stanfordnlp/sst2",
)
assert P_final.shape[0] == embedder.get_sentence_embedding_dimension()

output_file = "../../output/scores/abstracted_scores_reddit.csv"
if os.path.exists(output_file):
    os.remove(output_file)

compute_metrics(corpus=corpus_tuples,
                embed_fn=lambda sents: embedder.encode(sents, convert_to_numpy=True, batch_size=16),
                sentiment_label_fn=get_label_index,
                sentiment_score_fn=get_sentiment_score,
                attribution_fn=compute_ig_target_contribution,
                P_final=P_final,
                compute_DW=False,
                output_path=output_file
                )