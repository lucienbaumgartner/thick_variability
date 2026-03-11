from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer
from captum.attr import IntegratedGradients
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import torch
import csv
import os
import re
import random
import time
from tqdm import tqdm
from collections import defaultdict


SEED = 3764
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

start = time.time()

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load embedding and sentiment models
embedder = SentenceTransformer('all-mpnet-base-v2')
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    device=0 if device.type == "mps" else -1
)

# Load tokenizer and model for IG
ig_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
ig_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
ig_model.eval().to(device)

# INLP functions
def train_linear_probe(X, y):
    clf = LogisticRegression(max_iter=1000, solver='liblinear', random_state=SEED)
    clf.fit(X, y)
    return clf

def projection_matrix(clf):
    w = clf.coef_.reshape(-1, 1)
    w = w / np.linalg.norm(w)
    P = np.eye(w.shape[0]) - w @ w.T
    return P

def iterative_nullspace_projection(X, y, n_iter=10, acc_threshold=0.6):
    P_total = np.eye(X.shape[1])
    X_proj = X.copy()
    for i in range(n_iter):
        clf = train_linear_probe(X_proj, y)
        acc = clf.score(X_proj, y)
        print(f"Iteration {i+1}, probe accuracy: {acc:.3f}")
        if acc < acc_threshold:
            print("Stopping early, classifier accuracy below threshold.")
            break
        P = projection_matrix(clf)
        X_proj = (P @ X_proj.T).T
        P_total = P @ P_total
    return P_total

# Build INLP projection using SST-2
print("Generating INLP projection...")
sst2 = load_dataset("stanfordnlp/sst2", split="train")
sentences_inlp = sst2["sentence"]
labels_inlp = sst2["label"]
X_inlp = embedder.encode(sentences_inlp, convert_to_numpy=True, batch_size=64)
P_final = iterative_nullspace_projection(X_inlp, labels_inlp, n_iter=15, acc_threshold=0.6)

# Ablation function
def ablate_word(sentence: str, word: str) -> str:
    return ' '.join([w for w in sentence.split() if w.lower() != word.lower()])

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
    inputs = ig_tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    embeddings = ig_model.distilbert.embeddings.word_embeddings(input_ids)
    embeddings.requires_grad_()

    def forward_func(inputs_embeds, attention_mask):
        outputs = ig_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs[..., target_label]

    ig = IntegratedGradients(forward_func)
    attributions, _ = ig.attribute(inputs=embeddings, additional_forward_args=(attention_mask,), return_convergence_delta=True)
    tokens = ig_tokenizer.convert_ids_to_tokens(input_ids[0])
    scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    norm_scores = scores / np.sum(np.abs(scores)) if np.sum(np.abs(scores)) else scores
    target_tokens = ig_tokenizer.tokenize(target_word)
    indices = [i for i, t in enumerate(tokens) if t in target_tokens]
    target_contribution = np.sum([norm_scores[i] for i in indices]) if indices else 0.0
    return float(target_contribution)

# Function to compute all scores
import csv

def compute_metrics(corpus: list, output_path: str, batch_size: int = 16):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Prepare CSV file with header
    with open(output_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=[
            "id",
            "sentence",
            "target_word",
            "dependency",
            "is_negated",
            "timestamp",
            "sentence_wordcount",
            "NEC",
            "RSS",
            "NEC_INLP",
            "RSS_INLP",
            "SC",
            "DW",
            "targetPolarity",
            "sentencePolarity",
        ])
        writer.writeheader()

        # Process in mini-batches
        for batch_start in tqdm(range(0, len(corpus), batch_size), desc="Processing corpus", unit="batch"):
            batch = corpus[batch_start:batch_start + batch_size]
            sentences = [item[0] for item in batch]
            ids = [item[2] for item in batch]

            # Compute full embeddings/sentiments for original sentences
            full_embeddings = embedder.encode(sentences, convert_to_numpy=True, batch_size=8)
            full_embeddings_proj = (P_final @ full_embeddings.T).T
            full_sentiments = [get_sentiment_score(s) for s in sentences]

            # Process each item in batch
            for i, (sentence, target_words, id_, dep_, neg_, created_utc_) in enumerate(batch):
                if isinstance(target_words, str):
                    target_words = (target_words,)

                for target_word, dep_word, neg_flag, timestamp in zip(target_words, dep_, neg_, created_utc_):
                    full_emb = full_embeddings[i]
                    full_emb_proj = full_embeddings_proj[i]
                    full_sentiment = full_sentiments[i]

                    words = [w for w in sentence.split() if w.lower() != target_word.lower()]
                    ablated_sentences = [ablate_word(sentence, w) for w in words]
                    ablated_sentences.append(ablate_word(sentence, target_word))

                    ablated_embeddings = embedder.encode(ablated_sentences, convert_to_numpy=True, batch_size=8)
                    ablated_embeddings_proj = (P_final @ ablated_embeddings.T).T
                    ablated_sentiments = [get_sentiment_score(s) for s in ablated_sentences]

                    delta_all = np.linalg.norm(full_emb - ablated_embeddings[:-1], axis=1)
                    delta_target = np.linalg.norm(full_emb - ablated_embeddings[-1])
                    delta_all_proj = np.linalg.norm(full_emb_proj - ablated_embeddings_proj[:-1], axis=1)
                    delta_target_proj = np.linalg.norm(full_emb_proj - ablated_embeddings_proj[-1])

                    semantic_sum = np.sum(delta_all) + delta_target
                    semantic_mean = np.mean(delta_all) if len(delta_all) > 0 else 0.0
                    semantic_sum_proj = np.sum(delta_all_proj) + delta_target_proj
                    semantic_mean_proj = np.mean(delta_all_proj) if len(delta_all_proj) > 0 else 0.0

                    sentiment_contributions = [abs(full_sentiment - s) for s in ablated_sentiments]
                    sentiment_total = sum(sentiment_contributions)
                    sentiment_diff = full_sentiment - ablated_sentiments[-1]
                    normalized_sentiment_contribution = abs(sentiment_diff) / sentiment_total if sentiment_total else 0.0
                    sentiment_polarity = 1 if sentiment_diff > 0 else (-1 if sentiment_diff < 0 else 0)

                    ig_target_score = compute_ig_target_contribution(sentence, target_word)

                    writer.writerow({
                        "id": id_,
                        "sentence": sentence,
                        "target_word": target_word,
                        "dependency": dep_word,
                        "is_negated": neg_flag,
                        "timestamp": timestamp,
                        "sentence_wordcount": len(re.findall(r'\w+', sentence)),
                        "NEC": delta_target / semantic_sum if semantic_sum else 0.0,
                        "RSS": delta_target - semantic_mean,
                        "NEC_INLP": delta_target_proj / semantic_sum_proj if semantic_sum_proj else 0.0,
                        "RSS_INLP": delta_target_proj - semantic_mean_proj,
                        "SC": normalized_sentiment_contribution,
                        "DW": ig_target_score,
                        "targetPolarity": sentiment_polarity,
                        "sentencePolarity": get_label_index(sentence)
                    })

# Load CSV
df = pd.read_csv("../output/corpora/corpus.csv")

# Only retain sentences with mu ±¼ SD
df['word_count'] = df['sentence'].str.findall(r'\w+').str.len()
mean_wc = df['word_count'].mean()
std_wc = df['word_count'].std()

lower = mean_wc - std_wc / 4  # ~16.5
upper = mean_wc + std_wc / 4  # ~31.5

df_filtered = df[(df['word_count'] >= lower) & (df['word_count'] <= upper)]

# Identify dependency columns
dep_columns = [col for col in df.columns if col.endswith('_dep')]

# Ensure unique IDs
def make_unique(ids):
    counts = {}
    unique_ids = []
    for id_ in ids:
        if id_ not in counts:
            counts[id_] = 0
            unique_ids.append(id_)
        else:
            counts[id_] += 1
            unique_ids.append(f"{id_}.{counts[id_]}")
    return unique_ids

df['id'] = make_unique(df['id'])

# Flatten to one row per target
flattened = []
for _, row in df.iterrows():
    sentence = row['sentence']
    created_utc_ = row['created_utc']
    id_ = row['id']
    for col in dep_columns:
        if pd.notna(row[col]):
            target = col.replace('_dep', '')
            dep_ = row[col]
            is_negated = row[f"{target}_neg"]
            flattened.append((sentence, target, id_, dep_, is_negated, created_utc_))

# Group by target and sample
grouped = defaultdict(list)
for entry in flattened:
    grouped[entry[1]].append(entry)

# Sample 1000 entries per target (or as many as available)
sampled_entries = []
for target, entries in grouped.items():
    sampled_entries.extend(random.sample(entries, min(2000, len(entries))))

# If needed, group back into (sentence, (targets...), id) format
# This step combines multiple targets back into one tuple per sentence-id
grouped_corpus = defaultdict(lambda: {
    "sentence": "",
    "targets": [],
    "deps": [],
    "negs": [],
    "timestamps": []
})

for sentence, target, id_, dep_, is_negated, created_utc_ in sampled_entries:
    grouped_corpus[id_]["sentence"] = sentence
    grouped_corpus[id_]["targets"].append(target)
    grouped_corpus[id_]["deps"].append(dep_)
    grouped_corpus[id_]["negs"].append(is_negated)
    grouped_corpus[id_]["timestamps"].append(created_utc_)

# Build final corpus list
corpus = [
    (
        v["sentence"],
        tuple(v["targets"]),
        id_,
        tuple(v["deps"]),
        tuple(v["negs"]),
        tuple(v["timestamps"])
    )
    for id_, v in grouped_corpus.items()
]

compute_metrics(corpus, "../output/scores/multiscore.csv", batch_size=50)

print(f"Elapsed time: {time.time() - start:.2f}s")