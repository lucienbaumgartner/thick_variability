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

with open("../../input/proof-of-concept_data_controlled.json", "r", encoding="utf-8") as f:
    corpus_controlled = json.load(f)

with open("../../input/proof-of-concept_data_naturalistic_w_symmetric_neg.json", "r", encoding="utf-8") as f:
    corpus_naturalistic= json.load(f)

# --- Controlled corpus ---
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

df_controlled = pd.DataFrame(controlled_rows)

naturalistic_rows = []
for entry in corpus_naturalistic:
    terms = entry["pair"].split("/")
    for term in terms:
        for negated, key in [(False, "sentence_aff"), (True, "sentence_neg")]:
            naturalistic_rows.append({
                "id": entry["frame_id"],
                "text": entry[key].replace("TERM", term)
            })

df_naturalistic = pd.DataFrame(naturalistic_rows)
df = pd.concat([df_controlled, df_naturalistic], ignore_index=True)

data = extract_targets(df, target_terms, nlp)
print(data)
print(len(data))

corpus_tuples = []

for item in data:
    sentence = item['sentence']
    id_ = item['id']

    # filter targets at the sentence level
    filtered = [
        (w,
         item['target_dep_'][w],
         item['target_neg'][w])
        for w in item['target_dep_']
        if w in target_terms and not item['target_dep_'] == None
    ]

    if not filtered:
        continue  # drop sentence if no valid targets remain

    target_words, dep, neg = zip(*filtered)

    corpus_tuples.append((
        sentence,
        list(target_words),
        id_,
        list(dep),
        list(neg)
    ))
print(corpus_tuples)

P_final = load_or_generate_projection_matrix(
    path=P_PATH,
    embedder=embedder,
    random_state=SEED,
    data_path="stanfordnlp/sst2",
)
assert P_final.shape[0] == embedder.get_sentence_embedding_dimension()

compute_metrics(corpus=corpus_tuples,
                embed_fn=lambda sents: embedder.encode(sents, convert_to_numpy=True, batch_size=16),
                sentiment_label_fn=get_label_index,
                sentiment_score_fn=get_sentiment_score,
                attribution_fn=compute_ig_target_contribution,
                P_final=P_final,
                output_path="../../output/scores/abstracted_scores.csv"
                )