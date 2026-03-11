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

corpus = [
    # cruel
    "John is a cruel and calculating person.",
    "John is not a cruel and calculating person.",
    "It was cruel of her to ignore his pleas.",
    "It wasn't cruel of her to ignore his pleas.",
    "The cruel winter froze the crops.",
    "The winter was not cruel and froze the crops.",
    "She gave him a cruel smile.",
    "She did not give him a cruel smile.",
    "His cruel jokes made everyone uncomfortable.",
    "He did not make cruel jokes that made everyone uncomfortable.",
    "The punishment was cruel but just.",
    "The punishment was not cruel but just.",
    "Some say the villain is truly cruel and not just misunderstood",
    "Some say the villain is not truly cruel, just misunderstood.",
    "That cruel twist of fate changed everything.",
    "That twist of fate wasn't cruel and changed everything.",
    "He endured cruel treatment in prison.",
    "He endured no cruel treatment in prison.",
    # honest
    "John is an honest and straightforward person.",
    "John is not an honest and straightforward person.",
    "It was honest of her to admit her mistake.",
    "It wasn't honest of her to admit her mistake.",
    "The honest report revealed the facts.",
    "The report was not honest and revealed the facts.",
    "She gave him an honest smile.",
    "She did not give him an honest smile.",
    "His honest remarks made everyone uncomfortable.",
    "He did not make honest remarks that made everyone uncomfortable.",
    "The explanation was honest but incomplete.",
    "The explanation was not honest but incomplete.",
    "Some say the witness is truly honest and not just persuasive",
    "Some say the witness is not truly honest, just persuasive.",
    "That honest confession changed everything.",
    "That confession wasn't honest and changed everything.",
    "He was known for honest conduct in office.",
    "He was known for no honest conduct in office.",
]

df = pd.DataFrame({"text": corpus})
df['id'] = df.reset_index().index

target_terms = ["cruel", "honest"]
data = extract_targets(df, target_terms, nlp)
print(data)

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