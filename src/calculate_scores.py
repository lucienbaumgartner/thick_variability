from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
from sentence_transformers import SentenceTransformer
from captum.attr import IntegratedGradients
import numpy as np
import torch
import json
import os

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load embedding and sentiment models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if device.type == "mps" else -1
)

# Load tokenizer and model for IG
ig_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
ig_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
ig_model.eval().to(device)

# Function to ablate target word
def ablate_word(sentence: str, word: str) -> str:
    return ' '.join([w for w in sentence.split() if w.lower() != word.lower()])

# Function to compute sentiment score
def get_sentiment_score(sentence: str) -> float:
    result = sentiment_pipeline(sentence)[0]
    score = result['score']
    label = result['label'] # 'POSITIVE' or 'NEGATIVE'
    return score if label == 'POSITIVE' else -score

def get_label_index(sentence):
    result = sentiment_pipeline(sentence)[0]
    label = result['label']  # 'POSITIVE' or 'NEGATIVE'
    return 1 if label == 'POSITIVE' else 0

# Function to compute normalized IG score of target word
def compute_ig_target_contribution(sentence, target_word, target_label=None):
    if target_label is None:
        target_label = get_label_index(sentence)
    inputs = ig_tokenizer(sentence, return_tensors="pt")
    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Get input embeddings from the model's embedding layer
    embeddings = ig_model.distilbert.embeddings.word_embeddings(input_ids)
    embeddings.requires_grad_()  # Enable gradients on embeddings

    def forward_func(inputs_embeds, attention_mask):
        outputs = ig_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        # Use softmax on logits, take target label prob
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

    # Normalize scores
    norm_scores = scores / np.sum(np.abs(scores)) if np.sum(np.abs(scores)) else scores

    # Find target word token(s)
    target_tokens = ig_tokenizer.tokenize(target_word)
    indices = [i for i, t in enumerate(tokens) if t in target_tokens]
    target_contribution = np.sum([norm_scores[i] for i in indices]) if indices else 0.0

    return float(target_contribution)

# Main function to compute metrics
def compute_metrics_batch(corpus: list) -> list:
    sentences = [item[0] for item in corpus]
    full_embeddings = embedder.encode(sentences, convert_to_numpy=True, batch_size=64) # Embeddings
    full_sentiments = [get_sentiment_score(s) for s in sentences] # Sentence sentiment

    results = []
    for i, (sentence, target_word) in enumerate(corpus):
        full_emb = full_embeddings[i]
        full_sentiment = full_sentiments[i]

        words = [w for w in sentence.split() if w.lower() != target_word.lower()] # All words except the target
        ablated_sentences = [ablate_word(sentence, w) for w in words]
        ablated_sentences.append(ablate_word(sentence, target_word)) # The ablation of the target is added last

        ablated_embeddings = embedder.encode(ablated_sentences, convert_to_numpy=True, batch_size=64)
        ablated_sentiments = [get_sentiment_score(s) for s in ablated_sentences]

        delta_all = np.linalg.norm(full_emb - ablated_embeddings[:-1], axis=1) # Semantic contribution of all words except target
        delta_target = np.linalg.norm(full_emb - ablated_embeddings[-1]) # Semantic contribution of target
        semantic_sum = np.sum(delta_all) + delta_target
        semantic_mean = np.mean(delta_all) if len(delta_all) > 0 else 0.0

        #sentiment_contributions = [abs(full_sentiment - s) for s in ablated_sentiments[:-1]]
        sentiment_contributions = [abs(full_sentiment - s) for s in ablated_sentiments] # Sentiment contribution of all words including target
        sentiment_total = sum(sentiment_contributions)

        sentiment_diff = full_sentiment - ablated_sentiments[-1]
        normalized_sentiment_contribution = abs(sentiment_diff) / sentiment_total if sentiment_total else 0.0
        sentiment_polarity = 1 if sentiment_diff > 0 else (-1 if sentiment_diff < 0 else 0)

        ig_target_score = compute_ig_target_contribution(sentence, target_word)

        results.append({
            "sentence": sentence,
            "target_word": target_word,
            "Normalized Semantic Contribution": delta_target / semantic_sum if semantic_sum else 0.0,
            "Residual Semantic Shift": delta_target - semantic_mean,
            "Normalized Sentiment Contribution": normalized_sentiment_contribution,
            "Sentiment Target": sentiment_polarity,
            "Sentence Polarity": get_label_index(sentence),
            "IG Target Word Contribution": ig_target_score
        })

    return results

# Example corpus
corpus = [
    ("John is a cruel and calculating person.", "cruel"),
    ("It was cruel of her to ignore his pleas.", "cruel"),
    ("The cruel winter froze the crops.", "cruel"),
    ("She gave him a cruel smile.", "cruel"),
    ("His cruel jokes made everyone uncomfortable.", "cruel"),
    ("The punishment was cruel but just.", "cruel"),
    ("Some say the villain is not truly cruel, just misunderstood.", "cruel"),
    ("That cruel twist of fate changed everything.", "cruel"),
    ("He endured cruel treatment in prison.", "cruel")
]

results = compute_metrics_batch(corpus)
for r in results:
    print(r)

def convert_floats(obj):
    if isinstance(obj, dict):
        return {k: convert_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_floats(i) for i in obj]
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    else:
        return obj

os.makedirs("../output", exist_ok=True)

with open("../output/scores.json", "w") as f:
    json.dump(convert_floats(results), f, indent=2)