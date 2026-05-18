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

SEED = 3764
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

# Initialize NLP
nlp = init_nlp()

# Example sentence
sentence = "Based on how she handled the scholarship fund, most people did not find her to be generous."

# Run spaCy parse
doc = nlp(sentence)

for token in doc:
    print(f"Token: {token.text:<10} Dep: {token.dep_:<6} Head: {token.head.text} pos: {token.pos_}")


# Inspect the dependency tag of "cruel"
cruel_token = [token for token in doc if token.text.lower() == "generous"][0]
print(f"'generous' token dep_: {cruel_token.dep_}, head: {cruel_token.head.text}, pos: {cruel_token.pos_}")

# Create DataFrame and extract targets
df = pd.DataFrame({"text": [sentence], "id": [1]})
target_terms = ["generous"]
data = extract_targets(df, target_terms, nlp)
print("Extracted targets:", data)

if False:
    # --- Setup IG model for demonstration ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ig_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    ig_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    ig_model.eval().to(device)

    # IG function with robust subtoken matching
    def compute_ig_target_contribution(sentence, target_word, target_label=1):
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
        target_tokens = ig_tokenizer.tokenize(target_word)

        # Robust subtoken matching
        indices = [i for i, t in enumerate(tokens) if any(t.replace("##", "") == tt for tt in target_tokens)]

        # Debug print
        print("Sentence:", sentence)
        print("Tokens:", tokens)
        print("Target tokens:", target_tokens)
        print("Matched indices:", indices)

        scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        norm_scores = scores / np.sum(np.abs(scores)) if np.sum(np.abs(scores)) else scores

        target_contribution = np.sum([norm_scores[i] for i in indices]) if indices else 0.0
        return float(target_contribution)

    # Test IG contribution for "cruel"
    contribution = compute_ig_target_contribution(sentence, "cruel")
    print("IG contribution for 'cruel':", contribution)