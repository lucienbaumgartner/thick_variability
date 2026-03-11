import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# --- Step 0: Setup ---

device = torch.device("mps")

# Load high-quality sentence embedding model (no sentiment finetuning)
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embedder.to(device)

# --- Step 1: Load SST-2 data (67k sentences with binary sentiment labels) ---

dataset = load_dataset("sst2", split="train")  # Using HuggingFace Datasets

sentences = dataset["sentence"]
labels = dataset["label"]

# Optional: downsample for faster experimentation
# sentences = sentences[:10000]
# labels = labels[:10000]

# --- Step 2: Compute sentence embeddings ---

batch_size = 64
X = embedder.encode(sentences, convert_to_numpy=True, batch_size=batch_size, device=device)
y = np.array(labels)

# Optional: split into train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# --- Step 3: INLP helper functions ---

def train_linear_probe(X, y):
    clf = LogisticRegression(max_iter=1000, solver='liblinear')
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

# --- Step 4: Run INLP on training data ---

P_final = iterative_nullspace_projection(X_train, y_train, n_iter=10, acc_threshold=0.6)

# --- Step 5: Use P_final to remove sentiment information ---

def remove_sentiment(embedding):
    return P_final @ embedding

# Example usage
test_sentence = "This was a really disappointing film."
test_emb = embedder.encode(test_sentence)
test_emb_no_sent = remove_sentiment(test_emb)

print("\nOriginal embedding norm:", np.linalg.norm(test_emb))
print("Sentiment-removed embedding norm:", np.linalg.norm(test_emb_no_sent))
