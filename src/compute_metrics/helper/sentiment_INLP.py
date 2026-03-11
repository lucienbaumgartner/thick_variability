import numpy as np
from sklearn.linear_model import LogisticRegression
from datasets import load_dataset


# -----------------------------
# Linear probe
# -----------------------------

def train_linear_probe(X, y, *, random_state=1234, C=1.0):
    """
    Train a linear probe (L2-regularized logistic regression).
    """
    clf = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=random_state,
        C=C,
    )
    clf.fit(X, y)
    return clf


# -----------------------------
# Projection matrix
# -----------------------------

def projection_matrix(clf, eps=1e-8):
    """
    Construct an orthogonal projection matrix onto the nullspace
    of the probe direction.
    """
    # Binary classifier: coef_ shape is (1, d)
    w = clf.coef_.reshape(-1, 1)
    norm = np.linalg.norm(w)

    if norm < eps:
        raise ValueError(
            "Degenerate probe direction encountered (near-zero norm). "
            "INLP cannot proceed safely."
        )

    w = w / norm
    P = np.eye(w.shape[0]) - w @ w.T
    return P


# -----------------------------
# INLP core
# -----------------------------

def iterative_nullspace_projection(
    X,
    y,
    *,
    n_iter=10,
    acc_threshold=0.6,
    random_state=1234,
    C=1.0,
):
    """
    Iterative Nullspace Projection (INLP).

    Returns
    -------
    P_total : np.ndarray
        Cumulative projection matrix (d x d)
    """

    d = X.shape[1]
    P_total = np.eye(d)
    X_proj = X.copy()

    for i in range(n_iter):
        clf = train_linear_probe(
            X_proj,
            y,
            random_state=random_state,
            C=C,
        )

        acc = clf.score(X_proj, y)
        print(f"Iteration {i + 1}, probe accuracy: {acc:.3f}")

        if acc < acc_threshold:
            print(
                f"Stopping early at iteration {i + 1}: "
                f"accuracy below threshold ({acc_threshold})."
            )
            break

        P = projection_matrix(clf)

        # Project representations
        X_proj = (P @ X_proj.T).T

        # Compose projection operators
        P_total = P @ P_total

    print(f"INLP completed after {i + 1} iteration(s).")
    return P_total


# -----------------------------
# Public API
# -----------------------------

def generate_projection_matrix(
    *,
    embedder,
    split="train",
    n_iter=15,
    acc_threshold=0.6,
    batch_size=64,
    random_state=1234,
    C=1.0,
    data_path
):
    """
    Generate an INLP projection matrix using SST-2.

    Parameters
    ----------
    embedder : object
        Must expose `.encode(sentences, convert_to_numpy=True, batch_size=...)`

    Returns
    -------
    P_final : np.ndarray
        Cumulative INLP projection matrix
    """

    if embedder is None:
        raise ValueError("An embedder must be provided.")

    print("Loading SST-2 dataset...")
    data = load_dataset(data_path, split=split)

    sentences = data["sentence"]
    labels = np.array(data["label"])

    print("Encoding sentences for INLP...")
    X = embedder.encode(
        sentences,
        convert_to_numpy=True,
        batch_size=batch_size,
    )

    print("Generating INLP projection...")
    P_final = iterative_nullspace_projection(
        X,
        labels,
        n_iter=n_iter,
        acc_threshold=acc_threshold,
        random_state=random_state,
        C=C,
    )

    return P_final