import numpy as np
import csv
import os
import re
import torch
from tqdm import tqdm
from typing import Callable, Iterable, Tuple, Optional


# -----------------------------
# Utilities
# -----------------------------

def ablate_word(sentence: str, word: str) -> str:
    return ' '.join(
        w for w in sentence.split()
        if w.lower() != word.lower()
    )


# -----------------------------
# Core metric engine
# -----------------------------

def compute_metrics(
    corpus: Iterable[Tuple],
    output_path: str,
    *,
    embed_fn: Callable[[list[str]], np.ndarray],
    sentiment_score_fn: Callable[[str], float],
    sentiment_label_fn: Callable[[str], int],
    attribution_fn: Optional[Callable[[str, str], float]] = None,
    causal_fn: Optional[Callable[[str, str], dict]] = None,
    P_final: Optional[np.ndarray] = None,
    batch_size: int = 16,
):
    """
    Pure metric computation engine.

    Parameters
    ----------
    corpus : iterable
        Each item must be:
        (sentence, target_words, id, dep, neg, created_utc)

    embed_fn : callable
        sentences -> (n, d) numpy array

    sentiment_score_fn : callable
        sentence -> signed scalar sentiment score

    sentiment_label_fn : callable
        sentence -> sentiment class index

    attribution_fn : callable
        (sentence, target_word) -> scalar attribution score

    P_final : np.ndarray or None
        Optional INLP projection matrix (d x d)

    batch_size : int
        Mini-batch size for embedding computation
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "id",
        "sentence",
        "target_word",
        "dependency",
        "is_negated",
        "sentence_wordcount",
        "NEC",
        "RSS",
        "NEC_INLP",
        "RSS_INLP",
        "SC",
        "SC_signed",
        "cw_SC_signed",
        "DW",
        "cf_effect",
        "expected_cf_score",
        "cf_baseline",
        "cf_n_candidates",
        "targetPolarity",
        #"sentencePolarity",
        "sentenceSentimentProb"
    ]

    with open(output_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for batch_start in tqdm(
            range(0, len(corpus), batch_size),
            desc="Processing corpus",
            unit="batch"
        ):
            batch = corpus[batch_start:batch_start + batch_size]
            sentences = [item[0] for item in batch]

            # --- full sentence representations ---
            full_embeddings = embed_fn(sentences)
            if P_final is not None:
                full_embeddings_proj = (P_final @ full_embeddings.T).T
            else:
                full_embeddings_proj = None

            full_sentiments = [sentiment_score_fn(s) for s in sentences]

            # --- per-item processing ---
            for i, (sentence, target_words, id_, dep_, neg_) in enumerate(batch):
                if isinstance(target_words, str):
                    target_words = (target_words,)

                for target_word, dep_word, neg_flag in zip(
                    target_words, dep_, neg_
                ):
                    full_emb = full_embeddings[i]
                    full_sentiment = full_sentiments[i]

                    if P_final is not None:
                        full_emb_proj = full_embeddings_proj[i]

                    # --- ablations ---
                    words = [
                        w for w in sentence.split()
                        if w.lower() != target_word.lower()
                    ]

                    ablated_sentences = [ablate_word(sentence, w) for w in words]
                    ablated_sentences.append(ablate_word(sentence, target_word))

                    ablated_embeddings = embed_fn(ablated_sentences)
                    ablated_sentiments = [
                        sentiment_score_fn(s) for s in ablated_sentences
                    ]

                    if P_final is not None:
                        ablated_embeddings_proj = (P_final @ ablated_embeddings.T).T

                    # --- semantic metrics ---
                    delta_all = np.linalg.norm(
                        full_emb - ablated_embeddings[:-1],
                        axis=1
                    )
                    delta_target = np.linalg.norm(
                        full_emb - ablated_embeddings[-1]
                    )

                    semantic_sum = np.sum(delta_all) + delta_target
                    semantic_mean = np.mean(delta_all) if len(delta_all) else 0.0

                    if P_final is not None:
                        delta_all_proj = np.linalg.norm(
                            full_emb_proj - ablated_embeddings_proj[:-1],
                            axis=1
                        )
                        delta_target_proj = np.linalg.norm(
                            full_emb_proj - ablated_embeddings_proj[-1]
                        )

                        semantic_sum_proj = np.sum(delta_all_proj) + delta_target_proj
                        semantic_mean_proj = (
                            np.mean(delta_all_proj) if len(delta_all_proj) else 0.0
                        )
                    else:
                        delta_target_proj = semantic_sum_proj = semantic_mean_proj = 0.0

                    # --- sentiment metrics ---
                    sentiment_contributions = [
                        abs(full_sentiment - s)
                        for s in ablated_sentiments
                    ]

                    sentiment_total = sum(sentiment_contributions)
                    sentiment_diff = full_sentiment - ablated_sentiments[-1]

                    SC = (
                        abs(sentiment_diff) / sentiment_total
                        if sentiment_total else 0.0
                    )

                    SC_signed = (
                        sentiment_diff / sentiment_total if sentiment_total else 0.0
                        if sentiment_total else 0.0
                    )

                    cw_SC_signed = (
                        (sentiment_diff / sentiment_total) * (2 * full_sentiment - 1)
                        if sentiment_total else 0.0
                    )

                    target_polarity = (
                        1 if sentiment_diff > 0
                        else -1 if sentiment_diff < 0
                        else 0
                    )

                    if attribution_fn is not None:
                        DW = attribution_fn(sentence, target_word)
                    else:
                        DW = None

                    if causal_fn is not None:
                        causal_result = causal_fn(sentence, target_word)
                    else:
                        causal_result = None

                    writer.writerow({
                        "id": id_,
                        "sentence": sentence,
                        "target_word": target_word,
                        "dependency": dep_word,
                        "is_negated": neg_flag,
                        "sentence_wordcount": len(re.findall(r'\w+', sentence)),
                        "NEC": delta_target / semantic_sum if semantic_sum else 0.0,
                        "RSS": delta_target - semantic_mean,
                        "NEC_INLP": (
                            delta_target_proj / semantic_sum_proj
                            if semantic_sum_proj else 0.0
                        ),
                        "RSS_INLP": delta_target_proj - semantic_mean_proj,
                        "SC": SC,
                        "SC_signed": SC_signed,
                        "cw_SC_signed": cw_SC_signed,
                        "DW": DW,
                        "cf_effect": causal_result["effect"] if causal_result else None,
                        "expected_cf_score": causal_result["expected_cf_score"] if causal_result else None,
                        "cf_baseline": causal_result["baseline"] if causal_result else None,
                        "cf_n_candidates": causal_result["n_candidates"] if causal_result else None,
                        "targetPolarity": target_polarity,
                        #"sentencePolarity": sentiment_label_fn(sentence),
                        "sentenceSentimentProb": full_sentiments[i]
                    })

def compute_metrics_roberta(
    corpus: Iterable[Tuple],
    output_path: str,
    *,
    causal_fn: Callable[[str, str], dict],
    batch_size: int = 16,
):
    """
    Causal counterfactual metric computation engine.

    Parameters
    ----------
    corpus : iterable
        Each item: (sentence, target_words, id, dep, neg)

    causal_fn : callable
        (sentence, target_word) -> dict with effect, expected_cf_score,
        baseline, n_candidates, replacement_entropy; or None if skipped

    batch_size : int
        Mini-batch size for processing
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fieldnames = [
        "id",
        "sentence",
        "target_word",
        "dependency",
        "is_negated",
        "sentence_wordcount",
        "cf_effect",
        "replacement_entropy",
        "expected_cf_score",
        "cf_baseline",
        "cf_n_candidates",
    ]

    with open(output_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for batch_start in tqdm(
            range(0, len(corpus), batch_size),
            desc="Processing corpus",
            unit="batch"
        ):
            batch = corpus[batch_start:batch_start + batch_size]

            for i, (sentence, target_words, id_, dep_, neg_) in enumerate(batch):
                if isinstance(target_words, str):
                    target_words = (target_words,)

                for target_word, dep_word, neg_flag in zip(target_words, dep_, neg_):

                    causal_result = causal_fn(sentence, target_word)

                    writer.writerow({
                        "id": id_,
                        "sentence": sentence,
                        "target_word": target_word,
                        "dependency": dep_word,
                        "is_negated": neg_flag,
                        "sentence_wordcount": len(re.findall(r'\w+', sentence)),
                        "cf_effect": -causal_result["effect"] if causal_result else None,
                        "replacement_entropy": causal_result["replacement_entropy"] if causal_result else None,
                        "expected_cf_score": causal_result["expected_cf_score"] if causal_result else None,
                        "cf_baseline": causal_result["baseline"] if causal_result else None,
                        "cf_n_candidates": causal_result["n_candidates"] if causal_result else None,
                    })