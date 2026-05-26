import numpy as np
import torch
from typing import Callable, Optional


def causal_sentiment_effect_roberta(
    sentence: str,
    target_word: str,
    mlm_model,
    mlm_tokenizer,
    sentiment_fn: Callable[[str], float],
    top_p: float = 0.9,
    top_n: int = 50,
    device: str = "cpu",
) -> Optional[dict]:
    """
    Estimates the causal effect of a target word on sentiment via
    RoBERTa-based masked language modeling using weighted expectation.

    The effect is defined as:
        E[sentiment | context, word replaced] - sentiment(original sentence)

    A positive effect means the target word pushes sentiment higher than
    a typical word in that context would. A negative effect means it
    suppresses sentiment relative to context.

    Sentences where the target word occurs more than once are skipped.
    Multi-subword target words are masked jointly; replacement words are
    sampled from the top-n most probable tokens at the first mask position,
    weighted by their MLM probability.

    Args:
        sentence:       Input sentence
        target_word:    Word whose causal effect to estimate
        mlm_model:      RoBERTa MLM model (e.g. RobertaForMaskedLM)
        mlm_tokenizer:  Corresponding RoBERTa tokenizer
        sentiment_fn:   Function mapping a string to a scalar sentiment score
        top_p:          Nucleus sampling threshold for candidate filtering
        top_n:          Number of top tokens to consider for weighted expectation
        device:         Torch device

    Returns:
        dict with effect, expected_cf_score, baseline, n_candidates
        or None if the sentence is skipped
    """
    # -------------------------------------------------
    # 1. Tokenize and find target word span
    # -------------------------------------------------
    encoding = mlm_tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    input_ids = encoding["input_ids"][0]
    offset_mapping = encoding["offset_mapping"][0].tolist()

    target_lower = target_word.lower()
    target_spans = []

    # Scan all possible start positions using character offsets directly
    for start_idx, (char_start, char_end) in enumerate(offset_mapping):
        # Skip special tokens
        if char_start == char_end:
            continue

        # Try extending the span from this start position
        for end_idx in range(start_idx, len(offset_mapping)):
            end_char_start, end_char_end = offset_mapping[end_idx]

            # Skip special tokens as end position
            if end_char_start == end_char_end:
                break

            # Reconstruct text covered by tokens start_idx..end_idx
            reconstructed = sentence[char_start:end_char_end].lower().strip()

            if reconstructed == target_lower:
                target_spans.append(list(range(start_idx, end_idx + 1)))
                break
            elif len(reconstructed) > len(target_lower):
                # Overshot — no point extending further
                break

    # -------------------------------------------------
    # 2. Skip if target word is absent or occurs multiple times
    # -------------------------------------------------
    if len(target_spans) == 0:
        return None
    if len(target_spans) > 1:
        return None

    target_indices = target_spans[0]

    # -------------------------------------------------
    # 3. Baseline sentiment
    # -------------------------------------------------
    base_score = sentiment_fn(sentence)

    # -------------------------------------------------
    # 4. Build masked input
    # -------------------------------------------------
    masked_ids = input_ids.clone()
    for idx in target_indices:
        masked_ids[idx] = mlm_tokenizer.mask_token_id

    masked_input = {
        "input_ids": masked_ids.unsqueeze(0).to(device),
        "attention_mask": encoding["attention_mask"].to(device)
    }

    # -------------------------------------------------
    # 5. Get MLM logits at first mask position
    # -------------------------------------------------
    with torch.no_grad():
        outputs = mlm_model(**masked_input)
    logits = outputs.logits[0]

    first_mask_pos = target_indices[0]
    probs = torch.softmax(logits[first_mask_pos], dim=-1)

    # Top-p filtering
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=0)
    cutoff_mask = cum_probs <= top_p
    cutoff_mask[0] = True
    valid_probs = sorted_probs[cutoff_mask]
    valid_idx = sorted_idx[cutoff_mask]

    # Further truncate to top_n for computational tractability
    top_n_actual = min(top_n, len(valid_idx))
    valid_probs = valid_probs[:top_n_actual]
    valid_idx = valid_idx[:top_n_actual]

    # Renormalize after truncation
    valid_probs = valid_probs / valid_probs.sum()

    # Calculate entropy
    entropy = -torch.sum(valid_probs * torch.log(valid_probs + 1e-10)).item()

    # -------------------------------------------------
    # 6. Compute weighted expectation
    # -------------------------------------------------
    span_start = offset_mapping[target_indices[0]][0]
    span_end = offset_mapping[target_indices[-1]][1]

    weighted_scores = []
    for prob, token_id in zip(valid_probs.tolist(), valid_idx.tolist()):
        replacement_word = mlm_tokenizer.decode([token_id]).strip()
        counterfactual = sentence[:span_start] + replacement_word + sentence[span_end:]
        cf_score = sentiment_fn(counterfactual)
        weighted_scores.append(prob * cf_score)

    expected_cf_score = float(sum(weighted_scores))
    effect = expected_cf_score - base_score

    return {
        "effect": effect,
        "expected_cf_score": expected_cf_score,
        "baseline": base_score,
        "n_candidates": top_n_actual,
        "replacement_entropy": entropy,
    }

import numpy as np
import torch
from captum.attr import IntegratedGradients
from typing import Callable, Optional

def causal_sentiment_effect_roberta_verbose(
    sentence: str,
    target_word: str,
    mlm_model,
    mlm_tokenizer,
    sentiment_fn: Callable[[str], float],
    ig_model,
    ig_tokenizer,
    top_p: float = 0.9,
    top_n: int = 50,
    n_illustrative: int = 3,
    device: str = "cpu",
) -> Optional[dict]:
    """
    Verbose version of causal_sentiment_effect_roberta with IG attribution.
    Prints a detailed trace of the computation for illustration purposes.
    IG is computed for all three classes (negative, neutral, positive).
    """

    # -------------------------------------------------
    # 1. Tokenize and find target word span
    # -------------------------------------------------
    encoding = mlm_tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True
    )
    input_ids = encoding["input_ids"][0]
    offset_mapping = encoding["offset_mapping"][0].tolist()

    target_lower = target_word.lower()
    target_spans = []

    for start_idx, (char_start, char_end) in enumerate(offset_mapping):
        if char_start == char_end:
            continue
        for end_idx in range(start_idx, len(offset_mapping)):
            end_char_start, end_char_end = offset_mapping[end_idx]
            if end_char_start == end_char_end:
                break
            reconstructed = sentence[char_start:end_char_end].lower().strip()
            if reconstructed == target_lower:
                target_spans.append(list(range(start_idx, end_idx + 1)))
                break
            elif len(reconstructed) > len(target_lower):
                break

    # -------------------------------------------------
    # 2. Skip conditions
    # -------------------------------------------------
    if len(target_spans) == 0:
        print(f"[SKIP] Target word '{target_word}' not found in tokenization.")
        return None
    if len(target_spans) > 1:
        print(f"[SKIP] Target word '{target_word}' occurs multiple times.")
        return None

    target_indices = target_spans[0]

    # -------------------------------------------------
    # Print: input & masked sequence
    # -------------------------------------------------
    tokens = mlm_tokenizer.convert_ids_to_tokens(input_ids)

    print("=" * 60)
    print(f"  Input sentence   : {sentence}")
    print(f"  Target word      : {target_word}")
    print(f"  Token span       : indices {target_indices} → "
          f"{[tokens[i] for i in target_indices]}")
    print(f"  Masked sentence  : "
          f"{mlm_tokenizer.decode(input_ids, skip_special_tokens=True).replace(target_word, '<mask>')}")
    print("=" * 60)

    # -------------------------------------------------
    # 3. Baseline sentiment
    # -------------------------------------------------
    base_score = sentiment_fn(sentence)
    print(f"\n  Baseline sentiment ({target_word!r:>15}): {base_score:.4f}")

    # -------------------------------------------------
    # 4. Integrated Gradients attribution (all three classes)
    # -------------------------------------------------
    ig_inputs = ig_tokenizer(sentence, return_tensors="pt", truncation=True)
    ig_input_ids = ig_inputs["input_ids"].to(device)
    ig_attention_mask = ig_inputs["attention_mask"].to(device)

    ig_tokens = ig_tokenizer.convert_ids_to_tokens(ig_input_ids[0])

    # Get predicted class for reference
    with torch.no_grad():
        result = ig_model(ig_input_ids, attention_mask=ig_attention_mask)
    probs = torch.nn.functional.softmax(result.logits, dim=-1)
    predicted_label = probs.argmax(dim=-1).item()
    class_names = {0: "negative", 1: "neutral", 2: "positive"}
    print(f"  Predicted class  : {class_names[predicted_label]} "
          f"(p={probs[0, predicted_label].item():.4f})")

    # Find target word token indices using character offsets
    ig_inputs_with_offsets = ig_tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True
    )
    ig_offset_mapping = ig_inputs_with_offsets["offset_mapping"][0].tolist()

    indices = []
    for i, (char_start, char_end) in enumerate(ig_offset_mapping):
        if char_start == char_end:
            continue
        span_text = sentence[char_start:char_end].lower().strip()
        if span_text == target_lower:
            indices.append(i)
        elif target_lower.startswith(span_text):
            for j in range(i + 1, len(ig_offset_mapping)):
                end_char = ig_offset_mapping[j][1]
                reconstructed = sentence[char_start:end_char].lower().strip()
                if reconstructed == target_lower:
                    indices.extend(range(i, j + 1))
                    break
                elif len(reconstructed) > len(target_lower):
                    break

    # Run IG for all three classes
    ig_contributions = {}
    all_norm_scores = {}

    embeddings = ig_model.roberta.embeddings(ig_input_ids)

    print(f"\n  Integrated Gradients attribution (per class):")
    print(f"  {'Token':<20} {'Negative':>12} {'Neutral':>12} {'Positive':>12}")
    print(f"  {'-'*56}")

    for class_idx, class_name in class_names.items():

        embeddings_cls = embeddings.clone().detach().requires_grad_(True)

        def forward_func(inputs_embeds, attention_mask, cls=class_idx):
            outputs = ig_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
            return torch.nn.functional.softmax(outputs.logits, dim=-1)[..., cls]

        ig_attr = IntegratedGradients(forward_func)
        attributions, _ = ig_attr.attribute(
            inputs=embeddings_cls,
            additional_forward_args=(ig_attention_mask,),
            return_convergence_delta=True
        )

        scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
        norm_scores = (
            scores / np.sum(np.abs(scores))
            if np.sum(np.abs(scores)) else scores
        )
        all_norm_scores[class_name] = norm_scores
        ig_contributions[class_name] = (
            float(np.sum([norm_scores[i] for i in indices]))
            if indices else 0.0
        )

    # Print token-level attributions for all classes
    for i, tok in enumerate(ig_tokens):
        scores_per_class = [all_norm_scores[c][i] for c in class_names.values()]
        marker = " ◄" if i in indices else ""
        if any(abs(s) > 0.01 for s in scores_per_class) or i in indices:
            print(f"  {tok:<20} "
                  f"{all_norm_scores['negative'][i]:>12.4f} "
                  f"{all_norm_scores['neutral'][i]:>12.4f} "
                  f"{all_norm_scores['positive'][i]:>12.4f}"
                  f"{marker}")

    print(f"\n  IG contribution of '{target_word}':")
    for class_name, contribution in ig_contributions.items():
        marker = " ◄ predicted" if class_names[predicted_label] == class_name else ""
        print(f"    [{class_name:<8}]: {contribution:>8.4f}{marker}")

    # -------------------------------------------------
    # 5. Build masked input for MLM
    # -------------------------------------------------
    masked_ids = input_ids.clone()
    for idx in target_indices:
        masked_ids[idx] = mlm_tokenizer.mask_token_id

    masked_input = {
        "input_ids": masked_ids.unsqueeze(0).to(device),
        "attention_mask": encoding["attention_mask"].to(device)
    }

    # -------------------------------------------------
    # 6. Get MLM logits
    # -------------------------------------------------
    with torch.no_grad():
        outputs = mlm_model(**masked_input)
    logits = outputs.logits[0]

    first_mask_pos = target_indices[0]
    probs = torch.softmax(logits[first_mask_pos], dim=-1)

    # Top-p filtering
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=0)
    cutoff_mask = cum_probs <= top_p
    cutoff_mask[0] = True
    valid_probs = sorted_probs[cutoff_mask]
    valid_idx = sorted_idx[cutoff_mask]

    # Truncate to top_n
    top_n_actual = min(top_n, len(valid_idx))
    valid_probs = valid_probs[:top_n_actual]
    valid_idx = valid_idx[:top_n_actual]
    valid_probs = valid_probs / valid_probs.sum()

    # Entropy
    entropy = -torch.sum(valid_probs * torch.log(valid_probs + 1e-10)).item()

    # -------------------------------------------------
    # Print: top replacement candidates
    # -------------------------------------------------
    print(f"\n  Top-p candidates : {len(valid_idx)} tokens "
          f"(top_p={top_p}, truncated to top_n={top_n_actual})")
    print(f"  Replacement entropy (RE): {entropy:.4f}")
    print(f"\n  {'Rank':<6} {'Token':<20} {'P(token)':<12} {'Cum. P':<10}")
    print(f"  {'-'*48}")
    cum = 0.0
    for rank, (p, idx) in enumerate(zip(valid_probs.tolist(), valid_idx.tolist())):
        if rank >= 10:
            break
        cum += p
        word = mlm_tokenizer.decode([idx]).strip()
        print(f"  {rank+1:<6} {word:<20} {p:<12.4f} {cum:<10.4f}")
    if top_n_actual > 10:
        print(f"  ... ({top_n_actual - 10} more candidates used in computation)")

    # -------------------------------------------------
    # 7. Compute weighted expectation
    # -------------------------------------------------
    span_start = offset_mapping[target_indices[0]][0]
    span_end = offset_mapping[target_indices[-1]][1]

    print(f"\n  Illustrative substitutions (top {n_illustrative}):")
    print(f"  {'Replacement':<20} {'P(token)':<12} {'Sentiment':<12} {'Weighted':<10}")
    print(f"  {'-'*54}")

    weighted_scores = []
    for rank, (prob, token_id) in enumerate(
        zip(valid_probs.tolist(), valid_idx.tolist())
    ):
        replacement_word = mlm_tokenizer.decode([token_id]).strip()
        counterfactual = sentence[:span_start] + replacement_word + sentence[span_end:]
        cf_score = sentiment_fn(counterfactual)
        weighted_scores.append(prob * cf_score)

        if rank < n_illustrative:
            print(f"  {replacement_word:<20} {prob:<12.4f} {cf_score:<12.4f} "
                  f"{prob * cf_score:<10.4f}")
            print(f"    → \"{counterfactual}\"")

    expected_cf_score = float(sum(weighted_scores))
    effect = expected_cf_score - base_score

    # -------------------------------------------------
    # Print: summary
    # -------------------------------------------------
    print(f"\n  {'─' * 48}")
    print(f"  Expected CF sentiment (RE-weighted) : {expected_cf_score:.4f}")
    print(f"  Baseline sentiment                  : {base_score:.4f}")
    print(f"  CEE (baseline - expected CF)        : {base_score - expected_cf_score:.4f}")
    print(f"  RE  (replacement entropy)           : {entropy:.4f}")
    print(f"  IG  [negative]                      : {ig_contributions['negative']:.4f}")
    print(f"  IG  [neutral]                       : {ig_contributions['neutral']:.4f}")
    print(f"  IG  [positive]                      : {ig_contributions['positive']:.4f}")
    print(f"  {'─' * 48}\n")

    return {
        "effect": effect,
        "expected_cf_score": expected_cf_score,
        "baseline": base_score,
        "n_candidates": top_n_actual,
        "replacement_entropy": entropy,
        "ig_negative": ig_contributions["negative"],
        "ig_neutral": ig_contributions["neutral"],
        "ig_positive": ig_contributions["positive"],
    }