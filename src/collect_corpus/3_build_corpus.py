import os
import pandas as pd
from tqdm import tqdm
import json

from src.collect_corpus.helper.data_processing import read_zst_file
from src.compute_metrics.helper.dep_parsing import (
    extract_targets_from_sentence,
    init_nlp
)

# ---------------------------------------------------
# spaCy (single process)
# ---------------------------------------------------
nlp = init_nlp()

# ---------------------------------------------------
# WRITE RESULTS
# ---------------------------------------------------
def write_results_to_csv(results, output_file, mode="a"):
    if not results:
        return

    df = pd.DataFrame(results)

    write_header = not os.path.exists(output_file) or mode == "w"
    df.to_csv(output_file, index=False, mode=mode, header=write_header)


# ---------------------------------------------------
# PROCESS SINGLE ITEM (no batching)
# ---------------------------------------------------
def process_item(item, target_terms):
    # --- media filter ---
    if item.get("media") is not None:
        return []

    # --- selftext filter ---
    selftext = item.get("selftext", "").strip()
    if selftext in {"", "[deleted]", "[removed]"}:
        return []

    text = f"{item['title']}. {item['selftext']}"
    doc = nlp(text)

    if doc is None:
        return []

    results = []

    metadata = {
        "id": item["id"],
        "author": item.get("author"),
        "subreddit": item.get("subreddit"),
        "created_utc": item.get("created_utc"),
        "crosspost_parent": item.get("crosspost_parent"),
    }

    for sentence in doc.sents:
        deps, negs = extract_targets_from_sentence(sentence, target_terms)

        if not deps:
            continue

        results.append({
            "sentence": sentence.text,
            "id": metadata.get("id"),
            "author": metadata.get("author"),
            "subreddit": metadata["subreddit"],
            "crosspost_parent": metadata.get("crosspost_parent"),
            "created_utc": metadata["created_utc"],
            "target_dep_": deps,
            "target_neg": negs,
        })

    return results


# ---------------------------------------------------
# MAIN PIPELINE (SEQUENTIAL)
# ---------------------------------------------------
def process_single_file_sequential(file_path, output_file, target_terms):
    file_size = os.path.getsize(file_path)

    json_generator = read_zst_file(file_path)

    all_results = []

    if os.path.exists(output_file):
        os.remove(output_file)

    with tqdm(total=file_size, unit="B", unit_scale=True, desc="Reading ZST") as pbar:

        for item, bytes_read in json_generator:

            results = process_item(item, target_terms)

            if results:
                write_results_to_csv(results, output_file, mode="a")

            pbar.n = bytes_read
            pbar.refresh()

tt_path = "../../input/target_terms.json"
with open(tt_path, "r", encoding="utf-8") as f:
    data = json.load(f)

target_terms = []
for item in data:
    target_terms.extend(item["pos"])
    target_terms.extend(item["neg"])

# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":

    input_file = "../../output/corpora/reddit/term_filtering/output_submissions.zst"
    output_file = "../../output/corpora/reddit/built_corpus/corpus_with_duplicates.csv"

    process_single_file_sequential(
        file_path=input_file,
        output_file=output_file,
        target_terms=target_terms
    )