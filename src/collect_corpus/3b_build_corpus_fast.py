import os
import json
import csv
from tqdm import tqdm

from src.collect_corpus.helper.data_processing import read_zst_file
from src.compute_metrics.helper.dep_parsing import (
    extract_targets_from_sentence,
    init_nlp
)

# ---------------------------------------------------
# spaCy (IMPORTANT: disable external parallelization issues)
# ---------------------------------------------------
nlp = init_nlp()


# ---------------------------------------------------
# STREAM ITEM FILTER
# ---------------------------------------------------
def is_valid_item(item):
    if item.get("media") is not None:
        return False

    selftext = item.get("selftext", "").strip()
    if selftext in {"", "[deleted]", "[removed]"}:
        return False

    return True


# ---------------------------------------------------
# MAIN PIPELINE (FAST + PARALLEL spaCy)
# ---------------------------------------------------
def process_single_file_parallel(file_path, output_file, target_terms, batch_size=256, n_process=4):

    if os.path.exists(output_file):
        os.remove(output_file)

    # CSV writer (streaming, no pandas overhead)
    fieldnames = [
        "sentence", "id", "author", "subreddit",
        "crosspost_parent", "created_utc",
        "target_dep_", "target_neg"
    ]

    f_out = open(output_file, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    json_generator = read_zst_file(file_path)

    batch_items = []
    batch_texts = []
    batch_meta = []

    file_size = os.path.getsize(file_path)

    with tqdm(total=file_size, unit="B", unit_scale=True, desc="Processing") as pbar:

        for item, bytes_read in json_generator:

            pbar.n = bytes_read
            pbar.refresh()

            if not is_valid_item(item):
                continue

            text = f"{item['title']}. {item['selftext']}"

            batch_items.append(item)
            batch_texts.append(text)

            batch_meta.append({
                "id": item["id"],
                "author": item.get("author"),
                "subreddit": item.get("subreddit"),
                "created_utc": item.get("created_utc"),
                "crosspost_parent": item.get("crosspost_parent"),
            })

            # -----------------------------------------
            # PROCESS BATCH
            # -----------------------------------------
            if len(batch_texts) >= batch_size:

                docs = nlp.pipe(batch_texts, n_process=n_process)

                for doc, meta, item_raw in zip(docs, batch_meta, batch_items):

                    for sentence in doc.sents:

                        deps, negs = extract_targets_from_sentence(sentence, target_terms)

                        if not deps:
                            continue

                        writer.writerow({
                            "sentence": sentence.text,
                            "id": meta["id"],
                            "author": meta["author"],
                            "subreddit": meta["subreddit"],
                            "crosspost_parent": meta["crosspost_parent"],
                            "created_utc": meta["created_utc"],
                            "target_dep_": deps,
                            "target_neg": negs,
                        })

                # reset batch
                batch_items.clear()
                batch_texts.clear()
                batch_meta.clear()

        # -----------------------------------------
        # FINAL FLUSH
        # -----------------------------------------
        if batch_texts:

            docs = nlp.pipe(batch_texts, n_process=n_process)

            for doc, meta, item_raw in zip(docs, batch_meta, batch_items):

                for sentence in doc.sents:

                    deps, negs = extract_targets_from_sentence(sentence, target_terms)

                    if not deps:
                        continue

                    writer.writerow({
                        "sentence": sentence.text,
                        "id": meta["id"],
                        "author": meta["author"],
                        "subreddit": meta["subreddit"],
                        "crosspost_parent": meta["crosspost_parent"],
                        "created_utc": meta["created_utc"],
                        "target_dep_": deps,
                        "target_neg": negs,
                    })

    f_out.close()


# ---------------------------------------------------
# LOAD TARGET TERMS
# ---------------------------------------------------
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

    process_single_file_parallel(
        file_path=input_file,
        output_file=output_file,
        target_terms=target_terms,
        batch_size=256,
        n_process=8
    )