import pandas as pd
import spacy
from tqdm import tqdm
import os
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed, wait
from src.collect_corpus.helper.data_processing import read_zst_file, process_text


# Initialize spaCy in a function to avoid conflicts across processes
def init_nlp():
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    #if "parser" in nlp.pipe_names:
    #    nlp.add_pipe("custom_sentencizer", before="parser")
    #else:
    #    nlp.add_pipe("custom_sentencizer", last=True)
    return nlp

def extract_sentence(texts, target_terms, nlp, subsample=None):
    filtered_texts = []
    if subsample is not None:
        texts = itertools.islice(texts, subsample)

    for text in texts:
        doc = process_text(text["title"] + ". " + text["selftext"], nlp=nlp)
        if doc is None:
            continue

        else:
            for sentence in doc.sents:
                if not any(
                        token.text in target_terms["ADJ"] + target_terms["NOUN"]
                        for token in sentence
                ):
                    continue
                else:
                    # Spellchecker
                    # sentence = correct(sentence.text).lower()
                    # Re-do pipeline
                    # sentence = nlp(sentence)
                    # Get sentence back
                    # sentence = sentence.doc
                    # Check for each term in the target_terms whether it matches the sentence
                    term_status = {}
                    dependencies = {}
                    negations = {}
                    for term in target_terms["ADJ"]:
                        # Check if the term matches as asserted or negated
                        term_in_sentence = False
                        term_negated = False
                        term_dep = None
                        for token in sentence:
                            if token.text.lower() == term.lower() and token.pos_ == "ADJ":
                                if token.text == "sick":
                                    if any(
                                            child.pos_ == "PREP" and child.text in ["of"]
                                            for child in token.children
                                    ):
                                        continue

                                head_of_ADJ = token.head
                                if head_of_ADJ.pos_ in ["PRON", "NOUN"] and token.dep_ == "amod":
                                    # Adjective modifies a noun or pronoun (e.g., "an unhealthy lifestyle")
                                    root_verb = head_of_ADJ.head
                                elif head_of_ADJ.pos_ in ["PRON", "NOUN"] and token.dep_ == "ccomp":
                                    # Adjective is part of a clausal complement (e.g., "I think it's unhealthy")
                                    root_verb = next((child for child in head_of_ADJ.children if
                                                      child.dep_ in ["ROOT", "xcomp"]), None)
                                elif head_of_ADJ.dep_ in ["ROOT", "xcomp"]:
                                    # Adjective is the root or part of an open clause complement (e.g., "He seems sick")
                                    root_verb = head_of_ADJ
                                elif head_of_ADJ.pos_ == "AUX":
                                    # Handle auxiliary verbs (e.g., "He is unhealthy")
                                    root_verb = head_of_ADJ.head
                                else:
                                    root_verb = None

                                if root_verb:
                                    term_in_sentence = True

                                    # Check if the adjective itself is negated (direct negation)
                                    if any(child.dep_ == "neg" for child in token.children):
                                        term_negated = True

                                    # If the root verb is an auxiliary verb and is negated, it might affect the adjective
                                    elif (
                                            # The main verb is an aux verb
                                            root_verb.dep_ == "aux" and
                                            # The main verb is negated
                                            any(child.dep_ == "neg" for child in root_verb.children) and not
                                            # The adjective itself is not negated
                                            any(child.dep_ == "neg" for child in token.children) and
                                            # The adjective is child of the main verb
                                            token in root_verb.children
                                    ):
                                        term_negated = True

                                    # Check for negated clausal complements (e.g., "feeling healthy")
                                    elif (
                                            # The main verb is not an aux verb
                                            root_verb.dep_ != "aux" and
                                            # If the main verb is part of a clausal complement, none if it's siblings are negations
                                            any(
                                                sibling.dep_ == "neg"
                                                for ancestor in root_verb.ancestors if root_verb.dep_ == "xcomp"
                                                for sibling in ancestor.children
                                            ) and not
                                            # The adjective itself is not negated
                                            any(child.dep_ == "neg" for child in token.children)
                                    ):
                                        term_negated = True

                                    # Check if the main verb itself is negated, and that it impacts the adjective (e.g., "feeling healthy")
                                    elif (
                                            # The sentence contains a subject
                                            any(t.dep_ == "nsubj" for t in sentence) and
                                            # The main verb is not an aux verb
                                            root_verb.dep_ != "aux" and
                                            # The main verb is negated
                                            any(child.dep_ == "neg" for child in
                                                root_verb.children) and not
                                            # The target adjective itself is not negated
                                            any(child.dep_ == "neg" for child in token.children) and
                                            # The target adjective is a child of the root verb
                                            token in root_verb.children and not
                                            # The target adjective does not precede the root verb
                                            token.i < root_verb.i
                                    ):
                                        term_negated = True

                                    if term_in_sentence:
                                        term_dep = token.dep_

                        # Add the result for this term
                        term_status[term] = term_in_sentence
                        dependencies[term] = term_dep
                        negations[term] = term_negated

                    for term in target_terms["NOUN"]:
                        # Check if the term matches as asserted or negated
                        term_in_sentence = False
                        #term_negated = False
                        term_dep = None
                        for token in sentence:
                            if token.text.lower() == term.lower() and token.pos_ == "NOUN":
                                head_of_NOUN = token.head
                                if head_of_NOUN.dep_ in ["ROOT", "xcomp"]:
                                    # Adjective is the root or part of an open clause complement (e.g., "He seems sick")
                                    root_verb = head_of_NOUN
                                elif head_of_NOUN.pos_ == "AUX":
                                    # Handle auxiliary verbs (e.g., "He is unhealthy")
                                    root_verb = head_of_NOUN.head
                                else:
                                    root_verb = None

                                if root_verb:
                                    # TODO: handle negation???
                                    term_in_sentence = True
                                    term_dep = token.dep_

                        # Add the result for this term
                        term_status[term] = term_in_sentence
                        dependencies[term] = term_dep
                        #negations[term] = term_negated

                    # Append the sentence and its term status to the results
                    if any(term_status.values()):
                        filtered_texts.append({
                            "sentence": sentence.text,
                            # "root_verb": root_verb,
                            "target_dep_": dependencies,
                            # "term_status": term_status,
                            "target_neg": negations,
                            "created_utc": text["created_utc"],
                            "subreddit": text["subreddit"],
                            "id": text["id"],
                            "author_flair": text["author_flair_text"],
                        })

    return filtered_texts

# Write results incrementally to avoid memory overload
def write_results_to_csv(results, output_file, mode="a"):
    df = pd.DataFrame(results)
    for col in ['target_dep_', 'target_neg']:
        if col in df:
            df_expanded = df[col].apply(pd.Series)
            suffix = "_dep" if col == "target_dep_" else "_neg"
            df_expanded.columns = [f"{col_name}{suffix}" for col_name in df_expanded.columns]
            df = pd.concat([df.drop(columns=[col]), df_expanded], axis=1)

    write_header = not os.path.isfile(output_file) or mode == "w"
    df.to_csv(output_file, index=False, mode=mode, header=write_header)

def process_batch(batch, target_terms):
    import spacy
    # Load nlp inside each worker process to avoid serialization issues
    nlp = spacy.load("en_core_web_sm")  # or your custom model
    return extract_sentence(batch, target_terms, nlp)

from tqdm import tqdm

def process_single_file_parallel(file_path, output_file, target_terms, batch_size=5000, max_workers=4, max_futures=10):
    file_size = os.path.getsize(file_path)
    batch = []
    futures = []

    bytes_pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Reading file")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        json_generator = read_zst_file(file_path)
        for json_obj, bytes_read in json_generator:
            batch.append(json_obj)
            bytes_pbar.n = bytes_read
            bytes_pbar.refresh()

            if len(batch) >= batch_size:
                while len(futures) >= max_futures:
                    done, futures = wait_for_some(futures)
                    for future in done:
                        results = future.result()
                        write_results_to_csv(results, output_file, mode="a")
                futures.append(executor.submit(process_batch, batch, target_terms))
                batch = []

        if batch:
            futures.append(executor.submit(process_batch, batch, target_terms))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Batches"):
            results = future.result()
            write_results_to_csv(results, output_file, mode="a")

    bytes_pbar.close()

def wait_for_some(futures):
    done, not_done = wait(futures, return_when='FIRST_COMPLETED')
    return done, list(not_done)

# Main script
if __name__ == "__main__":
    nlp = init_nlp()
    input_file = "../../output/corpora/output_submissions.zst"
    output_file = "../../output/corpora/corpus.csv"
    target_terms = {
        "ADJ": ["cruel", "compassionate", "rude", "friendly", "manipulative", "honest", "selfish", "generous", "wrong", "right", "good", "bad"],
        "NOUN": []
    }

    process_single_file_parallel(input_file, output_file, target_terms, batch_size=100, max_workers=6)
