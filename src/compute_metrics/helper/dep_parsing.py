import spacy
import itertools
from spacy.language import Language
from langdetect import detect
import language_tool_python
import re
from sympy import false

if false:
    # Initialize the LanguageTool instance
    tool = language_tool_python.LanguageTool('en-US')  # Use 'en-GB' for British English

    def correct(sentence):
        # Check the sentence
        matches = tool.check(sentence)

        # Automatically apply corrections
        corrected_sentence = language_tool_python.utils.correct(sentence, matches)

        return corrected_sentence

# Custom sentence tokenizer allowing newline characters as sentence punctuation
@Language.component("custom_sentencizer")
def custom_sentecizer(doc):
    # Iterate through token indices
    for i, token in enumerate(doc):
        if re.search(r"\n+|\s+", token.text):
            # If the current token is a newline, mark the next token as the start of a sentence
            doc[i].is_sent_start = False  # The newline itself is not a sentence start
            if i + 1 < len(doc):
                doc[i + 1].is_sent_start = True
    return doc

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def preprocess_text(text):
    # Only keep English text
    try:
        if detect(text) != 'en' or re.search("r\/", text):
            return None
    except:
        return None

    # Remove specific patterns with regex
    text = text.lower()
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"\b\w*;\w*\b(;)?", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\*|•|#", "", text)
    text = re.sub(r"\[(.*?)\]", "", text)
    text = re.sub(r"^.*?\]", "", text)
    text = re.sub(r"\[.*?$", "", text)
    text = re.sub(r"\((.*?)\)", "", text)
    text = re.sub(r"^.*?\)", "", text)
    text = re.sub(r"\(.*?$", "", text)
    text = re.sub(r"\b\w*&\w*\b", "", text)
    text = remove_emojis(text)

    return text

def process_text(text, nlp):
    # Check if the text is a float
    if isinstance(text, float):
        # Convert float to string
        text = str(text)

    # Process the text using spaCy
    text = preprocess_text(text)
    if text is not None:
        doc = nlp(text)
        return doc


# Initialize spaCy in a function to avoid conflicts across processes
def init_nlp():
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    #if "parser" in nlp.pipe_names:
    #    nlp.add_pipe("custom_sentencizer", before="parser")
    #else:
    #    nlp.add_pipe("custom_sentencizer", last=True)
    return nlp

def extract_targets(texts, target_terms, nlp, subsample=None):
    """
    Extract target adjectives and their negation status from sentences.
    Only adds adjectives that are actually present in the sentence.
    """
    import itertools
    filtered_texts = []
    if subsample is not None:
        texts = itertools.islice(texts, subsample)

    for _, row in texts.iterrows():
        doc = nlp(row["text"])
        if doc is None:
            continue

        for sentence in doc.sents:
            # Skip sentences with none of the target terms
            if not any(token.text.lower() in [t.lower() for t in target_terms] for token in sentence):
                continue

            dependencies = {}
            negations = {}

            for term in target_terms:
                term_dep = None
                term_negated = False

                for token in sentence:
                    if token.text.lower() != term.lower() or token.pos_ not in {"ADJ", "ADV"}:
                        continue

                    # Acceptable positions: amod modifying NOUN/PRON or acomp/ccomp/xcomp
                    if (token.dep_ == "amod" and token.head.pos_ in ["NOUN", "PRON", "ADJ", "ADP"]) or token.dep_ in ["acomp", "ccomp", "xcomp", "advcl", "oprd", "advmod", "pcomp", "amod"]:
                        term_dep = token.dep_

                        # --- 1. Direct negation on the adjective ---
                        if any(child.dep_ == "neg" for child in token.children):
                            term_negated = True

                        # --- 2. Negation on verbs affecting the adjective ---
                        # Define clause boundaries BEFORE the ancestor loop
                        CLAUSE_BOUNDARIES = {"ccomp", "advcl", "relcl", "acl", "xcomp"}

                        for ancestor in token.ancestors:
                            # Stop if we cross ANY clause boundary, not just ccomp
                            if token.dep_ in CLAUSE_BOUNDARIES:
                                break
                            # Also stop if any ancestor on the path is itself a clause root
                            if ancestor.dep_ in CLAUSE_BOUNDARIES:
                                break
                            if ancestor.pos_ in {"AUX", "VERB"}:
                                if any(child.dep_ == "neg" for child in ancestor.children):
                                    term_negated = True
                                    break

                        # --- 3. 'No' determiners attached to the noun modified by the adjective ---
                        if token.dep_ in ["amod", "advmod"] and token.head.pos_ in ["NOUN", "PRON", "ADJ"]:
                            if any(child.dep_ == "det" and child.text.lower() == "no" for child in token.head.children):
                                term_negated = True
                            elif token.head.dep_ == "attr" and token.head.text.lower() == "nothing":
                                term_negated = True

                        # --- 4. Optional: check for negations in conjunctions ---
                        for conj in token.conjuncts:
                            # Only accept negation from conjuncts in the same clause
                            if conj.dep_ not in CLAUSE_BOUNDARIES and \
                                    any(child.dep_ == "neg" for child in conj.children):
                                term_negated = True

                        break  # stop after first match for this term

                # Only add the term if it actually appears in the sentence
                if term_dep is not None:
                    dependencies[term] = term_dep
                    negations[term] = term_negated

            # Keep sentence only if at least one target adjective was found
            if dependencies:
                filtered_texts.append({
                    "sentence": sentence.text,
                    "target_dep_": dependencies,
                    "target_neg": negations,
                    "id": row["id"],
                })

    return filtered_texts


def extract_targets_from_sentence(sentence, target_terms):

    dependencies = {}
    negations = {}

    for term in target_terms:
        term_lower = term.lower()

        dependencies[term] = []
        negations[term] = []

        for i, token in enumerate(sentence):
            if token.text.lower() != term_lower or token.pos_ not in {"ADJ", "ADV"}:
                continue

            if token.text.lower() == "kind":
                if i + 1 < len(sentence) and sentence[i + 1].text.lower() == "of":
                    continue

            if token.text.lower() == "honest":
                if i >= 2:

                    # check "to be (ADV*) honest"
                    j = i - 1

                    # skip adverbs between be → honest
                    while j >= 1 and sentence[j].pos_ == "ADV":
                        j -= 1

                    if (
                            sentence[j].text.lower() == "be"
                            and sentence[j - 1].text.lower() == "to"
                    ):
                        continue

            if (token.dep_ == "amod" and token.head.pos_ in ["NOUN", "PRON", "ADJ", "ADP"]) or \
               token.dep_ in ["acomp", "ccomp", "xcomp", "advcl", "oprd", "advmod", "pcomp", "amod"]:

                term_dep = token.dep_
                term_negated = False

                # --- negation checks ---
                if any(child.dep_ == "neg" for child in token.children):
                    term_negated = True

                # --- 2. Negation on verbs affecting the adjective ---
                # Define clause boundaries BEFORE the ancestor loop
                CLAUSE_BOUNDARIES = {"ccomp", "advcl", "relcl", "acl", "xcomp"}

                for ancestor in token.ancestors:
                    # Stop if we cross ANY clause boundary, not just ccomp
                    if token.dep_ in CLAUSE_BOUNDARIES:
                        break
                    # Also stop if any ancestor on the path is itself a clause root
                    if ancestor.dep_ in CLAUSE_BOUNDARIES:
                        break
                    if ancestor.pos_ in {"AUX", "VERB"}:
                        if any(child.dep_ == "neg" for child in ancestor.children):
                            term_negated = True
                            break

                if token.dep_ in ["amod", "advmod"] and token.head.pos_ in ["NOUN", "PRON", "ADJ"]:
                    if any(child.dep_ == "det" and child.text.lower() == "no"
                           for child in token.head.children):
                        term_negated = True
                    elif token.head.dep_ == "attr" and token.head.text.lower() == "nothing":
                        term_negated = True

                # --- 4. Optional: check for negations in conjunctions ---
                for conj in token.conjuncts:
                    # Only accept negation from conjuncts in the same clause
                    if conj.dep_ not in CLAUSE_BOUNDARIES and \
                            any(child.dep_ == "neg" for child in conj.children):
                        term_negated = True

                # ✔ append instead of overwrite
                dependencies[term].append(term_dep)
                negations[term].append(term_negated)

    # keep only terms that actually occurred
    dependencies = {k: v for k, v in dependencies.items() if v}
    negations = {k: v for k, v in negations.items() if v}

    if not dependencies:
        return {}, {}

    return dependencies, negations

def build_corpus(items, nlp, target_terms):
    corpus = []

    for item in items:
        text = item["title"] + ". " + item["selftext"]
        doc = nlp(text)

        if doc is None:
            continue

        for sentence in doc.sents:
            deps, negs = extract_targets_from_sentence(sentence, target_terms)

            # drop empty sentences (your requirement)
            if not deps:
                continue

            corpus.append({
                "sentence": sentence.text,
                "id": item["id"],
                "subreddit": item.get("subreddit"),
                "target_dep_": deps,
                "target_neg": negs,
            })

    return corpus