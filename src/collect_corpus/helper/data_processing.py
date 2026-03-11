import json
import zstandard as zstd
from spacy.language import Language
from langdetect import detect
import language_tool_python
import re
import io

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

def read_zst_file(file_path):
    print(f"Opening file: {file_path}")
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8', errors='replace')
            for idx, line in enumerate(text_stream):
                line = line.strip()
                if not line:
                    continue
                try:
                    # current bytes read is the position in the underlying raw stream
                    bytes_read = reader.tell()
                    yield json.loads(line), bytes_read
                except json.JSONDecodeError as e:
                    print(f"JSON decode error at line {idx}: {e}")
                    continue

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
