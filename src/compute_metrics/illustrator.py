from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaForSequenceClassification
from transformers import pipeline
import torch
from helper.mlm_cf import causal_sentiment_effect_roberta_verbose

# --------------------------------------------------
# Config
# --------------------------------------------------
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MLM_MODEL_NAME = "roberta-base"

# --------------------------------------------------
# Device
# --------------------------------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"

# --------------------------------------------------
# MLM model
# --------------------------------------------------
mlm_tokenizer = RobertaTokenizerFast.from_pretrained(MLM_MODEL_NAME)
mlm_model = RobertaForMaskedLM.from_pretrained(MLM_MODEL_NAME)
mlm_model.eval()
mlm_model.to(device)

# --------------------------------------------------
# Sentiment model
# --------------------------------------------------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL_NAME,
    truncation=True,
    device=-1
)

def get_sentiment_score(sentence: str) -> float:
    result = sentiment_pipeline(sentence)[0]
    label = result["label"].lower()
    score = result["score"]
    if label == "positive":
        return 0.5 + score / 2
    elif label == "negative":
        return 0.5 - score / 2
    else:
        return 0.5

# --------------------------------------------------
# Example sentences
# --------------------------------------------------
examples = [
    ("Despite all his flaws and wrongdoings, everyone still considered him an honest man.", "honest"),
    ("John is sort of an honest guy.", "honest"),
    ("John is an honest person.", "honest"),
    ("John is a serious worker who shows a lot of commitment --- overall just an honest guy.", "honest"),
]

ig_tokenizer = RobertaTokenizerFast.from_pretrained(SENTIMENT_MODEL_NAME)
ig_model = RobertaForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
ig_model.eval().to(device)

for sentence, target_word in examples:
    causal_sentiment_effect_roberta_verbose(
        sentence=sentence,
        target_word=target_word,
        mlm_model=mlm_model,
        mlm_tokenizer=mlm_tokenizer,
        sentiment_fn=get_sentiment_score,
        ig_model=ig_model,
        ig_tokenizer=ig_tokenizer,
        top_p=0.9,
        top_n=50,
        n_illustrative=3,
        device=device
    )