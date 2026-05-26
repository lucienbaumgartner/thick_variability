from transformers import pipeline
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL_NAME,
    truncation=True,
    device=-1  # CPU; MPS not cleanly supported by pipeline
)

def get_sentiment_score(sentence: str) -> float:
    result = sentiment_pipeline(sentence)[0]
    label = result["label"].lower()  # "positive", "neutral", "negative"
    score = result["score"]

    if label == "positive":
        return 0.5 + score / 2
    elif label == "negative":
        return 0.5 - score / 2
    else:
        return 0.5


print(sentiment_pipeline("Everyone agreed that John was a sincere person."))
print(sentiment_pipeline("Everyone agreed that John was a deceitful person."))
