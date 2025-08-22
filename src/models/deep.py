from typing import List, Dict
from transformers import pipeline

class DeepSentiment:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        # uses default sentiment-analysis pipeline (POSITIVE/NEGATIVE)
        self.pipe = pipeline("sentiment-analysis", model=model_name)

    def predict(self, texts: List[str]) -> List[Dict]:
        results = self.pipe(texts, truncation=True)
        # Map POSITIVE/NEGATIVE into positive/negative and add neutral when near 0.5
        mapped = []
        for r in results:
            label = r["label"].lower()
            score = float(r["score"])
            # create a heuristic neutral band around 0.5
            if 0.45 < score < 0.55:
                sentiment = "neutral"
            else:
                sentiment = "positive" if "pos" in label else "negative"
            mapped.append({"label": sentiment, "score": score, "raw_label": r["label"]})
        return mapped
