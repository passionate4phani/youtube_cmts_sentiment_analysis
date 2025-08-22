import os
import joblib
import pandas as pd
from typing import List
from .deep import DeepSentiment

def load_classical(vectorizer_path: str, model_path: str):
    if not (os.path.exists(vectorizer_path) and os.path.exists(model_path)):
        return None, None
    vec = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)
    return vec, clf

def infer_classical(texts: List[str], vec, clf) -> List[str]:
    X = vec.transform(texts)
    preds = clf.predict(X)
    return preds

def infer_deep(texts: List[str], model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    deep = DeepSentiment(model_name=model_name)
    return deep.predict(texts)
