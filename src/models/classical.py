import argparse
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import yaml

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_pipeline(ngram_range=(1,2), max_features=30000):
    vec = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
    clf = LogisticRegression(max_iter=200, C=2.0, n_jobs=None)
    return Pipeline([("tfidf", vec), ("clf", clf)])

def train(train_csv: str, config_path="config.yaml"):
    cfg = load_config(config_path)
    df = pd.read_csv(train_csv)
    df = df.dropna(subset=["text","label"])
    X = df["text"].astype(str).tolist()
    y = df["label"].astype(str).tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_pipeline(tuple(cfg["models"]["classical"]["ngram_range"]), cfg["models"]["classical"]["max_features"])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    print(classification_report(y_te, y_pred))

    # save artifacts
    os.makedirs("artifacts", exist_ok=True)
    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    joblib.dump(vec, cfg["models"]["classical"]["artifacts"]["vectorizer"])
    joblib.dump(clf, cfg["models"]["classical"]["artifacts"]["model"])
    print("Saved artifacts to", cfg["models"]["classical"]["artifacts"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, help="CSV with columns text,label")
    args = ap.parse_args()
    if args.train:
        train(args.train)
    else:
        print("Nothing to do. Use --train data/sample_train.csv")
