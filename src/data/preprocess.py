import re
import emoji
import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
MULTISPACE_RE = re.compile(r"\s+")

def clean_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = HASHTAG_RE.sub(" ", s)
    s = emoji.replace_emoji(s, replace="")
    s = re.sub(r"[^a-z0-9'!?.,$%()+\-:; ]", " ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

def preprocess_df(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    out = df.copy()
    out["clean_text"] = out[text_col].astype(str).apply(clean_text)
    return out
