import os
import yaml
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from src.data.fetch_comments import fetch_comments
from src.data.preprocess import preprocess_df
from src.models.inference import load_classical, infer_classical, infer_deep
from src.utils.io_utils import save_df_csv
from src.utils.viz import plot_sentiment_distribution, plot_top_tokens, make_wordcloud
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

st.set_page_config(page_title="YouTube Sentiment", layout="wide")

@st.cache_resource
def load_cfg():
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

cfg = load_cfg()
load_dotenv()

st.title(" YouTube Comments Sentiment Analysis")
st.write("Enter a YouTube Video ID, fetch comments, and compare **classical ML** vs **deep learning** sentiment predictions.")

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input(" Enter your YouTube API Key", type="password")
    video_id = st.text_input("YouTube Video ID", value="")
    max_comments = st.slider("Max comments", 50, 1000, cfg["app"]["max_comments"], step=50)
    st.caption("Tip: Video ID is the part after `v=` in a YouTube URL.")
    fetch_btn = st.button("Fetch & Analyze", type="primary")

# Lazy load classical artifacts
vec_path = cfg["models"]["classical"]["artifacts"]["vectorizer"]
clf_path = cfg["models"]["classical"]["artifacts"]["model"]
vec, clf = load_classical(vec_path, clf_path)

col1, col2 = st.columns([2,1])
with col2:
    if st.button(" Re-train Classical on sample data"):
        import subprocess, sys
        st.info("Training... check terminal logs.")
        subprocess.run([sys.executable, "-m", "src.models.classical", "--train", "data/sample_train.csv"])
        st.success("Training completed. Reload the page to use new artifacts.")

if fetch_btn and video_id:
    with st.spinner("Calling YouTube API and processing..."):
        try:
            raw_df = fetch_comments(video_id, max_comments=max_comments, api_key=api_key_input or None)
        except Exception as e:
            st.error(f"Failed to fetch comments: {e}")
            st.stop()

        if raw_df.empty:
            st.warning("No comments fetched for this video.")
            st.stop()

        df = preprocess_df(raw_df, text_col="text")

        # Classical predictions (if artifacts exist)
        if vec and clf:
            classical_preds = infer_classical(df["clean_text"].tolist(), vec, clf)
            df["sentiment_classical"] = classical_preds
        else:
            st.warning("Classical artifacts not found. Click 'Re-train Classical' (uses data/sample_train.csv).")

        # Deep predictions
        deep_results = infer_deep(df["clean_text"].tolist(), model_name=cfg["models"]["deep"]["model_name"])
        df["sentiment_deep"] = [r["label"] for r in deep_results]
        df["deep_score"] = [r["score"] for r in deep_results]

        # Save snapshot
        os.makedirs(cfg["app"]["cache_dir"], exist_ok=True)
        csv_out = os.path.join(cfg["app"]["cache_dir"], "latest_comments_with_preds.csv")
        save_df_csv(df, csv_out)

    st.success(f"Fetched {len(df)} comments. Saved to {csv_out}.")

    tab1, tab2, tab3, tab4 = st.tabs([" Overview"," Explorer"," Words"," Model Details"])

    with tab1:
        st.subheader("Sentiment Overview (Deep Model)")
        fig = plot_sentiment_distribution(df, col="sentiment_deep")
        st.plotly_chart(fig, use_container_width=True)

        if "sentiment_classical" in df.columns:
            st.subheader("Comparison: Classical vs Deep (per comment head)")
            st.dataframe(df[["text","sentiment_classical","sentiment_deep","deep_score"]].head(20))

        st.metric("Total comments analyzed", len(df))

    with tab2:
        st.subheader("Comments Explorer")
        st.dataframe(df[["author","publishedAt","text","sentiment_deep"]])

    with tab3:
        st.subheader("Top Tokens & Wordclouds")
        tokens = []
        stops = set(ENGLISH_STOP_WORDS)
        for t in df["clean_text"].tolist():
            tokens.extend([w for w in t.split() if w not in stops and len(w) > 2])
        counts = Counter(tokens)
        tok_df = pd.DataFrame({"token": list(counts.keys()), "count": list(counts.values())}).sort_values("count", ascending=False)
        fig2 = plot_top_tokens(tok_df, top_n=25)
        st.plotly_chart(fig2, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.caption("Wordcloud (all comments)")
            wc_img = make_wordcloud(df["clean_text"].tolist())
            st.image(wc_img)
        with right:
            pos_texts = df[df["sentiment_deep"]=="positive"]["clean_text"].tolist()
            if pos_texts:
                st.caption("Wordcloud (positive)")
                st.image(make_wordcloud(pos_texts))

    with tab4:
        st.subheader("About the Models")
        st.markdown("""
        **Classical**: TF–IDF + Logistic Regression, trainable on CSV (`text,label`).  
        **Deep**: DistilBERT (`transformers` pipeline), mapped to {positive, negative, neutral} using a confidence band.
        """)
        if "sentiment_classical" in df.columns:
            agree = (df["sentiment_classical"] == df["sentiment_deep"]).mean()
            st.metric("Agreement (Classical vs Deep)", f"{agree*100:.1f}%")
        st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")

else:
    col1.write("Enter a YouTube Video ID and click **Fetch & Analyze** to begin.")
