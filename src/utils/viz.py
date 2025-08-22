import pandas as pd
import plotly.express as px
from wordcloud import WordCloud

def plot_sentiment_distribution(df: pd.DataFrame, col="sentiment"):
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "count"]
    fig = px.pie(counts, names=col, values="count", title="Sentiment Distribution")
    return fig

def plot_top_tokens(token_counts: pd.DataFrame, top_n=20):
    top = token_counts.head(top_n)
    fig = px.bar(top, x="token", y="count", title=f"Top {top_n} tokens")
    return fig

def make_wordcloud(texts, width=800, height=400):
    wc = WordCloud(width=width, height=height, background_color="white").generate(" ".join(texts))
    return wc.to_image()
