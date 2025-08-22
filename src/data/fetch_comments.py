import os
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

def get_youtube_client(api_key: Optional[str] = None):
    load_dotenv()
    api_key = api_key or os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        raise ValueError("YouTube API key not found. Enter your **YouTube API Key**")
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)

def fetch_comments(video_id: str, max_comments: int = 500, api_key: Optional[str]=None) -> pd.DataFrame:
    """
    Fetch top-level comments for a given video_id using YouTube Data API v3.
    Returns a DataFrame with columns: author, text, likeCount, publishedAt, updatedAt.
    """
    youtube = get_youtube_client(api_key)
    comments: List[Dict] = []
    next_page_token = None
    fetched = 0

    try:
        pbar = tqdm(total=max_comments, desc="Fetching comments", unit="c")
        while True:
            req = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=min(100, max_comments - fetched),
                pageToken=next_page_token,
                textFormat="plainText",
                order="time"
            )
            res = req.execute()
            for item in res.get("items", []):
                sn = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "author": sn.get("authorDisplayName"),
                    "text": sn.get("textDisplay", ""),
                    "likeCount": sn.get("likeCount", 0),
                    "publishedAt": sn.get("publishedAt"),
                    "updatedAt": sn.get("updatedAt"),
                })
                fetched += 1
                pbar.update(1)
                if fetched >= max_comments:
                    break
            pbar.close()
            if fetched >= max_comments:
                break
            next_page_token = res.get("nextPageToken")
            if not next_page_token:
                break
    except HttpError as e:
        raise RuntimeError(f"YouTube API error: {e}")

    return pd.DataFrame(comments)
