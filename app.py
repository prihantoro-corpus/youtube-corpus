import streamlit as st
import pandas as pd
import re
import time
from datetime import datetime
from io import BytesIO
from youtube_comment_downloader import YoutubeCommentDownloader
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)

# ---------------------------
# Helpers
# ---------------------------

def extract_video_id(url):
    patterns = [
        r"v=([^&]+)",
        r"youtu\.be/([^?]+)",
        r"youtube\.com/shorts/([^?]+)"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def transcript_via_yt_dlp(url):
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "quiet": True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        subs = info.get("subtitles") or info.get("automatic_captions")
        if not subs:
            return None

        lang = list(subs.keys())[0]
        sub_url = subs[lang][0]["url"]

        import requests
        text = requests.get(sub_url).text
        clean = re.sub(r"<[^>]+>", " ", text)
        return clean


def get_transcript(video_id, url):
    try:
        tr = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(t["text"] for t in tr)
    except Exception:
        return transcript_via_yt_dlp(url)


def get_comments(video_id, limit):
    downloader = YoutubeCommentDownloader()
    comments = []
    for c in downloader.get_comments_from_youtube(video_id):
        comments.append(c)
        if limit and len(comments) >= limit:
            break
    return comments


# ---------------------------
# UI
# ---------------------------

st.set_page_config("YouTube Corpus Builder", layout="wide")
st.title("📚 YouTube Corpus Builder (No API)")

urls_input = st.text_area("YouTube links (one per line)", height=150)

comment_limit = st.number_input(
    "Number of comments (0 = all)",
    min_value=0,
    value=100,
    step=50
)

if st.button("🚀 Build dataset"):
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    if not urls:
        st.error("Please enter at least one URL.")
        st.stop()

    transcript_book = {}
    comment_book = {}
    report_lines = []

    progress = st.progress(0)
    start = time.time()

    t_success = c_success = 0

    for i, url in enumerate(urls, 1):
        progress.progress(i / len(urls))
        video_id = extract_video_id(url)

        # -------- Transcript --------
        transcript_sentences = []
        transcript_text = get_transcript(video_id, url)

        if transcript_text:
            for s in sent_tokenize(transcript_text):
                transcript_sentences.append({
                    "video_id": video_id,
                    "video_url": url,
                    "sentence": s,
                    "scraped_at": datetime.now().isoformat()
                })
            t_success += 1
            report_lines.append(
                f"VIDEO {video_id}\nTranscript: SUCCESS ({len(transcript_sentences)} sentences)"
            )
        else:
            report_lines.append(
                f"VIDEO {video_id}\nTranscript: FAILED"
            )

        transcript_book[video_id[:31]] = pd.DataFrame(transcript_sentences)

        # -------- Comments --------
        comment_rows = []
        try:
            comments = get_comments(video_id, None if comment_limit == 0 else comment_limit)
            for c in comments:
                for s in sent_tokenize(c["text"]):
                    comment_rows.append({
                        "video_id": video_id,
                        "video_url": url,
                        "author": c.get("author"),
                        "like_count": c.get("votes"),
                        "published_at": c.get("time"),
                        "sentence": s,
                        "scraped_at": datetime.now().isoformat()
                    })
            c_success += 1
            report_lines.append(
                f"Comments: SUCCESS ({len(comment_rows)} sentences)\n"
            )
        except Exception:
            report_lines.append("Comments: FAILED\n")

        comment_book[video_id[:31]] = pd.DataFrame(comment_rows)

    # Deduplication
    transcript_book = {
        k: v.drop_duplicates(subset=["sentence"]) for k, v in transcript_book.items()
    }
    comment_book = {
        k: v.drop_duplicates(subset=["sentence"]) for k, v in comment_book.items()
    }

    # -------- Save files --------
    tx_buf = BytesIO()
    cm_buf = BytesIO()

    with pd.ExcelWriter(tx_buf, engine="openpyxl") as w:
        for sheet, df in transcript_book.items():
            df.to_excel(w, sheet_name=sheet, index=False)

    with pd.ExcelWriter(cm_buf, engine="openpyxl") as w:
        for sheet, df in comment_book.items():
            df.to_excel(w, sheet_name=sheet, index=False)

    runtime = round(time.time() - start, 2)

    report_lines.append(f"TOTAL VIDEOS: {len(urls)}")
    report_lines.append(f"TRANSCRIPTS SUCCESS: {t_success}")
    report_lines.append(f"COMMENTS SUCCESS: {c_success}")
    report_lines.append(f"RUNTIME: {runtime} seconds")

    report_text = "\n".join(report_lines)

    st.success("Dataset ready.")

    st.download_button("⬇ transcript.xlsx", tx_buf.getvalue(), "transcript.xlsx")
    st.download_button("⬇ comments.xlsx", cm_buf.getvalue(), "comments.xlsx")
    st.download_button("⬇ report.txt", report_text, "report.txt")
