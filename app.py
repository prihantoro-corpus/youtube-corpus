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
import random
import zipfile
import requests
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ---------------------------
# Initializers
# ---------------------------

analyzer = SentimentIntensityAnalyzer()
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        # Using 'base' model for speed/accuracy balance
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "Positive", compound
    elif compound <= -0.05:
        return "Negative", compound
    else:
        return "Neutral", compound

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


def asr_fallback(url):
    video_id = extract_video_id(url)
    print(f"DEBUG: {video_id} - Starting ASR Fallback as last resort...")
    st.info(f"🎙️ Using ASR Fallback for `{video_id}` (No captions found)")
    
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_file = os.path.join(tmpdir, f"{video_id}.mp3")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": audio_file,
            "quiet": True,
            "no_warnings": True,
            "extractor_args": {"youtube": {"player_client": ["android"]}},
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            if not os.path.exists(audio_file):
                print(f"DEBUG: {video_id} - Audio file download failed")
                return None
            
            model = get_whisper_model()
            result = model.transcribe(audio_file)
            return result.get("text", "").strip()
        except Exception as e:
            print(f"DEBUG: {video_id} - ASR error: {e}")
            return None

def transcript_via_yt_dlp(url):
    video_id = extract_video_id(url)
    clients = [["android"], ["web"], ["mweb"]]
    
    for client in clients:
        print(f"DEBUG: {video_id} - Attempting transcript with client: {client}")
        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "ignoreerrors": True,
            "extractor_args": {"youtube": {"player_client": client}},
        }
        
        # User agents corresponding to clients
        ua_map = {
            "android": "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Mobile Safari/537.36",
            "web": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            "mweb": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Mobile/15E148 Safari/604.1"
        }
        headers = {
            "User-Agent": ua_map.get(client[0], ua_map["web"]),
            "Accept-Language": "en-US,en;q=0.9",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                subs = info.get("subtitles") or {}
                auto_subs = info.get("automatic_captions") or {}
                
                combined_subs = {**auto_subs, **subs}
                if not combined_subs:
                    continue

                lang = "en" if "en" in combined_subs else list(combined_subs.keys())[0]
                formats = combined_subs[lang]
                
                sub_url = None
                for f in formats:
                    if f.get("ext") == "json3":
                        sub_url = f["url"]
                        break
                if not sub_url:
                    sub_url = formats[0]["url"]
                
                resp = requests.get(sub_url, headers=headers)
                if resp.status_code != 200 or "automated queries" in resp.text:
                    continue
                
                if '"events"' in resp.text and '"segs"' in resp.text:
                    try:
                        import json
                        data = json.loads(resp.text)
                        text_parts = []
                        for event in data.get("events", []):
                            for seg in event.get("segs", []):
                                if seg.get("utf8"):
                                    text_parts.append(seg["utf8"])
                        return " ".join(text_parts).strip()
                    except Exception:
                        pass

                clean = re.sub(r"<[^>]+>", " ", resp.text)
                result = clean.strip()
                if result:
                    return result
            except Exception as e:
                print(f"DEBUG: {video_id} - Client {client} failed: {e}")
                continue
                
    # If all yt-dlp attempts fail, try ASR as a last resort
    return asr_fallback(url)


def get_transcript(video_id, url):
    try:
        # Try direct API first
        tr = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(t["text"] for t in tr)
        if "automated queries" in text or "Google Home" in text:
            return transcript_via_yt_dlp(url)
        return text
    except Exception:
        return transcript_via_yt_dlp(url)


def get_comments(video_id, limit):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "skip_download": True,
        "getcomments": True,
        "max_comments": limit if limit > 0 else None,
    }
    
    comments_list = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            raw_comments = info.get("comments", [])
            for c in raw_comments:
                comments_list.append({
                    "author": c.get("author"),
                    "text": c.get("text"),
                    "votes": c.get("like_count"),
                    "time": c.get("timestamp"),
                })
    except Exception as e:
        print(f"DEBUG: yt-dlp comment error: {e}")
        # Fallback to secondary downloader if yt-dlp fails
        from youtube_comment_downloader import YoutubeCommentDownloader
        downloader = YoutubeCommentDownloader()
        for idx, c in enumerate(downloader.get_comments_from_youtube(video_id)):
            comments_list.append(c)
            if limit and len(comments_list) >= limit:
                break
                
    return comments_list


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
        st.info(f"🔍 Processing video {i}/{len(urls)}: `{video_id}`")
        
        # Avoid rapid blocks
        if i > 1:
            time.sleep(random.uniform(2, 5))

        # -------- Transcript --------
        transcript_sentences = []
        try:
            transcript_text = get_transcript(video_id, url)
        except Exception as e:
            st.warning(f"⚠️ Transcript error for `{video_id}`: {e}")
            transcript_text = None

        if transcript_text:
            st.success(f"✅ Transcript found for `{video_id}`")
            print(f"DEBUG: Successfully got transcript for {video_id} ({len(transcript_text)} chars)")
            for s in sent_tokenize(transcript_text):
                sent, score = get_sentiment(s)
                transcript_sentences.append({
                    "video_id": video_id,
                    "video_url": url,
                    "sentence": s,
                    "sentiment": sent,
                    "sentiment_score": score,
                    "scraped_at": datetime.now().isoformat()
                })
            t_success += 1
            report_lines.append(
                f"VIDEO {video_id}\nTranscript: SUCCESS ({len(transcript_sentences)} sentences)"
            )
        else:
            print(f"DEBUG: Failed to get transcript for {video_id}")
            report_lines.append(
                f"VIDEO {video_id}\nTranscript: FAILED"
            )

        transcript_book[video_id[:31]] = pd.DataFrame(transcript_sentences)

        # -------- Comments --------
        comment_rows = []
        try:
            comments = get_comments(video_id, None if comment_limit == 0 else comment_limit)
            if comments:
                st.success(f"✅ Found {len(comments)} comments for `{video_id}`")
            else:
                st.info(f"ℹ️ No comments returned for `{video_id}`")
            for c in comments:
                for s in sent_tokenize(c["text"]):
                    sent, score = get_sentiment(s)
                    comment_rows.append({
                        "video_id": video_id,
                        "video_url": url,
                        "author": c.get("author"),
                        "like_count": c.get("votes"),
                        "published_at": c.get("time"),
                        "sentence": s,
                        "sentiment": sent,
                        "sentiment_score": score,
                        "scraped_at": datetime.now().isoformat()
                    })
            c_success += 1
            report_lines.append(
                f"Comments: SUCCESS ({len(comment_rows)} sentences)\n"
            )
        except Exception as e:
            print(f"DEBUG: Failed to get comments for {video_id}: {e}")
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

    # Combined Previews
    st.subheader("📊 Data Previews")
    
    # Safely concat dataframes
    valid_transcripts = [df for df in transcript_book.values() if not df.empty]
    valid_comments = [df for df in comment_book.values() if not df.empty]
    
    all_transcripts = pd.concat(valid_transcripts) if valid_transcripts else pd.DataFrame(columns=["video_id", "video_url", "sentence", "sentiment", "sentiment_score", "scraped_at"])
    all_comments = pd.concat(valid_comments) if valid_comments else pd.DataFrame(columns=["video_id", "video_url", "author", "sentence", "sentiment", "sentiment_score", "scraped_at"])

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Transcripts")
        st.dataframe(all_transcripts.head(100), use_container_width=True)
    with col2:
        st.write("### Comments")
        st.dataframe(all_comments.head(100), use_container_width=True)

    # Downloads
    st.divider()
    st.subheader("⬇ Download Section")

    # Zip Generation
    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("transcript.xlsx", tx_buf.getvalue())
        zf.writestr("comments.xlsx", cm_buf.getvalue())
        zf.writestr("report.txt", report_text)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("📦 Save All (ZIP)", zip_buf.getvalue(), "corpus_package.zip", type="primary")
    c2.download_button("📄 Transcripts (.xlsx)", tx_buf.getvalue(), "transcript.xlsx")
    c3.download_button("💬 Comments (.xlsx)", cm_buf.getvalue(), "comments.xlsx")
    c4.download_button("📝 Report (.txt)", report_text, "report.txt")
