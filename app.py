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
import html
from nltk.tokenize import sent_tokenize, word_tokenize
import indonesian_utils

@st.cache_resource
def download_nltk():
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

download_nltk()

# ---------------------------
# Initializers (Cached)
# ---------------------------

@st.cache_resource
def get_analyzer():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()

@st.cache_resource
def get_whisper_model():
    import whisper
    # Using 'base' model for speed/accuracy balance
    return whisper.load_model("base")

@st.cache_resource
def get_aksara_models():
    from aksara import POSTagger, Lemmatizer
    return POSTagger(), Lemmatizer()

@st.cache_resource
def get_stanza_pipeline(lang):
    import stanza
    try:
        # Map display codes to Stanza codes
        s_map = {"zh-Hans": "zh-hans"}
        s_code = s_map.get(lang, lang)
        print(f"DEBUG: Checking Stanza model for: {s_code}")
        # Use logging_level instead of quiet
        stanza.download(s_code, processors='tokenize,pos,lemma', logging_level='WARN')
        return stanza.Pipeline(s_code, processors='tokenize,pos,lemma', use_gpu=False, logging_level='WARN')
    except Exception as e:
        st.error(f"❌ Stanza Initialization Error: {e}")
        raise e

def xml_escape(text):
    return html.escape(str(text))

def get_sentiment(text, strategy="Translate to English", source_lang=None):
    from deep_translator import GoogleTranslator
    from textblob import TextBlob
    
    analyzer = get_analyzer()
    if strategy == "Translate to English" and source_lang and source_lang != 'en':
        try:
            # Short-circuit if no English needed
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            text = translated if translated else text
        except Exception as e:
            print(f"DEBUG: Translation error: {e}")
            pass

    # Use VADER (better for English or translated text)
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    # TextBlob as fallback/secondary check if strategy is Language-specific
    if strategy == "Language-specific" and source_lang != 'en':
        try:
            blob = TextBlob(text)
            blob_score = blob.sentiment.polarity
            # Average or normalize? Let's use simple threshold for now
            if blob_score >= 0.1: return "Positive", blob_score
            if blob_score <= -0.1: return "Negative", blob_score
            return "Neutral", blob_score
        except Exception:
            pass

    if compound >= 0.05:
        return "Positive", compound
    elif compound <= -0.05:
        return "Negative", compound
    else:
        return "Neutral", compound

def tag_sentence(text, lang_code, method="Automatic", uniform_tag="TAG", **kwargs):
    rows = []
    
    # Pre-tokenization for Indonesian to handle clitics
    if lang_code == "id" and kwargs.get("enable_clitic", False):
        try:
            # Use the clitic tokenizer
            clitic_tokens = indonesian_utils.tokenize_text(text)
            # Reconstruct text with spaces between split tokens for the taggers to consume
            # This ensures "badanmu" -> "badan mu" so Stanza/Aksara sees them as separate words
            text = " ".join(clitic_tokens)
        except Exception as e:
            print(f"DEBUG: Indonesian clitic tokenization failed: {e}")
            # Fallback to original text if fails

    if method == "Uniform":
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        for t in tokens:
            rows.append({
                "token": t,
                "tag": uniform_tag,
                "lemma": t
            })
    else:
        # Automatic Mode
        if lang_code == "id" and kwargs.get("id_tagger") == "Aksara":
            try:
                pos_tagger, lemmatizer = get_aksara_models()
                # Tagging returns list of [token, tag]
                tags = pos_tagger.tag(text)
                for t, p in tags:
                    rows.append({
                        "token": t,
                        "tag": p,
                        "lemma": lemmatizer.lemmatize(t)
                    })
                return rows
            except Exception as e:
                st.warning(f"⚠️ Aksara Tagging Failed: {e}. Falling back to Stanza.")
                # Continue to Stanza fallback

        if lang_code == "en":
            try:
                import nltk
                nltk.download("averaged_perceptron_tagger", quiet=True)
                tokens = word_tokenize(text)
                pos_tags = nltk.pos_tag(tokens)
                for t, p in pos_tags:
                    rows.append({
                        "token": t,
                        "tag": p,
                        "lemma": t # NLTK doesn't have a simple lemmatizer without WordNet, keep as token
                    })
                return rows
            except Exception as e:
                print(f"DEBUG: NLTK English tagging error: {e}")
                # Fallback to Stanza or Uniform if NLTK fails
        
        try:
            with st.spinner(f"⌛ Loading tagging model for {lang_code}..."):
                nlp = get_stanza_pipeline(lang_code)
            doc = nlp(text)
            for sent in doc.sentences:
                for word in sent.words:
                    rows.append({
                        "token": word.text,
                        "tag": word.upos or word.xpos or "UNK",
                        "lemma": word.lemma or word.text
                    })
        except Exception as e:
            msg = f"⚠️ Tagging Failed for {lang_code}: {e}. Switching to Uniform tagging for this text."
            st.warning(msg)
            print(f"DEBUG: {msg}")
            # Explicitly switch to Uniform behavior for this call
            return tag_sentence(text, lang_code, method="Uniform", uniform_tag=uniform_tag)
    return rows

def build_xml_block(id_attr, data):
    """
    id_attr: string (video_id or index)
    data: list of dicts 
          If transcript: {num, sentiment, score, tokens: [{token, tag, lemma}]}
          If comments: {author, sentences: [{num, sentiment, score, tokens: [...]}]}
    """
    lines = [f'<text id="{xml_escape(id_attr)}">']
    for entry in data:
        if "author" in entry:
            # Comment grouping
            lines.append(f'<comment author="{xml_escape(entry["author"])}">')
            for s in entry["sentences"]:
                lines.append(f'<s number="{s["num"]}" sentiment="{xml_escape(s["sentiment"])}" sentiment_score="{s["score"]}">')
                for t in s["tokens"]:
                    lines.append(f'{xml_escape(t["token"])}\t{xml_escape(t["tag"])}\t{xml_escape(t["lemma"])}')
                lines.append('</s>')
            lines.append('</comment>')
        else:
            # Single sentence (Transcript)
            s = entry
            lines.append(f'<s number="{s["num"]}" sentiment="{xml_escape(s["sentiment"])}" sentiment_score="{s["score"]}">')
            for t in s["tokens"]:
                lines.append(f'{xml_escape(t["token"])}\t{xml_escape(t["tag"])}\t{xml_escape(t["lemma"])}')
            lines.append('</s>')
    lines.append('</text>')
    return "\n".join(lines)

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

def transcript_via_yt_dlp(url, target_lang="en"):
    video_id = extract_video_id(url)
    clients = [["android"], ["web"], ["mweb"]]
    
    for client in clients:
        print(f"DEBUG: {video_id} - Attempting transcript with client: {client} for lang: {target_lang}")
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
            "Accept-Language": f"{target_lang},en;q=0.9",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                subs = info.get("subtitles") or {}
                auto_subs = info.get("automatic_captions") or {}
                
                combined_subs = {**auto_subs, **subs}
                if not combined_subs:
                    continue

                # Targeted language match
                lang = target_lang if target_lang in combined_subs else \
                       ("en" if "en" in combined_subs else list(combined_subs.keys())[0])
                
                print(f"DEBUG: {video_id} - Available languages: {list(combined_subs.keys())}. Chosen: {lang}")
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


def get_transcript(video_id, url, target_lang="en"):
    try:
        # Try direct API first with target language
        tr = YouTubeTranscriptApi.get_transcript(video_id, languages=[target_lang, 'en'])
        text = " ".join(t["text"] for t in tr)
        if "automated queries" in text or "Google Home" in text:
            return transcript_via_yt_dlp(url, target_lang)
        return text
    except Exception:
        return transcript_via_yt_dlp(url, target_lang)


def get_comments(video_id, limit):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "skip_download": True,
        "getcomments": True,
        "max_comments": limit if (limit is not None and limit > 0) else None,
    }
    
    comments_list = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            raw_comments = info.get("comments", [])
            for c in raw_comments:
                if limit and len(comments_list) >= limit:
                    break
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

with st.sidebar:
    st.markdown("[📖 Manual](https://docs.google.com/document/d/1aTAcBRw94P3WBzOsHJbb0hIeTZNTW3x0/edit?usp=sharing&ouid=101822844091249125937&rtpof=true&sd=true)")
    st.divider()
    st.header("⚙️ Configuration")
    target_lang_name = st.selectbox(
        "Target Language",
        ["English", "Indonesian", "Chinese", "Japanese", "Korean"],
        index=0
    )
    lang_map = {
        "English": "en",
        "Indonesian": "id",
        "Chinese": "zh-Hans",
        "Japanese": "ja",
        "Korean": "ko"
    }
    target_lang = lang_map[target_lang_name]

    st.divider()
    st.header("🧠 Sentiment Strategy")
    st_strategy = st.radio(
        "Analysis Method",
        ["Translate to English", "Language-specific"],
        help="Strategy A: Uses native analyzers. Strategy B: Translates to English before scoring."
    )

    st.divider()
    st.header("🏷️ Tagging & Formatting")
    enable_tagging = st.checkbox("Enable Tokenization & Tagging", value=True)
    tag_method = st.radio("Tagging Method", ["Automatic", "Uniform"], index=0)
    
    id_tagger = "Stanza"
    enable_clitic = False
    if target_lang == "id":
        if tag_method == "Automatic":
            id_tagger = st.radio("Indonesian Tagger Model", ["Stanza", "Aksara"], index=1, help="Aksara is often more accurate for Indonesian.")
        enable_clitic = st.checkbox("Enable Clitic Tokenization", value=True, help="Splits clitics (e.g. 'ku-', '-mu') before tagging. Disabling this may speed up processing.")
        
    uniform_label = st.text_input("Uniform Tag Label", value="TAG")

if "transcript_book" not in st.session_state: st.session_state.transcript_book = {}
if "comment_book" not in st.session_state: st.session_state.comment_book = {}
if "transcript_xml_data" not in st.session_state: st.session_state.transcript_xml_data = {}
if "comment_xml_data" not in st.session_state: st.session_state.comment_xml_data = {}
if "report_lines" not in st.session_state: st.session_state.report_lines = []
if "batch_stats" not in st.session_state: st.session_state.batch_stats = {"t_success": 0, "c_success": 0, "runtime": 0, "total": 0}

urls_input = st.text_area("YouTube links (one per line)", height=150)

col_ui1, col_ui2 = st.columns([1, 2])
with col_ui1:
    download_all = st.checkbox("Download all comments", value=False)
with col_ui2:
    custom_limit = st.number_input("Comment limit", min_value=1, value=100, step=50, disabled=download_all)

comment_limit = 0 if download_all else custom_limit

btn_col_a, btn_col_b = st.columns([1, 1])
with btn_col_a:
    build_clicked = st.button("🚀 Build dataset", use_container_width=True)
with btn_col_b:
    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.transcript_book = {}
        st.session_state.comment_book = {}
        st.session_state.transcript_xml_data = {}
        st.session_state.comment_xml_data = {}
        st.session_state.report_lines = []
        st.session_state.batch_stats = {"t_success": 0, "c_success": 0, "runtime": 0, "total": 0}
        st.rerun()

if build_clicked:
    urls = [u.strip() for u in urls_input.splitlines() if u.strip()]
    if not urls:
        st.error("Please enter at least one URL.")
        st.stop()

    st.session_state.batch_stats["total"] = len(urls)
    progress = st.progress(0)
    start_batch = time.time()
    t_success = st.session_state.batch_stats["t_success"]
    c_success = st.session_state.batch_stats["c_success"]

    for i, url in enumerate(urls, 1):
        progress.progress(i / len(urls))
        video_id = extract_video_id(url)
        
        # Check if already processed in this session to allow "resuming" if one clicks Build again
        if video_id and any(video_id in k for k in st.session_state.transcript_book.keys()):
            st.info(f"⏭️ Skipping `{video_id}` (Already processed)")
            continue

        st.info(f"🔍 Processing video {i}/{len(urls)}: `{video_id}`")
        status_col, dl_col = st.columns([4, 1])
        status_box = status_col.empty()
        dl_box = dl_col.empty()
        
        if i > 1: time.sleep(random.uniform(2, 5))

        def render_checklist(stage, detail=""):
            s_pend, s_run, s_done = "⚪", "🔄", "✅"
            st_t_find = s_run if stage == "trans_find" else (s_done if stage in ["trans_tok", "comm_find", "comm_tok", "done"] else s_pend)
            st_t_tok  = s_run if stage == "trans_tok"  else (s_done if stage in ["comm_find", "comm_tok", "done"] else s_pend)
            st_c_find = s_run if stage == "comm_find"  else (s_done if stage in ["comm_tok", "done"] else s_pend)
            st_c_tok  = s_run if stage == "comm_tok"   else (s_done if stage == "done" else s_pend)
            current_action = f"**Current Action:** {detail}" if detail else ""
            md = f"### `{video_id}`\n\n{st_t_find} **1. Find Transcript**\n{st_t_tok} **2. Tokenize Transcript**\n{st_c_find} **3. Fetch Comments**\n{st_c_tok} **4. Tokenize Comments**\n\n{current_action}"
            status_box.markdown(md)

        # Processing logic...
        render_checklist("trans_find", "Searching for transcript...")
        transcript_sentences = []
        xml_sents_t = []
        try:
            transcript_text = get_transcript(video_id, url, target_lang)
        except Exception as e:
            st.warning(f"⚠️ Transcript error for `{video_id}`: {e}")
            transcript_text = None

        if transcript_text:
            st.success(f"✅ Transcript found for `{video_id}`")
            for s_num, s in enumerate(sent_tokenize(transcript_text), 1):
                sent_cat, score = get_sentiment(s, strategy=st_strategy, source_lang=target_lang)
                tokens = []
                if enable_tagging:
                    if s_num % 10 == 0: render_checklist("trans_tok", f"Tokenizing sentence {s_num}...")
                    tokens = tag_sentence(s, target_lang, method=tag_method, uniform_tag=uniform_label, id_tagger=id_tagger, enable_clitic=enable_clitic)
                transcript_sentences.append({
                    "video_id": video_id, "video_url": url, "sentence_num": s_num, "sentence": s,
                    "sentiment": sent_cat, "sentiment_score": score, "scraped_at": datetime.now().isoformat()
                })
                xml_sents_t.append({"num": s_num, "sentiment": sent_cat, "score": score, "tokens": tokens})
            t_success += 1
            st.session_state.report_lines.append(f"VIDEO {video_id}\nTranscript: SUCCESS ({len(transcript_sentences)} sentences)")
            st.session_state.transcript_xml_data[video_id] = xml_sents_t
        else:
            st.session_state.report_lines.append(f"VIDEO {video_id}\nTranscript: FAILED")

        st.session_state.transcript_book[video_id[:31]] = pd.DataFrame(transcript_sentences)

        render_checklist("comm_find", "Fetching comments...")
        comment_rows = []
        xml_sents_c = []
        try:
            comments = get_comments(video_id, None if comment_limit == 0 else comment_limit)
            for c_idx, c in enumerate(comments, 1):
                comment_author = c.get("author") or "Unknown"
                current_comment_xml = {"author": comment_author, "sentences": []}
                for s_num, s in enumerate(sent_tokenize(c["text"]), 1):
                    sent_cat, score = get_sentiment(s, strategy=st_strategy, source_lang=target_lang)
                    tokens = []
                    if enable_tagging:
                        if s_num % 5 == 0: render_checklist("comm_tok", f"Tokenizing comment {c_idx}, sentence {s_num}...")
                        tokens = tag_sentence(s, target_lang, method=tag_method, uniform_tag=uniform_label, id_tagger=id_tagger, enable_clitic=enable_clitic)
                    comment_rows.append({
                        "video_id": video_id, "video_url": url, "author": comment_author, "like_count": c.get("votes"),
                        "published_at": c.get("time"), "sentence_num": s_num, "sentence": s,
                        "sentiment": sent_cat, "sentiment_score": score, "scraped_at": datetime.now().isoformat()
                    })
                    current_comment_xml["sentences"].append({"num": s_num, "sentiment": sent_cat, "score": score, "tokens": tokens})
                if current_comment_xml["sentences"]: xml_sents_c.append(current_comment_xml)
            c_success += 1
            st.session_state.report_lines.append(f"Comments: SUCCESS ({len(comment_rows)} sentences)\n")
            st.session_state.comment_xml_data[video_id] = xml_sents_c
        except Exception:
            st.session_state.report_lines.append("Comments: FAILED\n")

        st.session_state.comment_book[video_id[:31]] = pd.DataFrame(comment_rows)
        render_checklist("done", "Finished.")
        
        try:
            v_zip_buf = BytesIO()
            with zipfile.ZipFile(v_zip_buf, "w") as zf:
                if not pd.DataFrame(transcript_sentences).empty:
                    tx_v_buf = BytesIO()
                    pd.DataFrame(transcript_sentences).to_excel(tx_v_buf, index=False)
                    zf.writestr("transcript.xlsx", tx_v_buf.getvalue())
                if not pd.DataFrame(comment_rows).empty:
                    cm_v_buf = BytesIO()
                    pd.DataFrame(comment_rows).to_excel(cm_v_buf, index=False)
                    zf.writestr("comments.xlsx", cm_v_buf.getvalue())
                if xml_sents_t: zf.writestr(f"transcript_{video_id}.xml", build_xml_block(video_id, xml_sents_t).encode("utf-8"))
                if xml_sents_c: zf.writestr(f"comments_{video_id}.xml", build_xml_block(video_id, xml_sents_c).encode("utf-8"))
            dl_box.download_button(label=f"⬇️ {video_id} ZIP", data=v_zip_buf.getvalue(), file_name=f"youtube_{video_id}.zip", key=f"dl_{video_id}_{i}_{time.time()}", use_container_width=True)
        except Exception: pass

    st.session_state.batch_stats.update({
        "t_success": t_success, "c_success": c_success,
        "runtime": round(time.time() - start_batch, 2)
    })
    st.rerun()

# ---------------------------
# Persisted Results Rendering
# ---------------------------
if st.session_state.transcript_book or st.session_state.comment_book:
    st.divider()
    st.success("Analysis results (Persisted in Session State)")
    
    report_text = "\n".join(st.session_state.report_lines)
    report_text += f"\nTOTAL VIDEOS: {st.session_state.batch_stats['total']}\n"
    report_text += f"TRANSCRIPTS SUCCESS: {st.session_state.batch_stats['t_success']}\n"
    report_text += f"COMMENTS SUCCESS: {st.session_state.batch_stats['c_success']}\n"
    report_text += f"RUNTIME: {st.session_state.batch_stats['runtime']} seconds"

    st.subheader("📊 Output & Downloads")
    
    # Final ZIP and Excel Generation
    tx_buf = BytesIO()
    with pd.ExcelWriter(tx_buf, engine="openpyxl") as writer:
        for sheet, df in st.session_state.transcript_book.items():
            if not df.empty: df.to_excel(writer, sheet_name=sheet, index=False)
        if not st.session_state.transcript_book: pd.DataFrame({"Info": ["No transcripts"]}).to_excel(writer, sheet_name="No Data", index=False)
            
    cm_buf = BytesIO()
    with pd.ExcelWriter(cm_buf, engine="openpyxl") as writer:
        for sheet, df in st.session_state.comment_book.items():
            if not df.empty: df.to_excel(writer, sheet_name=sheet, index=False)
        if not st.session_state.comment_book: pd.DataFrame({"Info": ["No comments"]}).to_excel(writer, sheet_name="No Data", index=False)

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("dataset/transcript.xlsx", tx_buf.getvalue())
        zf.writestr("dataset/comments.xlsx", cm_buf.getvalue())
        zf.writestr("dataset/report.txt", report_text)
        for vid, sents in st.session_state.transcript_xml_data.items():
            zf.writestr(f"corpus/separate/transcript/{vid}.xml", build_xml_block(vid, sents).encode("utf-8"))
        for vid, sents in st.session_state.comment_xml_data.items():
            zf.writestr(f"corpus/separate/comments/{vid}.xml", build_xml_block(vid, sents).encode("utf-8"))
        
        merged_t = ["<root>"] + [build_xml_block(v, s) for v, s in st.session_state.transcript_xml_data.items()] + ["</root>"]
        zf.writestr("corpus/merged/transcripts.xml", "\n".join(merged_t).encode("utf-8"))
        merged_c = ["<root>"] + [build_xml_block(v, s) for v, s in st.session_state.comment_xml_data.items()] + ["</root>"]
        zf.writestr("corpus/merged/comments.xml", "\n".join(merged_c).encode("utf-8"))

    tab_ds, tab_cp = st.tabs(["📁 Dataset Preview", "🌳 Corpus Preview"])
    with tab_ds:
        v_t = [df for df in st.session_state.transcript_book.values() if not df.empty]
        v_c = [df for df in st.session_state.comment_book.values() if not df.empty]
        all_t = pd.concat(v_t) if v_t else pd.DataFrame(columns=["video_id", "sentence", "sentiment"])
        all_c = pd.concat(v_c) if v_c else pd.DataFrame(columns=["video_id", "author", "sentence", "sentiment"])
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Transcripts** ({len(all_t)} total)")
            st.dataframe(all_t.head(100), height=300)
        with col2:
            st.write(f"**Comments** ({len(all_c)} total)")
            st.dataframe(all_c.head(100), height=300)
            
    with tab_cp:
        st.write("**TreeTagger XML Format (Sample Snippet)**")
        st.code("\n".join(merged_t[:20]) + ("\n..." if len(merged_t) > 20 else ""), language="xml")

    st.divider()
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    btn_col1.download_button("⬇️ Download All (ZIP Package)", zip_buf.getvalue(), "corpus_package.zip", use_container_width=True, type="primary")
    btn_col2.download_button("⬇️ transcript.xlsx", tx_buf.getvalue(), "transcript.xlsx", use_container_width=True)
    btn_col3.download_button("⬇️ comments.xlsx", cm_buf.getvalue(), "comments.xlsx", use_container_width=True)
