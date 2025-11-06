# app/app.py
# -*- coding: utf-8 -*-
"""
Health | Nutrition Q&A (Railway/Render friendly, low-RAM)

Fixes in this version:
- Graceful OpenAI 429 handling (insufficient_quota) with on-screen guidance.
- Video quotes show true video titles from metadata (not V1/V2).
- Trusted-sites augmentation performs query-scoped searches and excerpting.
- Unique "Users" metric uses a persistent browser cookie + SQLite (not sessions).

External, optional:
- pip install: faiss-cpu streamlit-cookies-manager httpx
"""

from __future__ import annotations

# ------------------ Minimal, safe env ------------------
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only

# ------------------ Imports -------------------
from pathlib import Path
import sys
import json
import time
import sqlite3
import uuid
import hashlib
from typing import List, Dict, Any, Tuple, Optional
import traceback

import streamlit as st
import httpx

# Optional libs
try:
    import faiss  # type: ignore
except Exception:
    faiss = None

try:
    from streamlit_cookies_manager import EncryptedCookieManager  # type: ignore
except Exception:
    EncryptedCookieManager = None

# ------------------ Configuration ------------------
APP_NAME = "Health | Nutrition Q&A"
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "video_metadata.json"   # {video_id: {title, url, channel, ...}}
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"        # [{id, video_id, text, ts, url, ...}]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # choose a light, cost-effective model

TRUSTED_SITES = [
    "nih.gov",
    "medlineplus.gov",
    "cdc.gov",
    "mayoclinic.org",
    "health.harvard.edu",
    "familydoctor.org",
]

# SQLite path for uniques; Railway ephemeral FS -> prefer /data if mounted
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", "/data/hnq.sqlite"))
SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ------------------ Utilities ------------------
def load_json(p: Path, default: Any) -> Any:
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default

def load_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    items = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items

def md_link(text: str, url: str) -> str:
    return f"[{text}]({url})"

def guard_openai():
    if not OPENAI_API_KEY:
        st.error("OpenAI key missing. Set OPENAI_API_KEY in your environment config.")
        st.stop()

# ------------------ Unique Users (cookie + sqlite) ------------------
def init_db() -> None:
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            uid TEXT PRIMARY KEY,
            first_seen INTEGER
        )
    """)
    con.commit()
    con.close()

def register_unique(uid: str) -> None:
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.cursor()
    cur.execute("INSERT OR IGNORE INTO users(uid, first_seen) VALUES(?, ?)", (uid, int(time.time())))
    con.commit()
    con.close()

def unique_count() -> int:
    con = sqlite3.connect(SQLITE_PATH)
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    n = cur.fetchone()[0]
    con.close()
    return int(n)

def get_or_set_user_cookie() -> str:
    """
    Uses EncryptedCookieManager if installed; falls back to a deterministic
    pseudo-uid from browser user agent + a random suffix stored in session.
    """
    if EncryptedCookieManager is not None:
        cookies = EncryptedCookieManager(
            prefix="hnq_",
            password=os.getenv("COOKIE_SECRET", "change-this-32-byte-secret"),
        )
        if not cookies.ready():
            st.stop()  # render once to initialize
        uid = cookies.get("uid")
        if not uid:
            uid = str(uuid.uuid4())
            cookies["uid"] = uid
            cookies.save()
        return uid

    # Fallback: approximate via session + UA hash; less accurate.
    if "fallback_uid" not in st.session_state:
        ua = st.session_state.get("_browser", "")  # Streamlit stores browser info internally
        base = hashlib.sha256((ua + str(time.time())).encode()).hexdigest()[:16]
        st.session_state["fallback_uid"] = f"fb-{base}"
    return st.session_state["fallback_uid"]

# ------------------ Retrieval data ------------------
@st.cache_data(show_spinner=False)
def load_index_and_meta():
    meta = load_json(META_PATH, {})
    chunks = load_jsonl(CHUNKS_PATH)
    index = None
    if faiss is not None and INDEX_PATH.exists():
        try:
            index = faiss.read_index(str(INDEX_PATH))
        except Exception:
            index = None
    return index, meta, chunks

# ------------------ Search trusted websites ------------------
def ddg_html_search(query: str, site: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Lightweight HTML search via DuckDuckGo's HTML endpoint.
    This avoids heavy API deps. If blocked in your env, swap with your search API.
    """
    url = "https://duckduckgo.com/html/"
    q = f"site:{site} {query}"
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(url, data={"q": q})
            r.raise_for_status()
            html = r.text
    except Exception:
        return []

    # crude parse to extract links/snippets
    results = []
    import re
    # Each result looks like: <a rel="nofollow" class="result__a" href="URL">Title</a>
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet">(.*?)</a>', html, re.S):
        link = m.group(1)
        title = re.sub("<.*?>", "", m.group(2))
        snippet = re.sub("<.*?>", "", m.group(3))
        if link.startswith("http"):
            results.append({"title": title.strip(), "url": link.strip(), "snippet": snippet.strip()})
        if len(results) >= max_results:
            break
    return results

def trusted_support(query: str, sites: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for s in sites:
        out.extend(ddg_html_search(query, s, max_results=2))
    return out[:6]

# ------------------ OpenAI client ------------------
class OpenAIClient:
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def chat(self, messages: List[Dict[str, str]], model: str, temperature: float = 0.2, max_tokens: int = 700) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            raise

# ------------------ Synthesis ------------------
def build_prompt(question: str, video_evidence: List[Dict[str, Any]], web_evidence: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    video_evidence: [{video_id, title, url, ts, quote}]
    web_evidence: [{title, url, snippet}]
    """
    v_lines = []
    for i, v in enumerate(video_evidence, 1):
        t = v.get("title") or f"Video {i}"
        ts = v.get("ts")
        ts_txt = f" @ {ts}" if ts else ""
        v_lines.append(f"- {t}{ts_txt}: \"{v.get('quote','').strip()}\" ({v.get('url','')})")
    w_lines = [f"- {w.get('title')}: {w.get('snippet')} ({w.get('url')})" for w in web_evidence]

    system = (
        "You are a cautious medical Q&A assistant. Answer with evidence from the provided quotes only. "
        "Cite inline as [V#] for video items and [W#] for web items. If uncertain, say so."
    )
    user = (
        f"Question: {question}\n\n"
        f"Video evidence:\n" + "\n".join(v_lines[:10]) + "\n\n"
        f"Trusted web evidence:\n" + "\n".join(w_lines[:10]) + "\n\n"
        "Return a concise answer with bullet points and a short, practical checklist."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

# ------------------ Retrieval helpers ------------------
def top_video_quotes(query: str, meta: Dict[str, Any], chunks: List[Dict[str, Any]], k: int = 6) -> List[Dict[str, Any]]:
    """
    Simple keyword pass filtered by selected experts in UI. For FAISS, you can swap in vector search.
    Here we do a minimal relevance ranking to avoid external deps.
    """
    q = query.lower()
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for ch in chunks:
        txt = ch.get("text", "")
        score = txt.lower().count(q) + (0.1 * len(set(q.split()) & set(txt.lower().split())))
        if score <= 0:
            # give some recall
            if any(w in txt.lower() for w in q.split()[:2]):
                score = 0.1
        if score > 0:
            v_id = ch.get("video_id")
            v = {
                "video_id": v_id,
                "title": meta.get(v_id, {}).get("title", f"Video {v_id}"),
                "url": ch.get("url") or meta.get(v_id, {}).get("url", ""),
                "ts": ch.get("ts"),
                "quote": txt.strip()[:500],
            }
            scored.append((score, v))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    seen = set()
    for _, v in scored:
        key = (v["video_id"], v["ts"])
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
        if len(out) >= k:
            break
    return out

def group_by_video(evidence: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = {}
    for v in evidence:
        g.setdefault(v["video_id"], []).append(v)
    return g

# ------------------ UI ------------------
def sidebar_controls(meta: Dict[str, Any]) -> Dict[str, Any]:
    st.sidebar.caption("Auto Mode : accuracy and diversity enabled")
    st.sidebar.markdown("### Experts")
    # Build expert filters from metadata if available
    channels = sorted({meta[v].get("channel","Unknown") for v in meta} - {""})
    default_sel = channels  # select all
    selected_channels = st.sidebar.multiselect(
        "Select which experts to include in search and answers.",
        options=channels,
        default=default_sel,
    )
    st.sidebar.markdown("### Trusted sites")
    st.sidebar.caption("Short excerpts from vetted medical sites are added as supporting evidence.")
    include_web = st.sidebar.checkbox("Include supporting website excerpts", value=True)
    site_checks = {}
    for s in TRUSTED_SITES:
        site_checks[s] = st.sidebar.checkbox(s, value=True, key=f"site_{s}")
    return {
        "channels": set(selected_channels),
        "include_web": include_web,
        "sites": [s for s,v in site_checks.items() if v],
    }

def filter_by_channels(evidence: List[Dict[str, Any]], meta: Dict[str, Any], allowed_channels: set) -> List[Dict[str, Any]]:
    if not allowed_channels:
        return evidence
    out = []
    for v in evidence:
        ch = meta.get(v["video_id"], {}).get("channel", "")
        if ch in allowed_channels:
            out.append(v)
    return out

# ------------------ Main ------------------
def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide")
    st.title("Health | Nutrition Q&A")

    # Unique users
    init_db()
    uid = get_or_set_user_cookie()
    register_unique(uid)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Unique users", value=unique_count())
    with col2:
        st.metric("Model", value=OPENAI_MODEL)
    with col3:
        st.metric("Mode", value="Evidence-first")

    # Load data
    index, meta, chunks = load_index_and_meta()
    controls = sidebar_controls(meta)

    question = st.chat_input("Ask a question, e.g., 'How to improve HRV?'")
    if question:
        st.session_state["question"] = question
    question = st.session_state.get("question", "How to improve HRV?")

    st.chat_message("user").write(question)

    # Retrieve
    video_ev = top_video_quotes(question, meta, chunks, k=8)
    video_ev = filter_by_channels(video_ev, meta, controls["channels"])

    # Web augmentation
    web_ev: List[Dict[str, str]] = []
    if controls["include_web"] and controls["sites"]:
        with st.spinner("Searching trusted websites..."):
            web_ev = trusted_support(question, controls["sites"])

    # Synthesis
    guard_openai()
    messages = build_prompt(question, video_ev, web_ev)
    error_text = None
    answer = ""
    from openai import OpenAIError  # type: ignore

    try:
        client = OpenAIClient(OPENAI_API_KEY)
        with st.spinner("Generating answer..."):
            answer = client.chat(messages, model=OPENAI_MODEL)
    except Exception as e:
        # Detect insufficient_quota explicitly
        tb = traceback.format_exc()
        e_str = str(e)
        if "insufficient_quota" in e_str or "429" in e_str:
            error_text = (
                "Generation failed with **insufficient_quota (429)**. "
                "Update billing or replace OPENAI_API_KEY in Railway. "
                "The 'Video quotes' and 'Trusted websites' tabs still work."
            )
        else:
            error_text = f"Generation failed: {e}\n\n```{tb[:700]}```"

    # ------------------ Display ------------------
    if error_text:
        st.warning(error_text)

    if answer:
        st.chat_message("assistant").markdown(answer)

    tabs = st.tabs(["Video quotes", "Trusted websites", "Web fetch trace"])

    # Tab 1: Video quotes grouped by real video titles
    with tabs[0]:
        if not video_ev:
            st.info("No quotes matched. Broaden your query or uncheck filters.")
        else:
            grouped = group_by_video(video_ev)
            for vid, items in grouped.items():
                title = meta.get(vid, {}).get("title", f"Video {vid}")
                vurl = meta.get(vid, {}).get("url", "")
                header = md_link(title, vurl) if vurl else title
                st.markdown(f"#### {header}")
                for it in items:
                    ts = f" @ {it.get('ts')}" if it.get("ts") else ""
                    url = it.get("url") or vurl
                    st.markdown(f"- **{it['ts'] or ''}** — “{it['quote']}” · {md_link('link', url)}")

    # Tab 2: Trusted websites results are query-specific, not generic
    with tabs[1]:
        if not controls["include_web"]:
            st.info("Trusted website augmentation disabled.")
        elif not web_ev:
            st.info("No trusted-site results matched the query.")
        else:
            for i, w in enumerate(web_ev, 1):
                st.markdown(f"- **W{i}: {md_link(w['title'], w['url'])}** · {w['snippet']}")

    # Tab 3: Debug trace for transparency
    with tabs[2]:
        st.code(json.dumps({
            "question": question,
            "selected_channels": list(controls["channels"]),
            "sites": controls["sites"],
            "video_evidence_sample": video_ev[:3],
            "web_evidence_sample": web_ev[:3],
            "model": OPENAI_MODEL,
        }, indent=2))


if __name__ == "__main__":
    main()
