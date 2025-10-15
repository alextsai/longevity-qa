# app/app.py
# -*- coding: utf-8 -*-

# --- Ensure project root is importable (for utils.labels) ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------

import os
import json
import pickle
from typing import List, Tuple, Dict, Any

import numpy as np
import faiss
import streamlit as st

# --- Auto-download data bundle if missing ---
import tarfile, tempfile, urllib.request, os

def _safe_extract_flat(tar: tarfile.TarFile, dst: str):
    """Extract bundle to dst, stripping one top-level folder if present."""
    members = tar.getmembers()

    # Detect a common top-level prefix (e.g. "data_bundle/")
    def top_prefix(ms):
        parts = [m.name.split('/', 1)[0] for m in ms if m.name]
        return parts[0] if parts and all(p == parts[0] for p in parts) else ""

    top = top_prefix(members)
    dst_abs = os.path.abspath(dst) + os.sep

    for m in members:
        name = m.name
        if top and name.startswith(top + "/"):
            name = name[len(top) + 1:]
        if not name:  # skip the top dir itself
            continue

        # prevent path traversal
        out_path = os.path.abspath(os.path.join(dst, name))
        if not out_path.startswith(dst_abs):
            continue

        m.name = name  # rewrite to our cleaned relative path
        tar.extract(m, dst)

def ensure_data_files():
    needed = [
        "data/index/faiss.index",
        "data/index/metas.pkl",
        "data/chunks/chunks.jsonl",
        "data/catalog/video_meta.json",
    ]
    if all(os.path.exists(p) for p in needed):
        return

    url = os.getenv("DATA_URL")
    if not url:
        st.error("DATA_URL not set in secrets; cannot download data bundle.")
        return

    st.warning("📦 Downloading pre-built index bundle — please wait (first run only)…")
    tmp = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(url, tmp.name)

    with tarfile.open(tmp.name, "r:gz") as tar:
        _safe_extract_flat(tar, ".")

    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        st.error(f"Bundle extracted, but missing: {missing}")
    else:
        st.success("✅ Data bundle extracted successfully!")

ensure_data_files()


from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Uses your existing helper that reads data/catalog/video_meta.json internally
from utils.labels import label_and_url  # signature: label_and_url(meta) -> (label, url)

# -------------------------
# Paths
# -------------------------
CHUNKS_PATH = ROOT / "data/chunks/chunks.jsonl"
INDEX_PATH = ROOT / "data/index/faiss.index"
METAS_PKL = ROOT / "data/index/metas.pkl"
VIDEO_META_JSON = ROOT / "data/catalog/video_meta.json"

# -------------------------
# Small helpers
# -------------------------
def _parse_ts(v) -> float:
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return 0.0
    if isinstance(v, str):
        parts = v.split(":")
        try:
            sec = 0.0
            for p in parts:
                sec = sec * 60 + float(p)
            return sec
        except Exception:
            return 0.0
    return 0.0

# -------------------------
# Data loaders (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_video_meta() -> Dict[str, Dict[str, str]]:
    if VIDEO_META_JSON.exists():
        try:
            return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

@st.cache_resource(show_spinner=False)
def load_chunks_aligned() -> Tuple[List[str], List[dict]]:
    """Load chunk texts + metas, normalizing video_id and start timestamp."""
    texts, metas = [], []
    if not CHUNKS_PATH.exists():
        return texts, metas
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            txt = (j.get("text") or "").strip()
            m = (j.get("meta") or {}).copy()

            vid = (
                m.get("video_id")
                or m.get("vid")
                or m.get("ytid")
                or j.get("video_id")
                or j.get("vid")
                or j.get("ytid")
                or j.get("id")
            )
            if vid:
                m["video_id"] = vid

            if "start" not in m and "start_sec" in m:
                m["start"] = m["start_sec"]
            m["start"] = _parse_ts(m.get("start", 0))

            if txt:
                texts.append(txt)
                metas.append(m)
    return texts, metas

@st.cache_resource(show_spinner=False)
def load_index_and_model():
    """Load FAISS index + metas.pkl (model name, metas)."""
    if not INDEX_PATH.exists() or not METAS_PKL.exists():
        return None, None, None
    index = faiss.read_index(str(INDEX_PATH))
    with METAS_PKL.open("rb") as f:
        payload = pickle.load(f)
    metas = payload.get("metas", [])
    model_name = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    embedder = SentenceTransformer(model_name)
    return index, metas, {"model_name": model_name, "embedder": embedder}

# -------------------------
# Retrieval (FAISS-only, same as before)
# -------------------------
def search_chunks(
    query: str,
    index: faiss.Index,
    embedder: SentenceTransformer,
    metas: List[dict],
    texts: List[str],
    initial_k: int,
    final_k: int,
    per_video_cap: int,
) -> List[Dict[str, Any]]:
    if not query.strip():
        return []
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")
    K = min(int(initial_k), len(texts))
    D, I = index.search(q_emb, K)
    indices = I[0].tolist()
    scores = D[0].tolist()

    hits = []
    for idx, score in zip(indices, scores):
        if idx < 0 or idx >= len(texts):
            continue
        meta = metas[idx] if idx < len(metas) else {}
        if not meta.get("video_id"):
            vid = meta.get("vid") or meta.get("ytid")
            if vid:
                meta["video_id"] = vid
        meta["start"] = _parse_ts(meta.get("start", 0))
        hits.append({"i": idx, "score": float(score), "text": texts[idx], "meta": meta})

    # Cap per-video and trim to final_k
    counts = {}
    capped = []
    for h in hits:
        vid = h["meta"].get("video_id")
        counts[vid] = counts.get(vid, 0) + 1
        if counts[vid] <= max(1, int(per_video_cap)):
            capped.append(h)
        if len(capped) >= int(final_k):
            break
    return capped

# -------------------------
# Answer synthesis (OpenAI)
# -------------------------
def openai_answer(model_name: str, question: str, history: List[Dict[str, str]], hits: List[Dict[str, Any]]) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "⚠️ OPENAI_API_KEY is not set. Export it and try again."

    # Keep last ~6 turns for conversational context (lightweight)
    recent = history[-6:]
    convo = []
    for m in recent:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            # We do NOT rely on prior answers as ground truth—model must use excerpts.
            label = "User" if role == "user" else "Assistant"
            convo.append(f"{label}: {content}")

    # Build evidence block
    lines = []
    for i, h in enumerate(hits, 1):
        lbl, _ = label_and_url(h["meta"])
        lines.append(f"[{i}] {lbl}\n{h['text']}\n")

    system = (
        "You are a careful assistant that answers using ONLY the provided video excerpts.\n"
        "Rules:\n"
        "• Never contradict the excerpts; do not invent facts.\n"
        "• If evidence is insufficient or unclear, say you don't know.\n"
        "• Be practical and concise; avoid medical diagnosis—suggest consulting a clinician when appropriate.\n"
        "• You may keep conversational continuity, but all factual claims must be grounded in the excerpts."
    )

    user_payload = (
        ("Recent conversation to preserve context:\n" + "\n".join(convo) + "\n\n") if convo else ""
    ) + (
        f"Question: {question}\n\nExcerpts:\n" + "\n".join(lines) +
        "\nWrite the best possible answer fully consistent with these excerpts. "
        "Cite ideas in prose (no footnotes). If unsure, say you don't know."
    )

    try:
        client = OpenAI()
        r = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_payload},
            ],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ Generation error: {e}"

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    # Plain-English explainer
    st.markdown("""
These settings control **how the AI searches your video library and builds answers**.  
You can adjust them anytime — the defaults work well for most questions.
""")

    initial_k = st.number_input("Initial candidates (FAISS)", min_value=20, max_value=2000, value=320, step=20)
    final_k = st.number_input("Final evidence chunks", min_value=5, max_value=200, value=80, step=5)
    per_video_cap = st.number_input("Max chunks per video", min_value=1, max_value=50, value=12, step=1)

    st.subheader("What do these settings do?")
    st.markdown("""
**Initial candidates (FAISS)**  
How many short video segments the system pulls up **before** filtering for relevance.  
- Higher = **broader** search (can find more context).  
- Lower = **faster**, but might miss things.

**Final evidence chunks**  
How many of the best segments are sent to the AI to **write your answer**.  
- Higher = **richer**, more complete answer (slower).  
- Lower = **snappier**, but less detailed.

**Max chunks per video**  
Stops one video from **dominating** the answer so you get multiple perspectives.
""")

    model_choice = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)

    st.divider()
    st.subheader("Index status")
    st.checkbox("data/chunks/chunks.jsonl", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("data/index/faiss.index", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("data/index/metas.pkl", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("data/catalog/video_meta.json", value=VIDEO_META_JSON.exists(), disabled=True)

    # Show basic stats + top 4 channels
    texts_cache, metas_cache = load_chunks_aligned()
    st.markdown(f"**Chunks indexed:** {len(texts_cache):,}")

    vm = load_video_meta()
    channel_counts: Dict[str, int] = {}
    for m in metas_cache:
        vid = m.get("video_id")
        if not vid:
            continue
        info = vm.get(vid, {})
        ch = (info.get("channel") or m.get("channel") or m.get("uploader") or "").strip()
        if not ch:
            ch = "Unknown"
        channel_counts[ch] = channel_counts.get(ch, 0) + 1

    if channel_counts:
        st.subheader("Primary sources")
        for ch, cnt in sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)[:4]:
            st.markdown(f"- **{ch}** — {cnt:,} chunks")

st.title("Longevity / Nutrition Q&A")

# --- Chat history state ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role":"user"/"assistant", "content": str}

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Single input at bottom (follow-ups preserved)
prompt = st.chat_input("Ask about sleep, protein timing, fasting, supplements, protocols…")
if prompt is None:
    st.stop()

# Append user message and show it immediately
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

# --- Load resources (once user asks) ---
index, metas_from_pkl, payload = load_index_and_model()
texts, metas = load_chunks_aligned()

if index is None or payload is None or not texts:
    with st.chat_message("assistant"):
        st.error("FAISS index / metas / chunks not found. Rebuild the index first.")
    st.stop()

embedder = payload["embedder"]
# Always use the aligned metas we just loaded (matches CHUNKS_PATH line order)
metas_list = metas

# Retrieval
with st.spinner("Searching sources…"):
    hits = search_chunks(
        prompt,
        index=index,
        embedder=embedder,
        metas=metas_list,
        texts=texts,
        initial_k=int(initial_k),
        final_k=int(final_k),
        per_video_cap=int(per_video_cap),
    )

# Answer
with st.chat_message("assistant"):
    if not hits:
        st.warning("No sources found for this answer.")
        st.session_state.messages.append({"role": "assistant", "content": "I couldn’t find relevant excerpts for that."})
        st.stop()

    with st.spinner("Synthesizing answer…"):
        answer = openai_answer(model_choice, prompt, st.session_state.messages, hits)

    st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Sources (compact, always show)
    st.markdown("**Sources & timestamps**")
    for i, h in enumerate(hits, 1):
        lbl, url = label_and_url(h["meta"])
        if url:
            st.markdown(f"{i}. [{lbl}]({url})")
        else:
            st.markdown(f"{i}. {lbl}")

st.caption("Answers synthesize the indexed podcasters’ videos. If evidence is weak, I’ll say so.")