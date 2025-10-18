# app/app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

# ---- Minimal, safe env knobs (must be set before heavy imports) ----
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU on Render

# ---- Standard imports ----
from pathlib import Path
import sys
import json
import pickle
from typing import List, Tuple, Dict, Any

import streamlit as st
import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# Paths / imports
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# All data lives under DATA_DIR (Render persistent disk defaults to /var/data)
DATA_ROOT = Path(os.getenv("DATA_DIR", "/var/data")).resolve()

# Local helper (reads video_meta.json to build nice labels/URLs)
from utils.labels import label_and_url

# -----------------------------------------------------------------------------
# Constants: required files (all under DATA_ROOT)
# -----------------------------------------------------------------------------
CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"

REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Data loaders (cached)
# -----------------------------------------------------------------------------
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
    texts: List[str] = []
    metas: List[dict] = []
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
    """
    Load FAISS index and the encoder specified in metas.pkl.
    Returns: (index, metas_from_pkl, {"model_name": str, "embedder": SentenceTransformer})
    """
    if not INDEX_PATH.exists() or not METAS_PKL.exists():
        return None, None, None

    # FAISS index
    index = faiss.read_index(str(INDEX_PATH))

    # metas.pkl carries {"metas": [...], "model": "sentence-transformers/all-MiniLM-L6-v2"}
    with METAS_PKL.open("rb") as f:
        payload = pickle.load(f)
    metas_from_pkl = payload.get("metas", [])
    model_name = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")

    # Light CPU encoder (kept cached)
    embedder = SentenceTransformer(model_name, device="cpu")

    return index, metas_from_pkl, {"model_name": model_name, "embedder": embedder}

# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------
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

    # Encode on CPU, normalized — matches the prebuilt index
    q_emb = embedder.encode([query], normalize_embeddings=True).astype("float32")

    K = min(int(initial_k), len(texts))
    D, I = index.search(q_emb, K)
    indices = I[0].tolist()
    scores = D[0].tolist()

    hits: List[Dict[str, Any]] = []
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
    counts: Dict[str, int] = {}
    capped: List[Dict[str, Any]] = []
    cap = max(1, int(per_video_cap))

    for h in hits:
        vid = h["meta"].get("video_id")
        counts[vid] = counts.get(vid, 0) + 1
        if counts[vid] <= cap:
            capped.append(h)
        if len(capped) >= int(final_k):
            break

    return capped

# -----------------------------------------------------------------------------
# Answer synthesis (OpenAI)
# -----------------------------------------------------------------------------
def openai_answer(model_name: str, question: str, history: List[Dict[str, str]], hits: List[Dict[str, Any]]) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "⚠️ OPENAI_API_KEY is not set. Add it in Render → Environment → Secret Files/Env Vars."

    # Keep last ~6 turns for conversational context
    recent = history[-6:]
    convo: List[str] = []
    for m in recent:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            label = "User" if role == "user" else "Assistant"
            convo.append(f"{label}: {content}")

    # Build evidence block
    lines: List[str] = []
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
        client = OpenAI(timeout=30)
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

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

with st.sidebar:
    st.header("Settings")
    st.markdown(
        "These settings control **how the AI searches your video library and builds answers**.\n\n"
        "You can adjust them anytime — the defaults work well for most questions."
    )

    # Memory-friendly defaults
    initial_k = st.number_input("Initial candidates (FAISS)", min_value=20, max_value=2000, value=320, step=20)
    final_k = st.number_input("Final evidence chunks", min_value=5, max_value=200, value=80, step=5)
    per_video_cap = st.number_input("Max chunks per video", min_value=1, max_value=50, value=12, step=1)

    st.subheader("OpenAI model")
    model_choice = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    st.session_state["model_choice"] = model_choice

    st.divider()
    st.subheader("Index status (under DATA_DIR)")
    st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    st.checkbox("data/chunks/chunks.jsonl", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("data/index/faiss.index", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("data/index/metas.pkl", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("data/catalog/video_meta.json", value=VIDEO_META_JSON.exists(), disabled=True)

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

    # Debug / health panel
    st.divider()
    with st.expander("Debug panel (for admin)", expanded=False):
        st.code(
            "\n".join([
                f"Exists: {p.relative_to(DATA_ROOT)} -> {p.exists()}" for p in REQUIRED
            ]), language="bash"
        )
        st.caption("If any are False, re-check your Render Disk contents & mount path.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role":"user"/"assistant", "content": str}

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Single input at bottom (follow-ups preserved)
prompt = st.chat_input("Ask about sleep, protein timing, fasting, supplements, protocols…")
if prompt is None:
    st.stop()

# Show user message immediately
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

# Hard guardrails on missing files (surface exact paths)
missing_bits = [p.relative_to(DATA_ROOT) for p in REQUIRED if not p.exists()]
if missing_bits:
    with st.chat_message("assistant"):
        st.error(
            "Required files were not found in DATA_DIR. Please copy them to your persistent disk and redeploy.\n\n"
            + "\n".join(f"- {DATA_ROOT / p}" for p in missing_bits)
        )
    st.stop()

# Load resources after we know files exist
try:
    index, metas_from_pkl, payload = load_index_and_model()
    texts, metas = load_chunks_aligned()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load FAISS index or chunks.")
        st.exception(e)
    st.stop()

if index is None or payload is None or not texts:
    with st.chat_message("assistant"):
        st.error("FAISS index / metas / chunks not found or failed to load.")
    st.stop()

embedder: SentenceTransformer = payload["embedder"]
metas_list = metas  # aligned with CHUNKS_PATH order

# Retrieval
with st.spinner("Searching sources…"):
    try:
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
    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Search failed.")
            st.exception(e)
        st.stop()

# Answer
with st.chat_message("assistant"):
    if not hits:
        st.warning("No sources found for this answer.")
        st.session_state.messages.append(
            {"role": "assistant", "content": "I couldn’t find relevant excerpts for that."}
        )
        st.stop()

    with st.spinner("Synthesizing answer…"):
        try:
            answer = openai_answer(
                st.session_state.get("model_choice", model_choice),
                prompt,
                st.session_state.messages,
                hits,
            )
        except Exception as e:
            st.error("OpenAI request failed. See error details below.")
            st.exception(e)
            raise

    st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Sources (collapsed by default)
    with st.expander("Sources & timestamps", expanded=False):
        for i, h in enumerate(hits, 1):
            lbl, url = label_and_url(h["meta"])
            st.markdown(f"{i}. [{lbl}]({url})" if url else f"{i}. {lbl}")
        st.caption("Answers synthesize the indexed podcasters’ videos. If evidence is weak, I’ll say so.")
