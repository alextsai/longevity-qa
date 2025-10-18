# app/app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

# ---------- Minimal, safe env knobs (set before heavy imports) ----------
import os

# Force CPU on Render and quiet noisy libs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Persist model caches on Render disk to avoid cold-start download failures
DATA_ROOT_ENV = os.getenv("DATA_DIR", "/var/data")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", f"{DATA_ROOT_ENV}/models")
os.environ.setdefault("HF_HOME", f"{DATA_ROOT_ENV}/models")

# ---------- Standard imports ----------
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

# ---------- Paths / imports ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(DATA_ROOT_ENV).resolve()

# Local helper for pretty labels+URLs
from utils.labels import label_and_url

# ---------- Required artifacts under DATA_ROOT ----------
CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"
REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]

# ---------- Helpers ----------
def _parse_ts(v) -> float:
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return 0.0
    if isinstance(v, str):
        try:
            sec = 0.0
            for p in v.split(":"):
                sec = sec * 60 + float(p)
            return sec
        except Exception:
            return 0.0
    return 0.0

# ---------- Data loaders (cached) ----------
@st.cache_resource(show_spinner=False)
def load_video_meta() -> Dict[str, Dict[str, str]]:
    if VIDEO_META_JSON.exists():
        try:
            return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

@st.cache_resource(show_spinner=False)
def load_chunks_aligned() -> Tuple[List[str], List[dict]]:
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
    Returns: (faiss.Index, metas_from_pkl, {"model_name": str, "embedder": SentenceTransformer})
    """
    if not INDEX_PATH.exists() or not METAS_PKL.exists():
        return None, None, None

    index = faiss.read_index(str(INDEX_PATH))

    with METAS_PKL.open("rb") as f:
        payload = pickle.load(f)
    metas_from_pkl = payload.get("metas", [])
    model_name = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")

    embedder = SentenceTransformer(model_name, device="cpu")
    return index, metas_from_pkl, {"model_name": model_name, "embedder": embedder}

# ---------- Retrieval ----------
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

    K = max(1, min(int(initial_k), int(index.ntotal)))
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

    counts: Dict[str, int] = {}
    capped: List[Dict[str, Any]] = []
    cap = max(1, int(per_video_cap))
    for h in hits:
        vid = h["meta"].get("video_id") or "unknown"
        counts[vid] = counts.get(vid, 0) + 1
        if counts[vid] <= cap:
            capped.append(h)
        if len(capped) >= int(final_k):
            break

    return capped

# ---------- Answer synthesis ----------
def openai_answer(model_name: str, question: str, history: List[Dict[str, str]], hits: List[Dict[str, Any]]) -> str:
    key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    if not key:
        return "⚠️ OPENAI_API_KEY is not set. Add it in Render → Environment or .streamlit/secrets.toml."

    recent: List[Dict[str, str]] = []
    for m in reversed(history[:-1]):  # exclude current prompt
        if m.get("role") in ("user", "assistant") and m.get("content"):
            recent.append(m)
        if len(recent) >= 6:
            break
    convo_lines: List[str] = []
    for m in reversed(recent):
        label = "User" if m["role"] == "user" else "Assistant"
        convo_lines.append(f"{label}: {m['content']}")

    evidence: List[str] = []
    for i, h in enumerate(hits, 1):
        lbl, _ = label_and_url(h["meta"])
        evidence.append(f"[{i}] {lbl}\n{h['text']}\n")

    system = (
        "You must answer using ONLY the provided video excerpts.\n"
        "If evidence is insufficient, say you don't know.\n"
        "Be concise and practical; no medical diagnosis."
    )
    user_payload = (
        (("Recent conversation:\n" + "\n".join(convo_lines) + "\n\n") if convo_lines else "")
        + f"Question: {question}\n\nExcerpts:\n"
        + "\n".join(evidence)
        + "\nWrite the best possible answer consistent with the excerpts. Cite ideas in prose. If unsure, say you don't know."
    )

    try:
        client = OpenAI()  # reads key from env/secrets
        r = client.chat.completions.create(
            model=model_name,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_payload},
            ],
            timeout=30,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ Generation error: {e}"

# ---------- UI ----------
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

with st.sidebar:
    st.header("Settings")
    st.markdown(
        "Tune how search and synthesis work. Defaults are safe for Render CPU."
    )

    initial_k = st.number_input("Initial candidates (FAISS)", min_value=20, max_value=2000, value=320, step=20)
    final_k = st.number_input("Final evidence chunks", min_value=5, max_value=200, value=80, step=5)
    per_video_cap = st.number_input("Max chunks per video", min_value=1, max_value=50, value=12, step=1)

    st.subheader("OpenAI model")
    # Keep to Chat Completions API models available broadly
    model_choice = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
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
        ch = (info.get("channel") or m.get("channel") or m.get("uploader") or "").strip() or "Unknown"
        channel_counts[ch] = channel_counts.get(ch, 0) + 1

    if channel_counts:
        st.subheader("Primary sources")
        for ch, cnt in sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)[:4]:
            st.markdown(f"- **{ch}** — {cnt:,} chunks")

    st.divider()
    with st.expander("Debug panel", expanded=False):
        st.code("\n".join([f"Exists: {p.relative_to(DATA_ROOT)} -> {p.exists()}" for p in REQUIRED]), language="bash")
        st.caption("If any are False, fix your Render Disk contents or mount path.")

# ---------- Chat history ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about sleep, protein timing, fasting, supplements, protocols…")
if prompt is None:
    st.stop()

# Show user message immediately
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

# ---------- Hard guardrails on missing files ----------
missing_bits = [p.relative_to(DATA_ROOT) for p in REQUIRED if not p.exists()]
if missing_bits:
    with st.chat_message("assistant"):
        st.error(
            "Required files were not found in DATA_DIR. Copy them to your Render persistent disk and redeploy.\n\n"
            + "\n".join(f"- {DATA_ROOT / p}" for p in missing_bits)
        )
    st.stop()

# ---------- Load resources ----------
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

# ---------- Alignment checks ----------
if int(index.ntotal) != len(texts):
    with st.chat_message("assistant"):
        st.error(
            f"Index size mismatch. FAISS ntotal={int(index.ntotal):,} but chunks.jsonl has {len(texts):,} rows.\n"
            "Rebuild the index with the exact same chunk order."
        )
    st.stop()

if metas_from_pkl is not None and len(metas_from_pkl) != len(texts):
    with st.chat_message("assistant"):
        st.error("metas.pkl length does not match chunks.jsonl. Rebuild index artifacts together.")
    st.stop()

embedder: SentenceTransformer = payload["embedder"]
built_model: str = payload["model_name"]

# ---------- Retrieval ----------
with st.spinner("Searching sources…"):
    try:
        hits = search_chunks(
            prompt,
            index=index,
            embedder=embedder,
            metas=metas,
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

# ---------- Answer ----------
with st.chat_message("assistant"):
    if not hits:
        st.warning("No sources found for this answer.")
        st.session_state.messages.append({"role": "assistant", "content": "I couldn’t find relevant excerpts for that."})
        st.stop()

    # Fail fast if API key missing
    openai_key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    if not openai_key:
        st.error("OPENAI_API_KEY is not set. Add it in Render → Environment or .streamlit/secrets.toml.")
        st.stop()

    with st.spinner("Synthesizing answer…"):
        answer = openai_answer(
            st.session_state.get("model_choice", model_choice),
            prompt,
            st.session_state.messages,
            hits,
        )

    st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.expander("Sources & timestamps", expanded=False):
        st.markdown(f"**Encoder (index/runtime):** `{built_model}`")
        for i, h in enumerate(hits, 1):
            lbl, url = label_and_url(h["meta"])
            st.markdown(f"{i}. [{lbl}]({url})" if url else f"{i}. {lbl}")
        st.caption("Answers synthesize the indexed podcasters’ videos. If evidence is weak, I’ll say so.")
