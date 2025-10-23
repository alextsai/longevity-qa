# app/app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

# ------------------ Minimal, safe env ------------------
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only

# ------------------ Standard imports -------------------
from pathlib import Path
import sys
import json
import pickle
from typing import List, Tuple, Dict, Any
import math
import time
import re

import streamlit as st
import numpy as np
import faiss

# Model
from sentence_transformers import SentenceTransformer

# OpenAI
from openai import OpenAI

# Optional web fetch (kept resilient; app still runs without)
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

# ---------------------------------------------------------------------
# Paths / setup
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(os.getenv("DATA_DIR", "/var/data")).resolve()

# Local label helper
try:
    from utils.labels import label_and_url
except Exception:
    # Fallback if helper missing
    def label_and_url(meta: dict) -> Tuple[str, str]:
        vid = meta.get("video_id") or "Unknown"
        start = meta.get("start", 0)
        ts = int(start)
        return (f"{vid} @ {ts}s", "")

# Required files
CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"
OFFSETS_NPY     = DATA_ROOT / "data/chunks/chunks.offsets.npy"  # created by bootstrap
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"

REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]

# ---------------------------------------------------------------------
# Trusted web domains
# ---------------------------------------------------------------------
TRUSTED_DOMAINS = [
    "nih.gov",
    "medlineplus.gov",
    "cdc.gov",
    "mayoclinic.org",
    "health.harvard.edu",
    "familydoctor.org",
    "healthfinder.gov",
    # AMA list is a PDF index; still allow:
    "ama-assn.org",
    "medicalxpress.com",
    "sciencedaily.com",
    "nejm.org",
    "med.stanford.edu",
    "icahn.mssm.edu",
]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _parse_ts(v) -> float:
    if isinstance(v, (int, float)):
        try: return float(v)
        except Exception: return 0.0
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

def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

# ---------------------------------------------------------------------
# Zero-copy chunk reader (RAM-safe)
# ---------------------------------------------------------------------
def _ensure_offsets() -> np.ndarray:
    if OFFSETS_NPY.exists():
        try:
            return np.load(OFFSETS_NPY)
        except Exception:
            pass
    # build offsets lazily if missing
    pos = 0
    offs = []
    with CHUNKS_PATH.open("rb") as f:
        for ln in f:
            offs.append(pos)
            pos += len(ln)
    arr = np.array(offs, dtype=np.int64)
    OFFSETS_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OFFSETS_NPY, arr)
    return arr

def iter_jsonl_rows(indices: List[int], limit: int | None = None):
    """Yield rows by file offsets without loading entire file."""
    if not CHUNKS_PATH.exists():
        return
    offsets = _ensure_offsets()
    want = [i for i in indices if 0 <= i < len(offsets)]
    if limit is not None:
        want = want[:limit]
    with CHUNKS_PATH.open("rb") as f:
        for i in want:
            f.seek(int(offsets[i]))
            raw = f.readline()
            try:
                j = json.loads(raw)
                yield i, j
            except Exception:
                continue

# ---------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_video_meta() -> Dict[str, Dict[str, str]]:
    if VIDEO_META_JSON.exists():
        try: return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}

@st.cache_resource(show_spinner=False)
def load_metas_and_model():
    """Load FAISS, metas, and a CPU embedder. Avoid HF network if model is local."""
    if not INDEX_PATH.exists() or not METAS_PKL.exists():
        return None, None, None

    index = faiss.read_index(str(INDEX_PATH))

    with METAS_PKL.open("rb") as f:
        payload = pickle.load(f)
    metas_from_pkl = payload.get("metas", [])
    model_name = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")

    # Prefer local cached model if present
    local_dir = DATA_ROOT / "models" / "all-MiniLM-L6-v2"
    try_name = str(local_dir) if (local_dir / "config.json").exists() else model_name

    embedder = SentenceTransformer(try_name, device="cpu")
    return index, metas_from_pkl, {"model_name": try_name, "embedder": embedder}

# ---------------------------------------------------------------------
# Retrieval + MMR re-rank
# ---------------------------------------------------------------------
def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, lambda_diversity: float = 0.4):
    """
    Maximal Marginal Relevance.
    query_vec: (d,)
    doc_vecs: (N, d) normalized
    returns list of selected indices
    """
    if doc_vecs.size == 0:
        return []
    sim_to_query = (doc_vecs @ query_vec.reshape(-1, 1)).ravel()  # cosine (normalized)
    selected = []
    candidates = set(range(doc_vecs.shape[0]))
    while candidates and len(selected) < k:
        if not selected:
            i = int(np.argmax(sim_to_query[list(candidates)]))
            pick = list(candidates)[i]
            selected.append(pick)
            candidates.remove(pick)
            continue
        # diversity term = max sim to any already selected
        selected_vecs = doc_vecs[selected]
        div = selected_vecs @ doc_vecs[list(candidates)].T  # (len(sel), |cand|)
        max_div = div.max(axis=0)
        cand_list = list(candidates)
        scores = lambda_diversity * sim_to_query[cand_list] - (1 - lambda_diversity) * max_div
        pick = cand_list[int(np.argmax(scores))]
        selected.append(pick)
        candidates.remove(pick)
    return selected

def search_chunks(
    query: str,
    index: faiss.Index,
    embedder: SentenceTransformer,
    initial_k: int,
    final_k: int,
    per_video_cap: int,
    apply_mmr: bool,
    mmr_lambda: float,
) -> List[Dict[str, Any]]:
    if not query.strip():
        return []

    # Encode query normalized
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]

    # First-pass FAISS search
    K = min(int(initial_k), index.ntotal if index is not None else initial_k)
    D, I = index.search(qv.reshape(1, -1), K)
    indices = [int(x) for x in I[0] if x >= 0]
    scores0 = [float(s) for s in D[0][:len(indices)]]

    # Read candidate rows without loading entire file
    raw_rows: List[Tuple[int, dict]] = list(iter_jsonl_rows(indices))
    texts: List[str] = []
    metas: List[dict] = []
    for _, j in raw_rows:
        txt = _normalize_text(j.get("text", ""))
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
            m["start"] = m.get("start_sec")
        m["start"] = _parse_ts(m.get("start", 0))
        if txt:
            texts.append(txt)
            metas.append(m)

    if not texts:
        return []

    # Embed candidate texts lightweight for re-rank (same encoder)
    doc_vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")

    # Optional MMR diversity
    order = list(range(len(texts)))
    if apply_mmr:
        sel = mmr(qv, doc_vecs, k=min(len(texts), max(8, final_k * 2)), lambda_diversity=float(mmr_lambda))
        order = sel

    # Pack hits with original scores (for tie-breaks)
    hits: List[Dict[str, Any]] = []
    for idx_in_order in order:
        i_global = indices[idx_in_order] if idx_in_order < len(indices) else None
        score = scores0[idx_in_order] if idx_in_order < len(scores0) else 0.0
        hits.append({"i": i_global, "score": float(score), "text": texts[idx_in_order], "meta": metas[idx_in_order]})

    # Cap per-video and trim to final_k
    seen_per_video: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    cap = max(1, int(per_video_cap))
    for h in hits:
        vid = h["meta"].get("video_id", "Unknown")
        seen_per_video[vid] = seen_per_video.get(vid, 0) + 1
        if seen_per_video[vid] <= cap:
            out.append(h)
        if len(out) >= int(final_k):
            break

    return out

# ---------------------------------------------------------------------
# Trusted web fetch (simple and safe)
# ---------------------------------------------------------------------
def fetch_trusted_snippets(query: str, max_snippets: int = 4, per_domain: int = 1, timeout: float = 6.0) -> List[Dict[str, str]]:
    """
    Very lightweight site-constrained fetch using DuckDuckGo HTML.
    Returns [{domain, url, text}], truncated. Skips if requests/bs4 absent.
    """
    if not requests or not BeautifulSoup:
        return []

    headers = {"User-Agent": "Mozilla/5.0"}
    snippets: List[Dict[str, str]] = []
    for domain in TRUSTED_DOMAINS:
        try:
            q = f"site:{domain} {query}"
            resp = requests.get("https://duckduckgo.com/html/", params={"q": q}, headers=headers, timeout=timeout)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            links = []
            for a in soup.select("a.result__a"):
                href = a.get("href")
                if href and domain in href:
                    links.append(href)
            links = links[:per_domain]
            for url in links:
                try:
                    r2 = requests.get(url, headers=headers, timeout=timeout)
                    if r2.status_code != 200:
                        continue
                    s2 = BeautifulSoup(r2.text, "html.parser")
                    # simple paragraph harvest
                    paras = [p.get_text(" ", strip=True) for p in s2.find_all("p")]
                    text = _normalize_text(" ".join(paras))[:1500]
                    if len(text) < 200:
                        continue
                    snippets.append({"domain": domain, "url": url, "text": text})
                except Exception:
                    continue
            if len(snippets) >= max_snippets:
                break
        except Exception:
            continue
    return snippets[:max_snippets]

# ---------------------------------------------------------------------
# Answer synthesis (OpenAI)
# ---------------------------------------------------------------------
def openai_answer(model_name: str, question: str, history: List[Dict[str, str]], hits: List[Dict[str, Any]], web_snips: List[Dict[str, str]]) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "⚠️ OPENAI_API_KEY is not set. Add it in your hosting environment."

    # Keep last ~6 turns for light context
    recent = history[-6:]
    convo: List[str] = []
    for m in recent:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            label = "User" if role == "user" else "Assistant"
            convo.append(f"{label}: {content}")

    # Build evidence block from video chunks
    lines: List[str] = []
    for i, h in enumerate(hits, 1):
        lbl, _ = label_and_url(h["meta"])
        # Include a short quoted span for grounding
        span = h["text"][:300]
        lines.append(f"[{i}] {lbl}\n“{span}”\n")

    # Add trusted web snippets if any
    web_lines: List[str] = []
    for j, s in enumerate(web_snips, 1):
        web_lines.append(f"(W{j}) {s['domain']} — {s['url']}\n“{s['text'][:300]}”\n")

    system = (
        "You answer ONLY with the provided evidence. Prioritize video excerpts, then trusted web snippets.\n"
        "Rules:\n"
        "• Do not invent facts. If uncertain, say you don't know.\n"
        "• Provide a concise, structured answer:\n"
        "  - Key takeaways\n"
        "  - Practical protocol or steps\n"
        "  - Safety notes and when to consult a clinician\n"
        "  - Sources cited inline in prose, like (Video 2) or (CDC W1)\n"
        "• Avoid medical diagnosis. Offer general guidance only."
    )

    user_payload = (
        (("Recent conversation:\n" + "\n".join(convo) + "\n\n") if convo else "")
        + f"Question: {question}\n\n"
        + "Video Excerpts:\n" + ("\n".join(lines) if lines else "None\n")
        + ("\nTrusted Web Snippets:\n" + "\n".join(web_lines) if web_lines else "\nTrusted Web Snippets: None\n")
        + "\nWrite the best possible answer fully consistent with this evidence."
    )

    try:
        client = OpenAI(timeout=45)
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

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

with st.sidebar:
    st.header("Settings")

    st.markdown(
        "Tune how the system searches your library and builds answers. "
        "Defaults are safe for most questions."
    )

    # Retrieval knobs with tooltips
    initial_k = st.number_input(
        "Candidate chunks searched",
        min_value=32, max_value=2000, value=128, step=32,
        help="How many top matches we pull from the index before re-ranking. Higher finds more but is slower."
    )
    final_k = st.number_input(
        "Evidence chunks kept",
        min_value=10, max_value=120, value=40, step=5,
        help="How many of the best chunks we keep to build your answer. Too high can dilute relevance."
    )
    per_video_cap = st.number_input(
        "Max chunks per video",
        min_value=1, max_value=20, value=6, step=1,
        help="Prevents one video from dominating. Keeps results diverse across sources."
    )

    st.subheader("Diversity boost")
    use_mmr = st.checkbox(
        "Use diversity (MMR)",
        value=True,
        help="Encourages variety among sources while staying relevant. Good for broader questions."
    )
    mmr_lambda = st.slider(
        "MMR balance",
        min_value=0.1, max_value=0.9, value=0.4, step=0.05,
        help="Higher = more relevance, lower = more diversity in selected chunks."
    )

    st.subheader("OpenAI model")
    model_choice = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)

    st.subheader("Trusted web boost")
    include_web = st.checkbox(
        "Include trusted web sources",
        value=False,
        help="Adds brief evidence from NIH, CDC, Mayo Clinic, NEJM, Stanford Medicine, and similar sites."
    )
    max_web = st.slider(
        "Max web snippets",
        min_value=0, max_value=8, value=3, step=1,
        help="Upper bound on web excerpts merged into the answer."
    )

    st.divider()
    st.subheader("Index status")
    st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    st.checkbox("data/chunks/chunks.jsonl", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("data/chunks/chunks.offsets.npy", value=OFFSETS_NPY.exists(), disabled=True)
    st.checkbox("data/index/faiss.index", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("data/index/metas.pkl", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("data/catalog/video_meta.json", value=VIDEO_META_JSON.exists(), disabled=True)

    # Primary sources summary
    vm = load_video_meta()
    # count videos per channel
    counts: Dict[str, int] = {}
    for vid, info in vm.items():
        ch = (info.get("channel") or "").strip() or "Unknown"
        counts[ch] = counts.get(ch, 0) + 1
    if counts:
        st.subheader("Primary sources")
        st.caption("Channels with most videos indexed")
        for ch, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:6]:
            url = ""
            # try to link to channel url if present in meta (optional)
            if isinstance(vm, dict):
                # find any video with this channel that has a url
                for _vid, _info in vm.items():
                    if (_info.get("channel") or "").strip() == ch:
                        url = _info.get("url", "")
                        break
            if url:
                st.markdown(f"- [{ch}]({url}) — **{cnt} videos**")
            else:
                st.markdown(f"- **{ch}** — **{cnt} videos**")

    # Health / debug
    st.divider()
    with st.expander("Health checks", expanded=False):
        ok = all(p.exists() for p in REQUIRED)
        st.write(f"Files present: {ok}")
        # show alignment counts if possible
        try:
            ix = faiss.read_index(str(INDEX_PATH))
            with METAS_PKL.open("rb") as f:
                mp = pickle.load(f)
            metas_n = len(mp.get("metas", []))
            offsets = _ensure_offsets()
            st.write(f"faiss.ntotal = {ix.ntotal:,}")
            st.write(f"metas = {metas_n:,}")
            st.write(f"chunks.lines = {len(offsets):,}")
            st.write("aligned =", ix.ntotal == metas_n == len(offsets))
        except Exception as e:
            st.write(f"alignment check error: {e}")

        # OpenAI ping
        try:
            if os.getenv("OPENAI_API_KEY"):
                client = OpenAI(timeout=10)
                ping = client.chat.completions.create(model=model_choice, messages=[{"role": "user", "content": "ping"}])
                st.write("OpenAI ping:", "ok" if ping.choices else "err")
            else:
                st.write("OpenAI ping: NO_KEY")
        except Exception as e:
            st.write(f"OpenAI ping error: {e}")

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about sleep, protein timing, fasting, supplements, protocols…")
if prompt is None:
    st.stop()

# Show user message
st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

# Hard guardrails
missing = [p for p in REQUIRED if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error(
            "Required files were not found. The service’s bootstrap must place these under DATA_DIR.\n\n"
            + "\n".join(f"- {p}" for p in missing)
        )
    st.stop()

# Load index + model
try:
    index, metas_from_pkl, payload = load_metas_and_model()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load FAISS index or embedder.")
        st.exception(e)
    st.stop()

if index is None or payload is None:
    with st.chat_message("assistant"):
        st.error("Index or model not available.")
    st.stop()

embedder: SentenceTransformer = payload["embedder"]

# Retrieval
with st.spinner("Searching library…"):
    try:
        hits = search_chunks(
            prompt,
            index=index,
            embedder=embedder,
            initial_k=int(initial_k),
            final_k=int(final_k),
            per_video_cap=int(per_video_cap),
            apply_mmr=bool(use_mmr),
            mmr_lambda=float(mmr_lambda),
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Search failed.")
            st.exception(e)
        st.stop()

# Optional trusted web augmentation
web_snips: List[Dict[str, str]] = []
if include_web:
    with st.spinner("Fetching trusted web sources…"):
        try:
            web_snips = fetch_trusted_snippets(prompt, max_snippets=int(max_web))
        except Exception:
            web_snips = []

# Answer
with st.chat_message("assistant"):
    if not hits and not web_snips:
        st.warning("No evidence found. Try a simpler phrasing or broaden the topic.")
        st.session_state.messages.append(
            {"role": "assistant", "content": "I couldn’t find sufficient evidence to answer that."}
        )
        st.stop()

    with st.spinner("Synthesizing…"):
        answer = openai_answer(
            model_choice,
            prompt,
            st.session_state.messages,
            hits,
            web_snips,
        )

    st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Sources
    with st.expander("Sources & timestamps", expanded=False):
        # Video sources
        for i, h in enumerate(hits, 1):
            lbl, url = label_and_url(h["meta"])
            st.markdown(f"{i}. [{lbl}]({url})" if url else f"{i}. {lbl}")
        # Web sources
        if web_snips:
            st.markdown("**Trusted web sources**")
            for j, s in enumerate(web_snips, 1):
                st.markdown(f"W{j}. [{s['domain']}]({s['url']})")
        st.caption("Answers synthesize video excerpts and optionally trusted web snippets. If evidence is weak, I’ll say so.")
