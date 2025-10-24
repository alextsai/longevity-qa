# app/app.py
# -*- coding: utf-8 -*-
"""
Longevity / Nutrition Q&A (Railway/Render friendly, low-RAM)
- Retrieval from curated video chunks (FAISS)
- MMR re-ranking + strict caps to keep answers focused
- Recency boost (half-life) to prefer newer videos
- Optional trusted web augmentation (hand-picked domains)
- Structured, citation-first synthesis via OpenAI
- Zero-copy chunk access (offsets) to avoid OOM
- Grouped sources (UI) and grouped evidence (prompt)
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
import pickle
from typing import List, Tuple, Dict, Any
import time
import re
from datetime import datetime

import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Optional web fetch: app works even if these aren’t available
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

# Local label helper (ok if missing)
try:
    from utils.labels import label_and_url
except Exception:
    def label_and_url(meta: dict) -> Tuple[str, str]:
        """Fallback label if utils.labels is missing."""
        vid = meta.get("video_id") or "Unknown"
        ts = int(meta.get("start", 0))
        return (f"{vid} @ {ts}s", "")

# Required artifacts (bootstrap should place them)
CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"
OFFSETS_NPY     = DATA_ROOT / "data/chunks/chunks.offsets.npy"   # created by bootstrap or lazily
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"
REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]

# Trusted web domains (add/remove here; only these are queried)
TRUSTED_DOMAINS = [
    "nih.gov",
    "medlineplus.gov",
    "cdc.gov",
    "mayoclinic.org",
    "health.harvard.edu",
    "familydoctor.org",
    "healthfinder.gov",
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
def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _parse_ts(v) -> float:
    """Parse seconds or 'HH:MM:SS' into seconds."""
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

def _iso_to_epoch(iso: str) -> float:
    """Parse ISO date or YYYY-MM-DD into epoch seconds."""
    if not iso:
        return 0.0
    try:
        if "T" in iso:
            return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
        return datetime.fromisoformat(iso).timestamp()
    except Exception:
        return 0.0

def _vid_epoch(vm: dict, vid: str) -> float:
    """Lookup publish date for a video id in video_meta.json."""
    if not isinstance(vm, dict):
        return 0.0
    info = vm.get(vid, {}) or {}
    return _iso_to_epoch(
        info.get("published_at")
        or info.get("publishedAt")
        or info.get("date")
        or ""
    )

def _recency_score(published_ts: float, now: float, half_life_days: float) -> float:
    """Exponential half-life decay; higher when newer."""
    if published_ts <= 0:
        return 0.0
    days = max(0.0, (now - published_ts) / 86400.0)
    return 0.5 ** (days / max(1e-6, half_life_days))

def _format_ts(sec: float) -> str:
    """H:MM:SS or M:SS for display."""
    sec = int(max(0, float(sec)))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

# ---------------------------------------------------------------------
# Zero-copy chunk reader (RAM-safe)
# ---------------------------------------------------------------------
def _ensure_offsets() -> np.ndarray:
    """Memory-map offsets so we can random-access JSONL without loading all lines."""
    if OFFSETS_NPY.exists():
        try:
            return np.load(OFFSETS_NPY)
        except Exception:
            pass
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
    """Yield rows from chunks.jsonl by file offsets (keeps memory low)."""
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
def load_video_meta() -> Dict[str, Dict[str, Any]]:
    if VIDEO_META_JSON.exists():
        try: return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}

@st.cache_resource(show_spinner=False)
def load_metas_and_model():
    """Load FAISS, metas, and a CPU embedder. Prefer local HF cache if present."""
    if not INDEX_PATH.exists() or not METAS_PKL.exists():
        return None, None, None
    index = faiss.read_index(str(INDEX_PATH))
    with METAS_PKL.open("rb") as f:
        payload = pickle.load(f)
    metas_from_pkl = payload.get("metas", [])
    model_name = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    local_dir = DATA_ROOT / "models" / "all-MiniLM-L6-v2"
    try_name = str(local_dir) if (local_dir / "config.json").exists() else model_name
    embedder = SentenceTransformer(try_name, device="cpu")
    return index, metas_from_pkl, {"model_name": try_name, "embedder": embedder}

# ---------------------------------------------------------------------
# MMR (diversity)
# ---------------------------------------------------------------------
def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, lambda_diversity: float = 0.4):
    """Maximal Marginal Relevance to balance relevance and novelty."""
    if doc_vecs.size == 0:
        return []
    sim_to_query = (doc_vecs @ query_vec.reshape(-1, 1)).ravel()
    selected = []
    candidates = set(range(doc_vecs.shape[0]))
    while candidates and len(selected) < k:
        if not selected:
            cand_list = list(candidates)
            pick = cand_list[int(np.argmax(sim_to_query[cand_list]))]
            selected.append(pick)
            candidates.remove(pick)
            continue
        selected_vecs = doc_vecs[selected]
        cand_list = list(candidates)
        max_div = (selected_vecs @ doc_vecs[cand_list].T).max(axis=0)  # similarity to any selected
        scores = lambda_diversity * sim_to_query[cand_list] - (1 - lambda_diversity) * max_div
        pick = cand_list[int(np.argmax(scores))]
        selected.append(pick)
        candidates.remove(pick)
    return selected

# ---------------------------------------------------------------------
# Retrieval with recency boost and strict caps
# ---------------------------------------------------------------------
def search_chunks(
    query: str,
    index: faiss.Index,
    embedder: SentenceTransformer,
    initial_k: int,
    final_k: int,
    max_videos: int,
    per_video_cap: int,
    apply_mmr: bool,
    mmr_lambda: float,
    recency_weight: float,
    half_life_days: float,
    vm: dict,
) -> List[Dict[str, Any]]:
    """Two-stage: FAISS → (optional) MMR → recency-blended score → strict caps."""
    if not query.strip():
        return []

    # 1) FAISS candidates
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
    K = min(int(initial_k), index.ntotal if index is not None else initial_k)
    D, I = index.search(qv.reshape(1, -1), K)
    indices = [int(x) for x in I[0] if x >= 0]
    scores0 = [float(s) for s in D[0][:len(indices)]]

    # 2) Read candidate rows (zero-copy)
    rows = list(iter_jsonl_rows(indices))
    texts: List[str] = []
    metas: List[dict] = []
    for _, j in rows:
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

    # 3) Re-embed candidates for MMR re-rank (light, CPU ok)
    doc_vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")

    # 4) Optional MMR to diversify before caps
    order = list(range(len(texts)))
    if apply_mmr:
        sel = mmr(qv, doc_vecs, k=min(len(texts), max(8, final_k * 2)), lambda_diversity=float(mmr_lambda))
        order = sel

    # 5) Recency blend: base FAISS score + time-decay
    now = time.time()
    blended = []
    for idx_in_order in order:
        i_global = indices[idx_in_order] if idx_in_order < len(indices) else None
        base = scores0[idx_in_order] if idx_in_order < len(scores0) else 0.0
        m = metas[idx_in_order]; t = texts[idx_in_order]
        vid = m.get("video_id")
        rec = _recency_score(_vid_epoch(vm, vid), now, half_life_days)
        score = (1.0 - recency_weight) * float(base) + recency_weight * float(rec)
        blended.append((i_global, score, t, m))

    # 6) Sort by blended score desc
    blended.sort(key=lambda x: x[1], reverse=True)

    # 7) Strict caps: use few videos and few chunks per video (keeps answers focused)
    picked: List[Dict[str, Any]] = []
    seen_per_video: Dict[str, int] = {}
    distinct_videos: List[str] = []

    for i_global, score, text, meta in blended:
        vid = meta.get("video_id", "Unknown")
        if vid not in distinct_videos and len(distinct_videos) >= int(max_videos):
            continue
        c = seen_per_video.get(vid, 0)
        if c >= int(per_video_cap):
            continue
        if vid not in distinct_videos:
            distinct_videos.append(vid)
        seen_per_video[vid] = c + 1
        picked.append({"i": i_global, "score": float(score), "text": text, "meta": meta})
        if len(picked) >= int(final_k):
            break

    return picked

# ---------------------------------------------------------------------
# Grouping helpers (for UI and prompt)
# ---------------------------------------------------------------------
def group_hits_by_video(hits: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for h in hits:
        vid = (h.get("meta") or {}).get("video_id") or "Unknown"
        groups.setdefault(vid, []).append(h)
    return groups

def build_grouped_evidence_for_prompt(hits: List[Dict[str, Any]], vm: dict, max_quotes_per_video: int = 3) -> str:
    """
    Build a concise, grouped evidence block for the LLM:
    - One header per video (title, channel, publish date if available)
    - Up to N short quotes per video with timestamp
    """
    groups = group_hits_by_video(hits)
    # Sort videos by their best score
    ordered = sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)

    lines = []
    for v_idx, (vid, items) in enumerate(ordered, 1):
        info = vm.get(vid, {}) if isinstance(vm, dict) else {}
        title = info.get("title") or vid
        channel = info.get("channel") or "Unknown"
        date = info.get("published_at") or info.get("publishedAt") or info.get("date") or ""
        head = f"[Video {v_idx}] {title} — {channel}" + (f" — {date}" if date else "")
        lines.append(head)

        # order by timestamp
        items_sorted = sorted(items, key=lambda h: float(h['meta'].get('start', 0)))
        for h in items_sorted[:max_quotes_per_video]:
            ts = _format_ts(h["meta"].get("start", 0))
            quote = (h["text"] or "").strip().replace("\n", " ")
            if len(quote) > 260:
                quote = quote[:260] + "…"
            lines.append(f"  • {ts}: “{quote}”")
        lines.append("")  # spacer
    return "\n".join(lines).strip()

# ---------------------------------------------------------------------
# Trusted web fetch (simple and safe)
# ---------------------------------------------------------------------
def fetch_trusted_snippets(query: str, max_snippets: int = 3, per_domain: int = 1, timeout: float = 6.0) -> List[Dict[str, str]]:
    """
    Very lightweight site-constrained fetch using DuckDuckGo HTML.
    Returns [{domain, url, text}]. Skips quietly if requests/bs4 unavailable.
    """
    if not requests or not BeautifulSoup or max_snippets <= 0:
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
                    paras = [p.get_text(" ", strip=True) for p in s2.find_all("p")]
                    text = _normalize_text(" ".join(paras))[:2000]
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
def openai_answer(
    model_name: str,
    question: str,
    history: List[Dict[str, str]],
    grouped_video_block: str,
    web_snips: List[Dict[str, str]],
) -> str:
    """Structured, citation-first synthesis. Prefers fewer, stronger sources."""
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ OPENAI_API_KEY is not set."

    # Keep last ~6 turns for light conversational continuity
    recent = history[-6:]
    convo: List[str] = []
    for m in recent:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            label = "User" if role == "user" else "Assistant"
            convo.append(f"{label}: {content}")

    # Web evidence block (short)
    web_lines: List[str] = []
    for j, s in enumerate(web_snips, 1):
        web_lines.append(f"(W{j}) {s['domain']} — {s['url']}\n“{s['text'][:300]}”\n")
    web_block = "\n".join(web_lines).strip() if web_lines else "None"

    system = (
        "Answer ONLY from the provided evidence. Prioritize grouped video evidence, then trusted web snippets.\n"
        "Use at most ~4 videos and ~3 trusted web snippets. Merge findings; avoid listing every quote.\n"
        "Structure:\n"
        "• Key takeaways\n"
        "• Practical protocol (clear, stepwise)\n"
        "• Safety notes and when to consult a clinician\n"
        "Cite inline like (Video 2) or (CDC W1). If evidence is insufficient, say so."
    )

    user_payload = (
        (("Recent conversation:\n" + "\n".join(convo) + "\n\n") if convo else "")
        + f"Question: {question}\n\n"
        + "Grouped Video Evidence:\n" + (grouped_video_block or "None") + "\n\n"
        + "Trusted Web Snippets:\n" + web_block + "\n\n"
        + "Write a concise, well-grounded answer."
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
    st.header("How the answer is built")

    st.markdown(
        "Adjust how much evidence we scan and how we pick it. "
        "If answers feel thin, raise the first setting a bit. "
        "If they feel scattered, lower the caps below."
    )

    # Plain-English controls
    initial_k = st.number_input(
        "How many passages to scan first",
        min_value=32, max_value=2000, value=128, step=32,
        help="First pass. We quickly scan this many likely passages from your library. Bigger = can find more ideas, but slower."
    )
    final_k = st.number_input(
        "How many passages to use",
        min_value=8, max_value=60, value=24, step=2,
        help="Second pass. We keep this many best passages to write the answer. Too high can make answers ramble."
    )

    st.subheader("Keep it focused")
    max_videos = st.number_input(
        "Maximum videos to use",
        min_value=1, max_value=12, value=4, step=1,
        help="At most this many different videos will contribute to the answer."
    )
    per_video_cap = st.number_input(
        "Passages per video",
        min_value=1, max_value=10, value=3, step=1,
        help="Stops any one video from dominating. 2–3 works well."
    )

    st.subheader("Balance variety and accuracy")
    use_mmr = st.checkbox(
        "Encourage variety (recommended)",
        value=True,
        help="Avoids near-duplicate passages so evidence covers different angles."
    )
    mmr_lambda = st.slider(
        "Balance: accuracy vs variety",
        min_value=0.1, max_value=0.9, value=0.4, step=0.05,
        help="Higher = closer match to your question. Lower = more diverse viewpoints."
    )

    st.subheader("Prefer newer videos")
    recency_weight = st.slider(
        "Recency influence",
        min_value=0.0, max_value=1.0, value=0.30, step=0.05,
        help="0 ignores date. 1 strongly prefers newer videos."
    )
    half_life = st.slider(
        "How fast recency fades (days)",
        min_value=7, max_value=720, value=180, step=7,
        help="Every N days, recency value halves. Smaller number favors very recent content."
    )

    st.subheader("Add trusted websites")
    include_web = st.checkbox(
        "Add short excerpts from trusted health sites",
        value=False,
        help="NIH, CDC, Mayo Clinic, Harvard Health, NEJM, Stanford Medicine, and similar."
    )
    max_web = st.slider(
        "Max website excerpts",
        min_value=0, max_value=8, value=3, step=1,
        help="Upper limit on website snippets we add for context."
    )

    st.subheader("Model")
    model_choice = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)

    st.divider()
    st.subheader("Library status")
    st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    st.checkbox("chunks.jsonl present", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("offsets built", value=OFFSETS_NPY.exists(), disabled=True)
    st.checkbox("faiss.index present", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("metas.pkl present", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("video_meta.json present", value=VIDEO_META_JSON.exists(), disabled=True)

    # Show channels by videos, not chunks
    vm = load_video_meta()
    counts: Dict[str, int] = {}
    for vid, info in vm.items():
        ch = (info.get("channel") or "").strip() or "Unknown"
        counts[ch] = counts.get(ch, 0) + 1
    if counts:
        st.subheader("Top channels (by videos)")
        for ch, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:6]:
            st.markdown(f"- **{ch}** — {cnt} videos")

    # Diagnostics
    st.divider()
    with st.expander("Diagnostics", expanded=False):
        ok = all(p.exists() for p in REQUIRED)
        st.write(f"All required files present: {ok}")
        try:
            ix = faiss.read_index(str(INDEX_PATH))
            with METAS_PKL.open("rb") as f:
                mp = pickle.load(f)
            st.write(f"faiss.ntotal = {ix.ntotal:,}")
            st.write(f"metas = {len(mp.get('metas', [])):,}")
            st.write(f"chunks.lines = {_ensure_offsets().shape[0]:,}")
            st.write("aligned =", ix.ntotal == len(mp.get('metas', [])) == _ensure_offsets().shape[0])
        except Exception as e:
            st.write(f"alignment check error: {e}")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about sleep, protein timing, fasting, supplements, protocols…")
if prompt is None:
    st.stop()

st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

# Guardrails: required files must exist
missing = [p for p in REQUIRED if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error(
            "I can’t find the index files. The server needs these under DATA_DIR.\n"
            + "\n".join(f"- {p}" for p in missing)
        )
    st.stop()

# Load index + model
try:
    index, metas_from_pkl, payload = load_metas_and_model()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load the index or the encoder.")
        st.exception(e)
    st.stop()

if index is None or payload is None:
    with st.chat_message("assistant"):
        st.error("Index or model not available.")
    st.stop()

embedder: SentenceTransformer = payload["embedder"]
vm = load_video_meta()

# Retrieval
with st.spinner("Searching your video library…"):
    try:
        hits = search_chunks(
            query=prompt,
            index=index,
            embedder=embedder,
            initial_k=int(initial_k),
            final_k=int(final_k),
            max_videos=int(max_videos),
            per_video_cap=int(per_video_cap),
            apply_mmr=bool(use_mmr),
            mmr_lambda=float(mmr_lambda),
            recency_weight=float(recency_weight),
            half_life_days=float(half_life),
            vm=vm,
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Search failed.")
            st.exception(e)
        st.stop()

# Optional trusted web augmentation
web_snips: List[Dict[str, str]] = []
if include_web:
    with st.spinner("Adding trusted website evidence…"):
        try:
            web_snips = fetch_trusted_snippets(prompt, max_snippets=int(max_web))
        except Exception:
            web_snips = []

# Build grouped evidence text for the model (keeps prompt compact and per-video)
grouped_video_block = build_grouped_evidence_for_prompt(hits, vm, max_quotes_per_video=3)

# Answer
with st.chat_message("assistant"):
    if not hits and not web_snips:
        st.warning("No relevant evidence found. Try a simpler wording or a related topic.")
        st.session_state.messages.append(
            {"role": "assistant", "content": "I couldn’t find enough evidence to answer that."}
        )
        st.stop()

    with st.spinner("Writing your answer…"):
        answer = openai_answer(
            model_name=model_choice,
            question=prompt,
            history=st.session_state.messages,
            grouped_video_block=grouped_video_block,
            web_snips=web_snips,
        )

    st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # -------- Grouped Sources UI (nested by video) --------
    with st.expander("Sources & timestamps", expanded=False):
        groups = group_hits_by_video(hits)
        # order by best score per video
        ordered = sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)

        for vid, items in ordered:
            info = vm.get(vid, {}) if isinstance(vm, dict) else {}
            title = info.get("title") or vid
            channel = info.get("channel") or ""
            url = info.get("url") or ""
            header = f"**{title}**" + (f" — _{channel}_" if channel else "")
            st.markdown(f"- [{header}]({url})" if url else f"- {header}")

            # nested bullets: timestamp + short quote
            for h in sorted(items, key=lambda r: float(r["meta"].get("start", 0))):
                ts = _format_ts(h["meta"].get("start", 0))
                quote = (h["text"] or "").strip().replace("\n", " ")
                if len(quote) > 140:
                    quote = quote[:140] + "…"
                st.markdown(f"  • **{ts}** — “{quote}”")

        if web_snips:
            st.markdown("**Trusted websites**")
            for j, s in enumerate(web_snips, 1):
                st.markdown(f"W{j}. [{s['domain']}]({s['url']})")

        st.caption("Chunks are grouped by video with timestamps. Trusted sites appear separately.")
