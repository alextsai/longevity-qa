# app/app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

# ---- Minimal, safe env knobs (must be set before heavy imports) ----
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU on PaaS

# ---- Standard imports ----
from pathlib import Path
import sys
import json
import pickle
import math
from typing import List, Tuple, Dict, Any, Iterable, Optional

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Page config MUST be the first Streamlit call (and only once)
# ──────────────────────────────────────────────────────────────────────────────
if not st.session_state.get("_page_config_set"):
    st.set_page_config(
        page_title="Longevity / Nutrition Q&A",
        page_icon="🍎",
        layout="wide"
    )
    st.session_state["_page_config_set"] = True

# -----------------------------------------------------------------------------
# Paths / imports
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# All data lives under DATA_DIR (Railway/Render persistent disk -> /var/data)
DATA_ROOT = Path(os.getenv("DATA_DIR", "/var/data")).resolve()

# Local helper (reads video_meta.json to build nice labels/URLs)
# Safe import: fallback if missing
def _label_and_url_fallback(meta: dict) -> Tuple[str, Optional[str]]:
    vid = (meta or {}).get("video_id") or "unknown"
    ts = meta.get("start", 0)
    h = int(ts // 3600); m = int((ts % 3600) // 60); s = int(ts % 60)
    hhmmss = f"{h:02d}:{m:02d}:{s:02d}"
    return (f"{vid} @ {hhmmss}", None)

try:
    from utils.labels import label_and_url as label_and_url
except Exception:
    label_and_url = _label_and_url_fallback

# -----------------------------------------------------------------------------
# Constants: required files (all under DATA_ROOT)
# -----------------------------------------------------------------------------
CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"
OFFSETS_NPY     = DATA_ROOT / "data/chunks/chunks.offsets.npy"  # random-access
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"
VID_CENTROIDS   = DATA_ROOT / "data/index/video_centroids.npy"
VID_IDS_TXT     = DATA_ROOT / "data/index/video_ids.txt"
VID_SUMMARIES   = DATA_ROOT / "data/catalog/video_summaries.json"

REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]

# -----------------------------------------------------------------------------
# Small helpers
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

def _safe_json_loads(line: str) -> Optional[dict]:
    try:
        return json.loads(line)
    except Exception:
        return None

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
def load_offsets() -> Optional[np.ndarray]:
    """Byte offsets per line to avoid loading the whole .jsonl into memory."""
    if OFFSETS_NPY.exists():
        try:
            arr = np.load(OFFSETS_NPY, mmap_mode="r")
            return arr
        except Exception:
            return None
    return None

@st.cache_resource(show_spinner=False)
def load_chunks_head(n: int = 5) -> List[str]:
    """Tiny peek to confirm file shape in the UI."""
    out = []
    if not CHUNKS_PATH.exists():
        return out
    try:
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            for i, ln in enumerate(f):
                if i >= n: break
                j = _safe_json_loads(ln)
                if not j: continue
                t = (j.get("text") or "").strip()
                if t: out.append(t[:120])
    except Exception:
        pass
    return out

@st.cache_resource(show_spinner=False)
def load_index_and_model():
    """
    Load FAISS index and the encoder specified in metas.pkl.
    Returns: (index, metas_from_pkl, {"model_name": str, "embedder": SentenceTransformer})
    """
    if not INDEX_PATH.exists() or not METAS_PKL.exists():
        return None, None, None

    index = faiss.read_index(str(INDEX_PATH))

    with METAS_PKL.open("rb") as f:
        payload = pickle.load(f)
    metas_from_pkl = payload.get("metas", [])
    model_name = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")

    # Cached CPU encoder
    model_dir = os.getenv("SENTENCE_TRANSFORMERS_HOME") or None
    try:
        # Prefer local cache if exists
        local_model = None
        if model_dir:
            cand = Path(model_dir) / "all-MiniLM-L6-v2"
            if (cand / "config.json").exists():
                local_model = str(cand)
        embedder = SentenceTransformer(local_model or model_name, device="cpu")
    except Exception:
        # Fallback to model name; if HF is rate-limited the index still works,
        # and the app will display a helpful error in the answer phase.
        embedder = SentenceTransformer(model_name, device="cpu")

    return index, metas_from_pkl, {"model_name": model_name, "embedder": embedder}

@st.cache_resource(show_spinner=False)
def load_centroids() -> Optional[Dict[str, Any]]:
    """Optional stage-A routing artifacts: one centroid per video."""
    try:
        if VID_CENTROIDS.exists() and VID_IDS_TXT.exists():
            C = np.load(VID_CENTROIDS, mmap_mode="r")  # (n_videos, d)
            vids = [ln.strip() for ln in VID_IDS_TXT.read_text().splitlines() if ln.strip()]
            if len(vids) == C.shape[0]:
                return {"C": C, "vids": vids}
    except Exception:
        pass
    return None

@st.cache_resource(show_spinner=False)
def load_metas_list_and_counts() -> Tuple[List[dict], Dict[str, int]]:
    """Lightweight: read metas.pkl for alignment counts per video_id."""
    if not METAS_PKL.exists():
        return [], {}
    try:
        with METAS_PKL.open("rb") as f:
            payload = pickle.load(f)
        metas = payload.get("metas", [])
        counts: Dict[str, int] = {}
        for m in metas:
            vid = m.get("video_id") or m.get("vid") or m.get("ytid")
            if vid:
                counts[vid] = counts.get(vid, 0) + 1
        return metas, counts
    except Exception:
        return [], {}

# -----------------------------------------------------------------------------
# Random-access line reader (avoid loading the 500k+ chunks into RAM)
# -----------------------------------------------------------------------------
def read_lines_by_indices(indices: Iterable[int]) -> List[Optional[dict]]:
    """
    Given integer indices into the .jsonl, fetch the lines using OFFSETS_NPY.
    If offsets are missing, falls back to streaming the file once.
    """
    idxs = [i for i in indices if i is not None and i >= 0]
    if not idxs:
        return []

    # With offsets
    offs = load_offsets()
    if offs is not None and len(offs) > max(idxs):
        out: List[Optional[dict]] = [None] * len(idxs)
        with CHUNKS_PATH.open("rb") as f:
            for pos, i in enumerate(idxs):
                try:
                    f.seek(int(offs[i]))
                    line = f.readline()
                    out[pos] = _safe_json_loads(line.decode("utf-8", errors="ignore"))
                except Exception:
                    out[pos] = None
        return out

    # Fallback: stream until we collect needed rows
    need = set(idxs)
    got: Dict[int, dict] = {}
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i in need:
                got[i] = _safe_json_loads(ln)
            if len(got) == len(need):
                break
    # preserve order
    return [got.get(i) for i in idxs]

# -----------------------------------------------------------------------------
# Retrieval
# -----------------------------------------------------------------------------
def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

def mmr_rerank(query_vec: np.ndarray,
               cand_vecs: np.ndarray,
               cand_idxs: List[int],
               top_k: int,
               lambda_: float = 0.4) -> List[int]:
    """
    Maximal Marginal Relevance for diversity.
    Returns reordered indices (subset of cand_idxs) of size <= top_k.
    """
    if cand_vecs.shape[0] == 0:
        return []
    top_k = min(top_k, cand_vecs.shape[0])
    selected: List[int] = []
    selected_mask = np.zeros(cand_vecs.shape[0], dtype=bool)

    # Precompute similarities
    q = query_vec.reshape(1, -1)
    S_q = (cand_vecs @ q.T).ravel()  # similarity to query
    S_ij = cand_vecs @ cand_vecs.T   # pairwise similarity

    for _ in range(top_k):
        best = -1
        best_score = -1e9
        for i in range(cand_vecs.shape[0]):
            if selected_mask[i]:
                continue
            if not selected:
                score = S_q[i]
            else:
                div = np.max(S_ij[i, selected])
                score = lambda_ * S_q[i] - (1 - lambda_) * div
            if score > best_score:
                best_score = score
                best = i
        if best < 0:
            break
        selected.append(best)
        selected_mask[best] = True
    # Map back to original row indices
    return [cand_idxs[i] for i in selected]

def stageA_video_filter(query: str,
                        embedder: SentenceTransformer,
                        centroids: Dict[str, Any],
                        n_videos: int) -> Optional[set]:
    """
    Route to best videos first (optional). Returns a set of allowed video_ids.
    """
    if not centroids:
        return None
    C = centroids["C"]  # (n_videos, d)
    vids = centroids["vids"]
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    D, I = faiss.IndexFlatIP(C.shape[1]).search(_normalize(q), min(n_videos, C.shape[0]))
    take = set(vids[i] for i in I[0] if 0 <= i < len(vids))
    return take

def search_chunks(
    query: str,
    index: faiss.Index,
    embedder: SentenceTransformer,
    metas: List[dict],
    initial_k: int,
    final_k: int,
    per_video_cap: int,
    use_mmr: bool,
    mmr_lambda: float,
    allowed_videos: Optional[set] = None,
    recency_bias_months: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Efficient retrieval with optional video routing, MMR, per-video capping, and recency bias.
    Uses random-access reads to avoid loading the full corpus into memory.
    """
    if not query.strip():
        return []

    # Encode query
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")

    K = min(int(initial_k), index.ntotal)
    if K <= 0:
        return []

    # FAISS search
    D, I = index.search(qv, K)
    cand_idxs = [int(x) for x in I[0] if x >= 0]
    cand_sims = [float(x) for x in D[0][:len(cand_idxs)]]

    if not cand_idxs:
        return []

    # Optional Stage-A: filter by top-N videos using centroids
    if allowed_videos is not None:
        cand_idxs = [i for i in cand_idxs if (i < len(metas) and (metas[i].get("video_id") in allowed_videos))]
        cand_sims = cand_sims[:len(cand_idxs)]

    if not cand_idxs:
        return []

    # Pull lines for the candidate indices
    lines = read_lines_by_indices(cand_idxs)

    # Build candidate list with minimal memory
    cands: List[Dict[str, Any]] = []
    for local_pos, (idx, ln) in enumerate(zip(cand_idxs, lines)):
        if not ln:
            continue
        txt = (ln.get("text") or "").strip()
        m = (ln.get("meta") or {}).copy()
        vid = (
            m.get("video_id")
            or m.get("vid")
            or m.get("ytid")
            or ln.get("video_id")
            or ln.get("vid")
            or ln.get("ytid")
            or ln.get("id")
        )
        if not txt or not vid:
            continue
        # Normalize start
        if "start" not in m and "start_sec" in m:
            m["start"] = m["start_sec"]
        m["start"] = _parse_ts(m.get("start", 0))

        # Optional recency bias: small score bump for newer uploads if metadata carries 'upload_date' (YYYY-MM-DD)
        score = cand_sims[local_pos]
        up = (m.get("upload_date") or "").strip()
        if recency_bias_months and up and len(up) >= 7:
            # crude month delta based on string "YYYY-MM"
            try:
                y, mo = up[:4], up[5:7]
                months_since_1970 = (int(y) - 1970) * 12 + int(mo)
                score += (months_since_1970 % recency_bias_months) * 1e-4
            except Exception:
                pass

        cands.append({"i": idx, "score": float(score), "text": txt, "meta": {"video_id": vid, **m}})

    if not cands:
        return []

    # Optional MMR diversity (embed a small subset to re-rank)
    if use_mmr:
        # Re-embed top 200 texts locally for MMR re-rank (keeps CPU+RAM modest)
        topN = min(200, len(cands))
        sub_texts = [cands[i]["text"] for i in range(topN)]
        sub_vecs = embedder.encode(sub_texts, normalize_embeddings=True, batch_size=64).astype("float32")
        sub_idxs = [i for i in range(topN)]
        order = mmr_rerank(qv[0], sub_vecs, sub_idxs, top_k=min(final_k * 2, topN), lambda_=mmr_lambda)
        cands = [cands[i] for i in order] + cands[topN:]

    # Per-video cap, then trim to final_k
    counts: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    cap = max(1, int(per_video_cap))
    for h in cands:
        vid = h["meta"].get("video_id")
        counts[vid] = counts.get(vid, 0) + 1
        if counts[vid] <= cap:
            out.append(h)
        if len(out) >= int(final_k):
            break

    return out

# -----------------------------------------------------------------------------
# Answer synthesis (OpenAI)
# -----------------------------------------------------------------------------
def openai_answer(model_name: str,
                  question: str,
                  history: List[Dict[str, str]],
                  hits: List[Dict[str, Any]],
                  trusted_blurbs: Optional[List[str]] = None) -> str:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "⚠️ OPENAI_API_KEY is not set. Add it in your platform's env vars."

    # Keep last ~6 turns for conversational context
    recent = history[-6:]
    convo: List[str] = []
    for m in recent:
        role = m.get("role")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            label = "User" if role == "user" else "Assistant"
            convo.append(f"{label}: {content}")

    # Build evidence block (group by video for readability)
    by_vid: Dict[str, List[Dict[str, Any]]] = {}
    for h in hits:
        v = h["meta"].get("video_id", "unknown")
        by_vid.setdefault(v, []).append(h)

    lines: List[str] = []
    for v, chunks in by_vid.items():
        # one label per video
        lbl, _ = label_and_url(chunks[0]["meta"])
        lines.append(f"Video: {lbl}")
        for i, h in enumerate(chunks, 1):
            txt = h["text"].strip()
            lines.append(f"  [{i}] {txt}")
        lines.append("")

    # Optional external trusted blurbs appended after the video evidence
    ext = ""
    if trusted_blurbs:
        ext = "\nTrusted sources:\n" + "\n".join(f"- {x}" for x in trusted_blurbs) + "\n"

    system = (
        "You answer from the provided video excerpts and from trusted blurbs.\n"
        "Rules:\n"
        "• Do not invent facts not grounded in the excerpts/blurbs.\n"
        "• If evidence is insufficient or unclear or conflicted, say you don't know.\n"
        "• Be practical, concise, and detailed; suggest seeing a clinician when appropriate.\n"
        "• Keep continuity with the conversation, but ground all factual claims."
    )

    user_payload = (
        ("Recent conversation:\n" + "\n".join(convo) + "\n\n") if convo else ""
    ) + (
        f"Question: {question}\n\nEvidence:\n" + "\n".join(lines) + ext +
        "\nWrite the best possible answer consistent with the evidence. "
        "Cite ideas inline in prose by referring to the podcaster or source. "
        "If unsure, say you don't know."
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
st.title("Longevity / Nutrition Q&A")

with st.sidebar:
    st.header("Search settings")

    st.markdown(
        "Tune how the app **searches your video library** and builds answers."
    )

    initial_k = st.number_input(
        "Candidates to fetch (fast)",
        min_value=50, max_value=5000, value=480, step=50,
        help="How many similar snippets to pull from the index before re-ranking.\n"
             "Higher = more coverage but slower. 300–800 is a good range."
    )
    final_k = st.number_input(
        "Evidence to keep (balanced)",
        min_value=5, max_value=200, value=60, step=5,
        help="How many snippets max the answer can cite. More isn’t always better.\n"
             "40–80 works well. This capping also controls cost."
    )
    per_video_cap = st.number_input(
        "Max per video (diversity)",
        min_value=1, max_value=30, value=6, step=1,
        help="Upper limit of how many snippets can come from the same video.\n"
             "Prevents the answer from being dominated by one source."
    )

    st.subheader("Re-ranking")
    use_mmr = st.checkbox(
        "Use diversity boost (MMR)",
        value=True,
        help="Improves variety by trading off similarity vs. novelty.\n"
             "Keeps the answer from repeating near-duplicates."
    )
    mmr_lambda = st.slider(
        "MMR trade-off λ",
        min_value=0.1, max_value=0.9, value=0.4, step=0.05,
        help="λ closer to 1.0 favors similarity; closer to 0.1 favors diversity."
    )

    st.subheader("Video routing & recency")
    use_centroids = st.checkbox(
        "Prefer best-matching videos first",
        value=True,
        help="If precomputed, the app scores entire videos first, then searches inside the top few.\n"
             "Reduces noise and speeds up retrieval."
    )
    top_videos = st.number_input(
        "Videos to consider",
        min_value=4, max_value=128, value=28, step=4,
        help="How many videos to scan deeply after the first routing pass."
    )
    recency_bias = st.checkbox(
        "Slight recency bias",
        value=True,
        help="Gently prefers newer uploads when metadata includes dates."
    )

    st.subheader("Model")
    model_choice = st.selectbox(
        "OpenAI model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        index=0,
        help="Controls answer quality, speed, and cost."
    )
    st.session_state["model_choice"] = model_choice

    st.subheader("Pick sources")
    vm = load_video_meta()
    # Build channel counts from metas.pkl (fast)
    metas_list, vid_chunk_counts = load_metas_list_and_counts()
    channel_counts: Dict[str, int] = {}
    for vid, info in vm.items():
        ch = (info.get("channel") or "").strip() or "Unknown"
        channel_counts[ch] = channel_counts.get(ch, 0) + 1

    selectable_channels = sorted(channel_counts.keys())
    chosen_channels = st.multiselect(
        "Channels to include",
        options=selectable_channels,
        default=selectable_channels[: min(6, len(selectable_channels))],
        help="Only include these sources in the answer. Start broad, then narrow if needed."
    )

    # Optional admin: trigger precompute from the UI (safe; remove later if desired)
    st.divider()
    with st.expander("Admin tools", expanded=False):
        if st.button("Rebuild video summaries now"):
            try:
                import scripts.precompute_video_summaries as pcs
                pcs.main() if hasattr(pcs, "main") else pcs.precompute()  # best-effort
                st.success("Precompute finished.")
            except Exception as e:
                st.error(f"Precompute failed: {e}")

    st.divider()
    st.subheader("Library status")
    st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    st.checkbox("data/chunks/chunks.jsonl", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("data/index/faiss.index", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("data/index/metas.pkl", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("data/catalog/video_meta.json", value=VIDEO_META_JSON.exists(), disabled=True)

    peek = load_chunks_head(3)
    if peek:
        st.caption("Sample chunks:"); [st.code(p) for p in peek]

    c_info = load_centroids()
    st.write("Video centroids:", "ready ✅" if c_info else "not found")

# Chat history (kept short)
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role":"user"/"assistant", "content": str}

# Render history
for m in st.session_state.messages[-8:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Single input at bottom
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
            "Required files were not found in DATA_DIR. "
            "Copy them to your persistent volume and redeploy.\n\n"
            + "\n".join(f"- {DATA_ROOT / p}" for p in missing_bits)
        )
    st.stop()

# Load resources after we know files exist
try:
    index, metas_from_pkl, payload = load_index_and_model()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load FAISS index or embedder.")
        st.exception(e)
    st.stop()

if index is None or payload is None:
    with st.chat_message("assistant"):
        st.error("FAISS index / metas / model not found or failed to load.")
    st.stop()

embedder: SentenceTransformer = payload["embedder"]

# Resolve allowed videos by chosen channels + optional centroid routing
allowed_vids_by_channel: Optional[set] = None
if chosen_channels:
    vm = load_video_meta()
    allowed_vids_by_channel = {vid for vid, info in vm.items() if (info.get("channel") or "Unknown") in set(chosen_channels)}

centroids = load_centroids() if use_centroids else None
stageA_allow: Optional[set] = None
if centroids and use_centroids:
    try:
        allow_by_centroid = stageA_video_filter(prompt, embedder, centroids, n_videos=int(top_videos))
    except Exception:
        allow_by_centroid = None
    # intersect with channel filter if present
    if allow_by_centroid and allowed_vids_by_channel:
        stageA_allow = allow_by_centroid.intersection(allowed_vids_by_channel)
    else:
        stageA_allow = allow_by_centroid or allowed_vids_by_channel
else:
    stageA_allow = allowed_vids_by_channel

# Retrieval
with st.spinner("Searching sources…"):
    try:
        hits = search_chunks(
            prompt,
            index=index,
            embedder=embedder,
            metas=metas_from_pkl if metas_from_pkl else [],
            initial_k=int(initial_k),
            final_k=int(final_k),
            per_video_cap=int(per_video_cap),
            use_mmr=bool(use_mmr),
            mmr_lambda=float(mmr_lambda),
            allowed_videos=stageA_allow,
            recency_bias_months=12 if recency_bias else None,
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Search failed.")
            st.exception(e)
        st.stop()

# Answer
with st.chat_message("assistant"):
    if not hits:
        st.warning("No sources found for this answer. Widen filters or select more channels.")
        st.session_state.messages.append(
            {"role": "assistant", "content": "I couldn’t find relevant excerpts for that."}
        )
        st.stop()

    with st.spinner("Synthesizing answer…"):
        answer = openai_answer(
            st.session_state.get("model_choice", model_choice),
            prompt,
            st.session_state.messages,
            hits,
            trusted_blurbs=None,  # external sites are optional; keep off by default
        )

    st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # Grouped sources (nested by video)
    with st.expander("Sources & timestamps", expanded=False):
        # Group hits by video_id
        by_video: Dict[str, List[Dict[str, Any]]] = {}
        for h in hits:
            by_video.setdefault(h["meta"]["video_id"], []).append(h)

        for vid, chunks in by_video.items():
            lbl, url = label_and_url(chunks[0]["meta"])
            st.markdown(f"**{lbl}**" + (f" — [link]({url})" if url else ""))
            for i, h in enumerate(chunks, 1):
                start = h["meta"].get("start", 0)
                mm = int(start // 60); ss = int(start % 60)
                st.markdown(f"• {i}. {h['text']}  \n  <sub>t≈{mm:02d}:{ss:02d}</sub>")

    st.caption("Answers synthesize the indexed experts’ videos. If evidence is weak, I’ll say so.")
