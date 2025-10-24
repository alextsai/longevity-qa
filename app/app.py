"""
Longevity / Nutrition Q&A — Streamlit app
-----------------------------------------

WHAT THIS FILE DOES
- Defines the UI and retrieval workflow.
- Loads your FAISS + metas + chunks under DATA_DIR (default /var/data).
- If precomputed per-video centroids exist (produced at boot by scripts/bootstrap_data.py),
  uses them to pre-filter the candidate videos (fast + low memory).
- Groups sources by video in the “Sources” panel.
- Fixes two issues you saw:
    1) set_page_config is now the *first* Streamlit call (Streamlit requirement).
    2) Avoids printing list-of-DeltaGenerators by not using list-comprehensions for st.code.

You can read the big “HOW IT WORKS” block near the top for a quick mental model.
"""

# ---------------------------------------------------------------------------
# 0) STREAMLIT MUST BE CONFIGURED BEFORE ANY WIDGETS ARE CREATED
# ---------------------------------------------------------------------------
import streamlit as st

st.set_page_config(
    page_title="Longevity / Nutrition Q&A",
    page_icon="🍎",
    layout="wide"
)

# ---------------------------------------------------------------------------
# 1) IMPORTS AND GLOBAL SWITCHES
# ---------------------------------------------------------------------------
# Silence the noisy torch.classes warning you saw in the logs.
import warnings
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

import os
import json
import time
import math
import pickle
import itertools
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import faiss  # CPU FAISS (installed by requirements.txt)
from sentence_transformers import SentenceTransformer

# If you use OpenAI for synthesis, import it here.
# (leave it optional; retrieval-only still works if no key is present)
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# 2) CONFIG / DATA LAYOUT
# ---------------------------------------------------------------------------
"""
HOW IT WORKS

DATA_DIR (default /var/data) is a volume. Inside it we expect:

/data/index/faiss.index          — FAISS index for all chunk embeddings
/data/index/metas.pkl            — dict with "metas": list per-chunk metadata
/data/chunks/chunks.jsonl        — 1 JSON object per line: {"text": ..., "meta": {...}}
/data/catalog/video_meta.json    — catalog of videos (id, title, channel, etc)

OPTIONAL (auto-built at boot by scripts/bootstrap_data.py):
/data/index/video_centroids.npy  — float32 matrix [Nvideos, d] = mean embedding per video
/data/index/video_ids.txt        — text file with one video_id per line
/data/catalog/video_summaries.json — lightweight title map for display (optional)

This app will:
- Load the FAISS index + metas + chunks lazily.
- If the video centroids exist, use them to preselect a small set of **videos** for
  deeper chunk search (this is the “max depth first, then breadth” behavior you wanted).
- Keep memory low by never loading all chunk texts at once (stream iterating chunks.jsonl).
"""

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data")).resolve()
INDEX_DIR = DATA_DIR / "data" / "index"
CHUNKS_DIR = DATA_DIR / "data" / "chunks"
CATALOG_DIR = DATA_DIR / "data" / "catalog"

FAISS_PATH  = INDEX_DIR / "faiss.index"
METAS_PATH  = INDEX_DIR / "metas.pkl"
CHUNKS_PATH = CHUNKS_DIR / "chunks.jsonl"
CATALOG_PATH = CATALOG_DIR / "video_meta.json"

# Optional, used when present
VID_CENTROIDS_PATH = INDEX_DIR / "video_centroids.npy"
VID_IDS_PATH       = INDEX_DIR / "video_ids.txt"
VID_SUMMARIES_PATH = CATALOG_DIR / "video_summaries.json"

# Model name comes from the index metadata. Fallback to MiniLM if missing.
DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

# Reusable cache for heavy objects
@st.cache_resource(show_spinner=False)
def _load_index_and_metas():
    ix = faiss.read_index(str(FAISS_PATH))
    mp = pickle.load(open(METAS_PATH, "rb"))
    model_name = mp.get("model", DEFAULT_ENCODER)
    return ix, mp, model_name

@st.cache_resource(show_spinner=False)
def _load_encoder(model_name: str):
    # Allow the model to resolve from local cache under /var/data/models as well
    # (bootstrap_data may have snapshot_download'ed it there).
    local_cache = str(DATA_DIR / "models")
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", local_cache)
    os.environ.setdefault("HF_HOME", local_cache)
    return SentenceTransformer(model_name, device="cpu")

@st.cache_resource(show_spinner=False)
def _load_video_centroids():
    if VID_CENTROIDS_PATH.exists() and VID_IDS_PATH.exists():
        mat = np.load(VID_CENTROIDS_PATH)
        ids = [ln.strip() for ln in open(VID_IDS_PATH, encoding="utf-8")]
        return mat.astype("float32", copy=False), ids
    return None, None

@st.cache_resource(show_spinner=False)
def _load_catalog():
    if CATALOG_PATH.exists():
        try:
            return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

@st.cache_resource(show_spinner=False)
def _load_titles_map():
    """
    Returns {video_id: title}. Handles both list-of-dicts and dict-of-dicts formats.
    """
    # Prefer precomputed summaries
    if VID_SUMMARIES_PATH.exists():
        try:
            d = json.loads(VID_SUMMARIES_PATH.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                # dictionary of {id: {title: ...}} OR {id: title}
                out = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        out[k] = v.get("title", "")
                    else:
                        out[k] = str(v)
                return out
            elif isinstance(d, list):
                # list of {id:..., title:...}
                return {str(x.get("video_id") or x.get("id") or ""): x.get("title", "") for x in d}
        except Exception:
            pass

    # Fallback to catalog (list or dict)
    cat = _load_catalog()
    m = {}
    if isinstance(cat, dict):
        for k, v in cat.items():
            if isinstance(v, dict):
                m[str(k)] = v.get("title", "")
            else:
                m[str(k)] = str(v)
    elif isinstance(cat, list):
        for row in cat:
            if not isinstance(row, dict):
                continue
            vid = str(row.get("video_id") or row.get("id") or "")
            if vid:
                m[vid] = row.get("title") or ""
    return m

# Minimal, fast JSONL streaming to avoid loading all chunks at once
def iter_chunks_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for ln in f:
            try:
                yield json.loads(ln)
            except Exception:
                continue

# ---------------------------------------------------------------------------
# 3) UTILS — SEARCH
# ---------------------------------------------------------------------------
def encode_query(q: str, enc: SentenceTransformer) -> np.ndarray:
    v = enc.encode([q], normalize_embeddings=True).astype("float32")
    return v

def top_videos_by_centroid(query_v: np.ndarray, top_v: int = 8) -> List[str]:
    """
    If video-level centroids are present, use them to select the most relevant videos first.
    Returns a list of video_ids (strings).
    """
    mat, ids = _load_video_centroids()
    if mat is None:
        return []  # centroids missing
    # cosine ~ inner product because embeddings are normalized
    D, I = faiss.IndexFlatIP(mat.shape[1]).search(query_v @ mat.T, top_v)  # fast but OK
    picked = []
    for i in I[0]:
        if 0 <= i < len(ids):
            picked.append(ids[i])
    return picked

def search_chunks_in_index(query_v: np.ndarray, ix: faiss.Index, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    FAISS search at chunk level.
    Returns (scores, indices) arrays.
    """
    D, I = ix.search(query_v, top_k)
    return D[0], I[0]

def filter_hits_to_videos(indices: np.ndarray,
                          desired_videos: List[str],
                          metas: List[dict],
                          keep: int) -> List[int]:
    """
    Keep only hits whose video_id ∈ desired_videos. Preserve order.
    Stop after 'keep'.
    """
    desired = set(desired_videos)
    out = []
    for idx in indices:
        if idx < 0:  # FAISS padding
            continue
        m = metas[idx] if 0 <= idx < len(metas) else None
        vid = None
        if m:
            vid = m.get("video_id") or m.get("vid") or m.get("ytid") or m.get("id")
        if vid and vid in desired:
            out.append(int(idx))
        if len(out) >= keep:
            break
    return out

def gather_chunks_text(indices: List[int], max_per_video: int = 3) -> Dict[str, List[Tuple[int, str]]]:
    """
    Read chunks.jsonl ONCE and take the exact chunk texts we need. Group by video_id.
    Returns dict: {video_id: [(chunk_index, text), ...]}
    """
    want = set(indices)
    grouped: Dict[str, List[Tuple[int, str]]] = {}
    # one pass over JSONL with a running index
    i = 0
    for row in iter_chunks_jsonl(CHUNKS_PATH):
        if i in want:
            meta = row.get("meta", {})
            vid = meta.get("video_id") or meta.get("vid") or meta.get("ytid") or row.get("video_id") or ""
            if vid:
                grouped.setdefault(vid, [])
                if len(grouped[vid]) < max_per_video:
                    grouped[vid].append((i, (row.get("text") or "").strip()))
        i += 1
    return grouped

def synthesize_answer(query: str,
                      grouped_chunks: Dict[str, List[Tuple[int, str]]],
                      titles_map: Dict[str, str],
                      model: str = "gpt-4o-mini") -> str:
    """
    Very small answer synthesizer. If OPENAI api key not present, returns a retrieval-only
    stitched summary. If OPENAI is available, we prompt it with 6–10 short evidence snippets.
    """
    # build short evidence set
    snippets = []
    for vid, pairs in grouped_chunks.items():
        for _, text in pairs:
            if not text:
                continue
            snippets.append(text[:400])
    snippets = snippets[:10]

    if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        # fallback: simple stitched summary
        if not snippets:
            return "No relevant evidence found in the selected sources."
        joined = " • ".join(snippets)
        return f"Evidence-based notes (retrieval-only): {joined}"

    openai.api_key = os.getenv("OPENAI_API_KEY")
    sys_prompt = (
        "You are a careful health assistant. Answer the user strictly from the provided evidence. "
        "Write short, clear steps and call out safety flags. Do not invent citations."
    )
    user_msg = f"QUESTION:\n{query}\n\nEVIDENCE SNIPPETS:\n" + "\n\n".join(f"- {s}" for s in snippets)

    # OpenAI Chat Completions API (compatible with gpt-4o-mini)
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# 4) SIDEBAR — SETTINGS + DATA STATUS
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    st.caption(
        "These controls change **how much we search** and **how much evidence** is sent to the AI. "
        "Defaults are safe. If the app restarts or shows errors, reset to defaults."
    )

    # Main knobs
    videos_to_consider = st.number_input(
        "Videos to consider before chunk search",
        min_value=1, max_value=50, value=8, step=1,
        help="When video centroids are available, we first rank videos by similarity to your question and only search chunks within the top N videos."
    )
    candidate_chunks = st.number_input(
        "Candidate chunks searched",
        min_value=16, max_value=512, value=96, step=16,
        help="How many chunk candidates to pull from the FAISS index before filtering to videos."
    )
    evidence_kept = st.number_input(
        "Evidence chunks kept",
        min_value=3, max_value=60, value=30, step=1,
        help="How many chunk texts to send into the answer step. Fewer = faster, More = potentially richer."
    )
    max_chunks_per_video = st.number_input(
        "Max chunks per video",
        min_value=1, max_value=10, value=3, step=1,
        help="Cap per-source so one video cannot dominate the answer."
    )

    st.divider()
    st.subheader("Data status (under DATA_DIR)")
    base = DATA_DIR
    need = [CHUNKS_PATH, FAISS_PATH, METAS_PATH, CATALOG_PATH]
    missing = [p for p in need if not p.exists()]

    # Clear, friendly status
    if missing:
        st.error("Required files are missing under DATA_DIR:")
        for p in missing:
            st.code(str(p))
    else:
        st.success("Core files found.")

    # Optional files
    have_centroids = VID_CENTROIDS_PATH.exists() and VID_IDS_PATH.exists()
    if have_centroids:
        st.info("Per-video centroids found — video-first search is enabled.")
    else:
        st.warning("Video centroids not found. The app will search chunks directly (still works).")

    # Small debug peek (fixed: no more DeltaGenerator dump)
    st.caption("Debug peek: first 3 expected paths")
    peek_paths = [str(p) for p in need[:3]]
    for p in peek_paths:
        st.code(p)


# ---------------------------------------------------------------------------
# 5) MAIN — APP BODY
# ---------------------------------------------------------------------------
st.title("Longevity / Nutrition Q&A")

# If core files are missing, do not proceed.
if missing:
    st.stop()

# Load heavy artifacts once
ix, metas_map, model_name = _load_index_and_metas()
metas: List[dict] = metas_map.get("metas", [])
enc = _load_encoder(model_name)
titles_map = _load_titles_map()

# Input
question = st.chat_input(
    placeholder="Ask about sleep, protein timing, fasting, supplements, protocols…"
)

# When there is no question yet, show a friendly empty state
if not question:
    st.write("Ask a question to see evidence-based suggestions.")
    st.stop()

# ---------------------------------------------------------------------------
# 6) RETRIEVAL PIPELINE
# ---------------------------------------------------------------------------
t0 = time.time()

# 6.1 encode the query
qv = encode_query(question, enc)  # shape (1, d)

# 6.2 video pre-selection (if centroids exist)
top_videos = top_videos_by_centroid(qv, top_v=videos_to_consider)
use_video_filter = len(top_videos) > 0

# 6.3 search the chunk-level index
scores, idxs = search_chunks_in_index(qv, ix, candidate_chunks)

# 6.4 if we have preselected videos, filter hits to those videos; else take top-K directly
if use_video_filter:
    kept = filter_hits_to_videos(idxs, top_videos, metas, keep=evidence_kept * 3)
    # fall back if filtering yields too few
    if len(kept) < evidence_kept:
        kept = [int(i) for i in idxs if i >= 0][:evidence_kept * 2]
else:
    kept = [int(i) for i in idxs if i >= 0][:evidence_kept * 2]

# 6.5 read those chunk texts from disk and group by video
grouped_chunks = gather_chunks_text(kept, max_per_video=max_chunks_per_video)

# 6.6 finally trim to the requested evidence_kept total
flat_pairs = []
for vid, pairs in grouped_chunks.items():
    for p in pairs:
        flat_pairs.append((vid, p))
flat_pairs = flat_pairs[:evidence_kept]

# rebuild grouped dict after trimming
grouped_final: Dict[str, List[Tuple[int, str]]] = {}
for vid, (idx, txt) in flat_pairs:
    grouped_final.setdefault(vid, []).append((idx, txt))

retrieval_ms = int((time.time() - t0) * 1000)

# ---------------------------------------------------------------------------
# 7) ANSWER
# ---------------------------------------------------------------------------
t1 = time.time()
answer = synthesize_answer(question, grouped_final, titles_map)
gen_ms = int((time.time() - t1) * 1000)

# Show the AI answer
st.subheader("Answer")
st.write(answer)

# Small timing note (useful when tuning)
st.caption(f"Retrieval: {retrieval_ms} ms, Synthesis: {gen_ms} ms")

# ---------------------------------------------------------------------------
# 8) SOURCES — GROUPED BY VIDEO
# ---------------------------------------------------------------------------
st.subheader("Sources")
if not grouped_final:
    st.write("No sources were selected for this answer.")
else:
    # group display per video with title, then its selected chunks
    for vid, pairs in grouped_final.items():
        title = titles_map.get(str(vid), f"Video {vid}")
        with st.expander(f"{title}  ·  id={vid}  ·  {len(pairs)} selected chunk(s)", expanded=False):
            for idx, txt in pairs:
                st.markdown(f"**Chunk #{idx}**")
                st.code(txt or "∅")

# End of file
