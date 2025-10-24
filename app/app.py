# app/app.py
# Streamlit UI for “Longevity / Nutrition Q&A”
# - Uses FAISS + JSONL chunks in /var/data
# - If video centroids exist, does video-first narrowing with the SAME embedder
#   used to build the FAISS index. This fixes the dim mismatch assert.
# - Left panel is simplified and non-technical. All controls are safe.
# - Works without optional files; shows status clearly.

from __future__ import annotations

import os, json, pickle, math, time
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Optional
from collections import defaultdict, Counter

import numpy as np
import streamlit as st
import faiss

# ── 0) Page config must be FIRST Streamlit call ────────────────────────────────
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")

# ── 1) Constants & paths ───────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data")).resolve()
IDX_DIR  = DATA_DIR / "data" / "index"
CHUNK_FP = DATA_DIR / "data" / "chunks" / "chunks.jsonl"
META_FP  = DATA_DIR / "data" / "catalog" / "video_meta.json"  # optional catalog for titles/channels

# Optional precomputes
VID_IDS_FP   = IDX_DIR / "video_ids.txt"
VID_CENT_FP  = IDX_DIR / "video_centroids.npy"
VID_SUM_FP   = DATA_DIR / "data" / "catalog" / "video_summaries.json"  # optional lightweight title map

# ── 2) Cached loaders ──────────────────────────────────────────────────────────
@st.cache_resource
def load_metas_and_model_name():
    """Load metas.pkl to discover the FAISS embed model used for chunks."""
    mp = pickle.load(open(IDX_DIR / "metas.pkl", "rb"))
    model_name = mp.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    return mp, model_name

@st.cache_resource
def load_retr_embedder(model_name: str):
    """Load the SAME embedder used to build FAISS index (CPU, local cache first)."""
    from sentence_transformers import SentenceTransformer
    local_dir = DATA_DIR / "models" / model_name.split("/")[-1]
    for p in (str(local_dir), model_name):
        try:
            return SentenceTransformer(p, device="cpu")
        except Exception:
            pass
    raise RuntimeError(f"Could not load retrieval embedder: {model_name}")

@st.cache_resource
def load_faiss_index():
    """Load the FAISS index for chunk embeddings."""
    ix = faiss.read_index(str(IDX_DIR / "faiss.index"))
    return ix

@st.cache_resource
def load_centroid_index():
    """Load per-video centroid matrix and FAISS index over it. Optional."""
    if not (VID_IDS_FP.exists() and VID_CENT_FP.exists()):
        return None, None, None
    video_ids = VID_IDS_FP.read_text().strip().splitlines()
    C = np.load(VID_CENT_FP).astype("float32")  # [Nvideos, d]
    d = C.shape[1]
    faiss.normalize_L2(C)                       # cosine via inner product
    idx = faiss.IndexFlatIP(d)
    idx.add(C)
    return video_ids, C, idx

@st.cache_resource
def load_title_map() -> Dict[str, Dict[str,str]]:
    """
    Returns: {video_id: {"title":..., "channel":...}}
    Accepts video_summaries.json (optional) or falls back to video_meta.json.
    """
    if VID_SUM_FP.exists():
        try:
            return json.loads(VID_SUM_FP.read_text())
        except Exception:
            pass
    if META_FP.exists():
        try:
            meta = json.loads(META_FP.read_text())
            out = {}
            for v in meta:
                vid = str(v.get("video_id") or v.get("id") or "")
                if not vid: 
                    continue
                out[vid] = {"title": v.get("title",""), "channel": v.get("channel","")}
            return out
        except Exception:
            pass
    return {}

def _mk_norm(text: str) -> str:
    return (text or "").strip().lower()

# ── 3) Chunk iterator (zero-copy streaming) ────────────────────────────────────
def iter_chunks(jsonl_path: Path) -> Iterable[Dict]:
    """Stream chunks.jsonl safely and cheaply."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for ln in f:
            try:
                j = json.loads(ln)
                t = (j.get("text") or "").strip()
                m = j.get("meta") or {}
                if not t:
                    continue
                # unify id fields
                vid = (
                    m.get("video_id") or j.get("video_id") or m.get("vid") or
                    m.get("ytid") or j.get("vid") or j.get("ytid") or m.get("id") or j.get("id")
                )
                start = m.get("start") or m.get("start_sec") or m.get("ts") or 0
                yield {"text": t, "video_id": str(vid or ""), "start": float(start), "meta": m}
            except Exception:
                continue

# ── 4) Retrieval helpers (video-first then chunk) ─────────────────────────────
mp, RETR_MODEL_NAME = load_metas_and_model_name()
retr_embedder = load_retr_embedder(RETR_MODEL_NAME)
ix = load_faiss_index()
VIDEO_IDS, CENTROIDS, CENTROID_FAISS = load_centroid_index()
TITLE_MAP = load_title_map()

@st.cache_data(show_spinner=False)
def _list_all_channels_and_videos() -> Tuple[List[str], Dict[str,List[Tuple[str,str]]]]:
    """
    Returns:
      channels: sorted unique channel names
      by_channel: {channel: [(video_id, title), ...]}
    """
    by_channel = defaultdict(list)
    # Prefer catalog; if missing, scan chunks.jsonl headers lazily
    if TITLE_MAP:
        for vid, rec in TITLE_MAP.items():
            ch = rec.get("channel","") or "Unknown"
            by_channel[ch].append((vid, rec.get("title","")))
    else:
        seen = set()
        for row in iter_chunks(CHUNK_FP):
            vid = row["video_id"]
            if not vid or vid in seen: 
                continue
            seen.add(vid)
            by_channel["Unknown"].append((vid, ""))
    channels = sorted(by_channel.keys())
    for ch in channels:
        by_channel[ch].sort(key=lambda x: _mk_norm(x[1]))
    return channels, by_channel

def top_videos_by_centroid(query_text: str, k: int) -> List[str]:
    """Use SAME embedder dim as FAISS to avoid asserts. Falls back to []."""
    if CENTROID_FAISS is None or retr_embedder is None:
        return []
    qv = retr_embedder.encode([query_text], normalize_embeddings=True)
    qv = np.asarray(qv, dtype="float32")      # [1, d] matches centroids dim
    faiss.normalize_L2(qv)
    k = max(1, min(k, CENTROID_FAISS.ntotal))
    D, I = CENTROID_FAISS.search(qv, k)
    return [VIDEO_IDS[j] for j in I[0] if j >= 0]

@st.cache_data(show_spinner=False)
def _vid_to_rowids() -> Dict[str, List[int]]:
    """Map video_id -> list of chunk row indices (for quick filtering)."""
    out = defaultdict(list)
    # metas.pkl contains "metas": per-chunk meta; we only need video_id location map
    metas = mp.get("metas", [])
    for i, m in enumerate(metas):
        vid = str(m.get("video_id") or m.get("vid") or m.get("id") or "")
        if vid:
            out[vid].append(i)
    return out

VID2ROWIDS = _vid_to_rowids()

def _search_chunks_in_videos(query_text: str, candidate_vids: List[str], k_chunks: int) -> List[Tuple[int, float]]:
    """
    Search within the union of rows for candidate videos, return top k (row_id, score).
    """
    # Build a tiny FAISS index view: use the global index, but restrict ids via IVF? We’ll fetch via subset.
    # Simple approach: search globally for more than needed, then filter by candidate vids.
    qv = retr_embedder.encode([query_text], normalize_embeddings=True).astype("float32")
    faiss.normalize_L2(qv)
    # Ask for 10x more then filter
    ask = min(max(k_chunks*10, k_chunks+32), ix.ntotal)
    D, I = ix.search(qv, ask)
    good = []
    cand = set(candidate_vids)
    metas = mp.get("metas", [])
    for d, i in zip(D[0], I[0]):
        if i < 0:
            continue
        vid = str(metas[i].get("video_id") or metas[i].get("vid") or metas[i].get("id") or "")
        if not cand or (vid in cand):
            good.append((i, float(d)))
            if len(good) >= k_chunks:
                break
    return good

def _search_chunks_global(query_text: str, k_chunks: int) -> List[Tuple[int, float]]:
    qv = retr_embedder.encode([query_text], normalize_embeddings=True).astype("float32")
    faiss.normalize_L2(qv)
    ask = min(k_chunks, ix.ntotal)
    D, I = ix.search(qv, ask)
    return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]

# ── 5) UI: Left panel (simple, descriptive) ───────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    st.caption(
        "These options change *how much* the app searches and how much evidence is shown. "
        "They are safe; reset to defaults if you see errors."
    )
    videos_to_consider = st.number_input(
        "Videos to consider before chunk search",
        min_value=1, max_value=100, value=8, step=1,
        help="When available, the app first finds a few likely videos using fast similarity. "
             "Then it searches chunks in those videos for precise answers."
    )
    candidate_chunks = st.number_input(
        "Candidate chunks searched",
        min_value=16, max_value=512, value=96, step=8,
        help="Upper bound on chunks evaluated for the answer."
    )
    evidence_kept = st.number_input(
        "Evidence chunks kept",
        min_value=1, max_value=50, value=30, step=1,
        help="How many chunks are returned as evidence for the final answer."
    )
    max_chunks_per_video = st.number_input(
        "Max chunks per video",
        min_value=1, max_value=10, value=3, step=1,
        help="Limit chunks from any single video to keep the answer balanced."
    )

    # Source picking (channels and optional specific videos)
    st.markdown("### Pick sources to include")
    channels, by_channel = _list_all_channels_and_videos()
    default_channels = channels  # include all by default
    chosen_channels = st.multiselect(
        "Channels to include",
        channels, default_channels,
        help="Only use videos from these channels."
    )

    # Conditionally allow picking specific videos
    all_vid_options = []
    for ch in chosen_channels:
        all_vid_options.extend([(f"{title or 'Untitled'} — {ch}", vid) for vid, title in by_channel.get(ch, [])])
    st.caption("Optional: pick specific videos (otherwise all from chosen channels).")
    chosen_video_label = st.selectbox(
        "Choose a video (optional)", ["<All videos>"] + [lbl for (lbl, _vid) in all_vid_options],
        index=0
    )
    chosen_video_ids = set()
    if chosen_video_label != "<All videos>":
        # map label -> id
        lbl2id = {lbl: vid for (lbl, vid) in all_vid_options}
        chosen_video_ids = {lbl2id[chosen_video_label]}

    # Trusted websites toggle (stubs for future live fetching)
    st.markdown("### Trusted websites")
    use_web = st.checkbox("Add short excerpts from trusted health sites", value=False)
    trusted_sites = st.multiselect(
        "Sites to include",
        ["nih.gov","medlineplus.gov","cdc.gov","mayoclinic.org","health.harvard.edu",
         "familydoctor.org","healthfinder.gov","nejm.org","stanford.edu","mountsinai.org",
         "medicalxpress.com","sciencedaily.com/health-medicine"],
        default=[]
    )

    # Data status
    st.markdown("### Data status (under DATA_DIR)")
    core_ok = CHUNK_FP.exists() and (IDX_DIR / "faiss.index").exists() and (IDX_DIR / "metas.pkl").exists()
    if core_ok:
        st.success("Core files found.")
    else:
        st.error("Required files are missing. Check your volume mount and bootstrap logs.")

    if VID_CENT_FP.exists() and VID_IDS_FP.exists():
        st.info("Per-video centroids found — video-first search is enabled.")
    else:
        st.warning("Video centroids not found. The app will search chunks directly (still works).")

    st.caption("Debug peek: first 3 expected paths")
    for p in [CHUNK_FP, IDX_DIR / "faiss.index", IDX_DIR / "metas.pkl"]:
        st.code(str(p), language="bash")

# ── 6) Main area ───────────────────────────────────────────────────────────────
st.markdown("## Longevity / Nutrition Q&A")
query = st.text_input(
    "Ask about sleep, protein timing, fasting, supplements, protocols…",
    help="Ask one clear question. The app returns a concise answer with citations."
)

if query and core_ok:
    try:
        t0 = time.time()

        # Source restriction set
        allowed_vids = set()
        for ch in chosen_channels:
            for vid, _title in by_channel.get(ch, []):
                allowed_vids.add(vid)
        if chosen_video_ids:
            allowed_vids &= chosen_video_ids

        # 1) Video-first narrowing if centroids exist
        top_vids = []
        if CENTROID_FAISS is not None:
            vids_k = min(videos_to_consider, CENTROID_FAISS.ntotal)
            top_vids = top_videos_by_centroid(query, vids_k)
            if allowed_vids:
                top_vids = [v for v in top_vids if v in allowed_vids]
            if not top_vids and allowed_vids:
                # If filter removed all, fall back to allowed set head
                top_vids = list(allowed_vids)[:videos_to_consider]

        # 2) Chunk search
        if top_vids:
            hits = _search_chunks_in_videos(query, top_vids, candidate_chunks)
        else:
            # direct global search then filter by allowed_vids if any
            raw = _search_chunks_global(query, candidate_chunks)
            if allowed_vids:
                metas = mp.get("metas", [])
                hits = [(i, s) for (i, s) in raw
                        if str(metas[i].get("video_id") or metas[i].get("vid") or metas[i].get("id") or "") in allowed_vids]
                if not hits:  # if filtering wipes everything, use raw
                    hits = raw
            else:
                hits = raw

        # 3) Gather evidence chunks while limiting per-video
        metas = mp.get("metas", [])
        per_vid = Counter()
        chosen_rows: List[int] = []
        for i, _score in hits:
            vid = str(metas[i].get("video_id") or metas[i].get("vid") or metas[i].get("id") or "")
            if per_vid[vid] >= max_chunks_per_video:
                continue
            per_vid[vid] += 1
            chosen_rows.append(i)
            if len(chosen_rows) >= evidence_kept:
                break

        # 4) Fetch text for chosen rows by streaming JSONL once
        want = set(chosen_rows)
        row2text: Dict[int,str] = {}
        row2meta: Dict[int,Dict] = {}
        row_idx = -1
        for row_idx, j in enumerate(iter_chunks(CHUNK_FP)):
            if row_idx in want:
                row2text[row_idx] = j["text"]
                row2meta[row_idx] = {"video_id": j["video_id"], "start": j["start"]}
            if len(row2text) == len(want):
                break

        # 5) Build simple final answer (LLM call left as stub)
        st.markdown("### Suggested plan")
        st.write(
            "This section would call your preferred LLM with the selected evidence to produce a concise, "
            "actionable answer. For now, it lists the chosen evidence snippets."
        )

        # Group evidence by video and show nested
        st.markdown("### Sources")
        by_vid = defaultdict(list)
        for rid in chosen_rows:
            md = row2meta.get(rid, {})
            by_vid[md.get("video_id","")].append((rid, row2text.get(rid,""), md.get("start",0.0)))

        for vid, items in by_vid.items():
            rec = TITLE_MAP.get(vid, {})
            ch  = rec.get("channel","")
            title = rec.get("title","(no title)")
            with st.expander(f"{title} — {ch}  •  {len(items)} chunk(s)"):
                for _rid, txt, start in items:
                    st.caption(f"t={int(start)}s")
                    st.write(txt)

        st.caption(f"Search completed in {time.time()-t0:.2f}s")

    except Exception as e:
        st.error(f"Unexpected error: {e}")

# ── 7) Minimal “How it works” help (kept concise, non-technical) ───────────────
with st.expander("How it works", expanded=False):
    st.write(
        """
        - The app stores data in a volume at `/var/data`.
        - It loads a FAISS index of chunk embeddings and searches efficiently.
        - If per-video “centroids” exist, it first picks likely videos, then searches inside them.
        - It never loads all text into memory. It streams chunk texts directly from file.
        """
    )
