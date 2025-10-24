# app/app.py
# -*- coding: utf-8 -*-
"""
Longevity / Nutrition Q&A — Experts-first RAG with trusted web support

What changed
- Stronger synthesis prompt geared for clinical usefulness.
- Better defaults for richer retrieval: route 15 videos, scan 512 passages, keep 30.
- Vertical checkbox lists for Experts and Trusted sites; section help only.
- Experts show video counts; excluded creators hidden globally.
- Clear Chat callback is reliable.
- Two-stage retrieval with recency + MMR.
- Web snippets as support; optional web-only fallback via WEB_FALLBACK.
- Admin diagnostics (DEV_MODE=1 or ?admin=1[&key=...]).
- Precompute freshness and dimensionality checks.
- De-dup repeated quotes in prompt and “Sources & timestamps”.
"""

from __future__ import annotations

# -------- Quiet defaults --------
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# -------- Imports --------
from pathlib import Path
import sys, json, pickle, time, re
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Optional web deps
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None; BeautifulSoup = None

# -------- Paths --------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(os.getenv("DATA_DIR", "/var/data")).resolve()

CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"
OFFSETS_NPY     = DATA_ROOT / "data/chunks/chunks.offsets.npy"
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"

# Precompute outputs
VID_CENT_NPY    = DATA_ROOT / "data/index/video_centroids.npy"
VID_IDS_TXT     = DATA_ROOT / "data/index/video_ids.txt"
VID_SUM_JSON    = DATA_ROOT / "data/catalog/video_summaries.json"

REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]

# -------- Flags --------
WEB_FALLBACK = os.getenv("WEB_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}

# -------- Trusted domains --------
TRUSTED_DOMAINS = [
    "nih.gov","medlineplus.gov","cdc.gov","mayoclinic.org","health.harvard.edu",
    "familydoctor.org","healthfinder.gov","ama-assn.org","medicalxpress.com",
    "sciencedaily.com","nejm.org","med.stanford.edu","icahn.mssm.edu"
]

# -------- Creators excluded from UI/routing --------
EXCLUDED_CREATORS = {
    "Dr. Pradip Jamnadas, MD",
    "The Primal Podcast",
    "The Diary of a CEO",
    "Louis Tomlinson",
    "NA",
}

# -------- Utils --------
def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _parse_ts(v) -> float:
    if isinstance(v, (int, float)):
        try: return float(v)
        except: return 0.0
    try:
        sec = 0.0
        for p in str(v).split(":"):
            sec = sec * 60 + float(p)
        return sec
    except: return 0.0

def _iso_to_epoch(iso: str) -> float:
    if not iso: return 0.0
    try:
        if "T" in iso: return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
        return datetime.fromisoformat(iso).timestamp()
    except: return 0.0

def _format_ts(sec: float) -> str:
    sec = int(max(0, float(sec))); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def _file_mtime(p: Path) -> float:
    try: return p.stat().st_mtime
    except: return 0.0

def _clear_chat():
    st.session_state["messages"] = []
    st.rerun()

# -------- Admin gate --------
def _is_admin() -> bool:
    if os.getenv("DEV_MODE", "0") == "1":
        return True
    try:
        qp = st.experimental_get_query_params()
    except Exception:
        qp = {}
    if qp.get("admin", ["0"])[0] != "1":
        return False
    expected = None
    try:
        expected = st.secrets["ADMIN_KEY"]
    except Exception:
        expected = None
    if expected is None:
        return True
    supplied = qp.get("key", [""])[0]
    return supplied == str(expected)

# -------- Diagnostics (admin) --------
def precompute_status(embedder_model: str) -> Dict[str, Any]:
    status = {
        "centroids_present": VID_CENT_NPY.exists(),
        "ids_present": VID_IDS_TXT.exists(),
        "summaries_present": VID_SUM_JSON.exists(),
        "chunks_mtime": _file_mtime(CHUNKS_PATH),
        "cent_mtime": _file_mtime(VID_CENT_NPY),
        "ids_mtime": _file_mtime(VID_IDS_TXT),
        "ok_shapes": None, "ok_norms": None, "ok_dim": None, "msg": []
    }
    try:
        if status["centroids_present"] and status["ids_present"]:
            C = np.load(VID_CENT_NPY)
            vids = VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
            status["ok_shapes"] = (C.shape[0] == len(vids) and C.ndim == 2)
            n = np.linalg.norm(C, axis=1) if C.ndim == 2 else np.array([])
            status["ok_norms"] = bool(len(n) and n.min() > 0.90 and n.max() < 1.10)
            try:
                enc_dim = _load_embedder(embedder_model).get_sentence_embedding_dimension()
                status["ok_dim"] = (C.shape[1] == enc_dim)
            except Exception:
                status["ok_dim"] = False
            if status["chunks_mtime"] > max(status["cent_mtime"], status["ids_mtime"]):
                status["msg"].append("chunks.jsonl is newer than centroids → re-run precompute.")
        else:
            status["msg"].append("Centroids or IDs missing → run precompute.")
    except Exception as e:
        status["msg"].append(f"Precompute check error: {e}")
    return status

def scan_chunks_for_terms(terms: List[str], vm: Dict[str, Dict[str, Any]], limit_examples: int = 200):
    if not CHUNKS_PATH.exists():
        return {"total_matches": 0, "per_creator": {}, "examples": []}
    pat = re.compile(r"(" + "|".join([re.escape(t) for t in terms]) + r")", re.IGNORECASE)
    total = 0; per_creator = {}; examples = []
    try:
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            for ln in f:
                try: j = json.loads(ln)
                except: continue
                t = j.get("text", "") or ""
                if not pat.search(t): continue
                m = (j.get("meta") or {})
                vid = m.get("video_id") or m.get("vid") or m.get("ytid") or j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id") or "Unknown"
                st_sec = _parse_ts(m.get("start", m.get("start_sec", 0)))
                info = vm.get(vid, {})
                creator = info.get("podcaster") or info.get("channel") or "Unknown"
                per_creator[creator] = per_creator.get(creator, 0) + 1
                total += 1
                if len(examples) < int(limit_examples):
                    snippet = _normalize_text(t)
                    if len(snippet) > 260: snippet = snippet[:260] + "…"
                    examples.append({"video_id": vid, "creator": creator, "ts": _format_ts(st_sec), "snippet": snippet})
    except Exception as e:
        examples.append({"video_id":"", "creator":"", "ts":"", "snippet": f"Scan error: {e}"})
    per_creator_sorted = dict(sorted(per_creator.items(), key=lambda kv: -kv[1]))
    return {"total_matches": total, "per_creator": per_creator_sorted, "examples": examples}

# -------- Video metadata --------
@st.cache_data(show_spinner=False, hash_funcs={Path: _file_mtime})
def load_video_meta(vm_path: Path = VIDEO_META_JSON) -> Dict[str, Dict[str, Any]]:
    if vm_path.exists():
        try: return json.loads(vm_path.read_text(encoding="utf-8"))
        except: return {}
    return {}

def _vid_epoch(vm: dict, vid: str) -> float:
    info = (vm or {}).get(vid, {})
    return _iso_to_epoch(info.get("published_at") or info.get("publishedAt") or info.get("date") or "")

def _recency_score(published_ts: float, now: float, half_life_days: float) -> float:
    if published_ts <= 0: return 0.0
    days = max(0.0, (now - published_ts) / 86400.0)
    return 0.5 ** (days / max(1e-6, half_life_days))

# -------- JSONL offsets --------
def _ensure_offsets() -> np.ndarray:
    if OFFSETS_NPY.exists():
        try:
            arr = np.load(OFFSETS_NPY)
            saved = len(arr)
            cur = sum(1 for _ in CHUNKS_PATH.open("rb"))
            if cur <= saved: return arr
        except: pass
    pos = 0; offs = []
    with CHUNKS_PATH.open("rb") as f:
        for ln in f:
            offs.append(pos); pos += len(ln)
    arr = np.array(offs, dtype=np.int64)
    OFFSETS_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OFFSETS_NPY, arr)
    return arr

def iter_jsonl_rows(indices: List[int], limit: int | None = None):
    if not CHUNKS_PATH.exists(): return
    offsets = _ensure_offsets()
    want = [i for i in indices if 0 <= i < len(offsets)]
    if limit is not None: want = want[:limit]
    with CHUNKS_PATH.open("rb") as f:
        for i in want:
            f.seek(int(offsets[i]))
            raw = f.readline()
            try: yield i, json.loads(raw)
            except: continue

# -------- Model + FAISS --------
@st.cache_resource(show_spinner=False)
def _load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")

@st.cache_resource(show_spinner=False)
def load_metas_and_model(index_path: Path = INDEX_PATH, metas_path: Path = METAS_PKL):
    if not index_path.exists() or not metas_path.exists():
        return None, None, None
    index = faiss.read_index(str(index_path))
    with metas_path.open("rb") as f:
        payload = pickle.load(f)
    metas_from_pkl = payload.get("metas", [])
    model_name = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    local_dir = DATA_ROOT / "models" / "all-MiniLM-L6-v2"
    try_name = str(local_dir) if (local_dir / "config.json").exists() else model_name
    embedder = _load_embedder(try_name)
    if index.d != embedder.get_sentence_embedding_dimension():
        raise RuntimeError(
            f"Embedding dim mismatch: FAISS={index.d} vs Encoder={embedder.get_sentence_embedding_dimension()}. "
            f"Rebuild index or load encoder used at index time: '{model_name}'."
        )
    return index, metas_from_pkl, {"model_name": try_name, "embedder": embedder}

@st.cache_resource(show_spinner=False)
def load_video_centroids():
    if not (VID_CENT_NPY.exists() and VID_IDS_TXT.exists()):
        return None, None
    C = np.load(VID_CENT_NPY).astype("float32")
    vids = VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
    if C.shape[0] != len(vids):
        return None, None
    return C, vids

@st.cache_data(show_spinner=False)
def load_video_summaries():
    if VID_SUM_JSON.exists():
        try: return json.loads(VID_SUM_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

# -------- MMR --------
def mmr(qv: np.ndarray, doc_vecs: np.ndarray, k: int, lambda_diversity: float = 0.4) -> List[int]:
    if doc_vecs.size == 0: return []
    sim = (doc_vecs @ qv.reshape(-1, 1)).ravel()
    sel = []; cand = set(range(doc_vecs.shape[0]))
    while cand and len(sel) < k:
        if not sel:
            cl = list(cand); pick = cl[int(np.argmax(sim[cl]))]
            sel.append(pick); cand.remove(pick); continue
        sv = doc_vecs[sel]; cl = list(cand)
        max_div = (sv @ doc_vecs[cl].T).max(axis=0)
        scores = lambda_diversity * sim[cl] - (1 - lambda_diversity) * max_div
        pick = cl[int(np.argmax(scores))]
        sel.append(pick); cand.remove(pick)
    return sel

# -------- De-dup passages --------
def _dedupe_passages(items: List[Dict[str, Any]], time_window_sec: float = 8.0, min_chars: int = 40):
    out = []; seen = []
    for h in sorted(items, key=lambda r: float((r.get("meta") or {}).get("start", 0))):
        ts = float((h.get("meta") or {}).get("start", 0))
        txt = _normalize_text(h.get("text", ""))
        if len(txt) < min_chars: continue
        dup = False
        for s in seen:
            ts_s = float((s.get("meta") or {}).get("start", 0))
            if abs(ts - ts_s) <= time_window_sec and _normalize_text(s.get("text", "")) == txt:
                dup = True; break
        if not dup:
            seen.append(h); out.append(h)
    return out

# -------- Retrieval stages --------
def stageA_route_videos(
    qv: np.ndarray, C: np.ndarray, vids: List[str], topN: int,
    allowed_vids: Set[str] | None, vm: dict,
    recency_weight: float, half_life_days: float,
    pin_boost: float = 0.0, pinned: Set[str] | None = None
) -> List[str]:
    sims = (C @ qv.reshape(-1, 1)).ravel()
    now = time.time(); pinned = pinned or set()
    blend = []
    for i, vid in enumerate(vids):
        if allowed_vids and vid not in allowed_vids: continue
        rec = _recency_score(_vid_epoch(vm, vid), now, half_life_days)
        score = (1.0 - recency_weight) * float(sims[i]) + recency_weight * float(rec)
        if vid in pinned: score += float(pin_boost)
        blend.append((vid, score))
    blend.sort(key=lambda x: -x[1])
    return [v for v, _ in blend[:topN]]

def stageB_search_chunks(
    query: str,
    index: faiss.Index, embedder: SentenceTransformer,
    candidate_vids: Set[str],
    initial_k: int, final_k: int, max_videos: int, per_video_cap: int,
    apply_mmr: bool, mmr_lambda: float,
    recency_weight: float, half_life_days: float, vm: dict
) -> List[Dict[str, Any]]:
    if index is None: return []
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
    K = min(int(initial_k), index.ntotal if index.ntotal > 0 else int(initial_k))
    if K <= 0: return []
    D, I = index.search(qv.reshape(1, -1), K)
    idxs = [int(x) for x in I[0] if x >= 0]
    scores0 = [float(s) for s in D[0][:len(idxs)]]

    rows = list(iter_jsonl_rows(idxs))
    texts = []; metas = []; keep_mask = []
    for _, j in rows:
        t = _normalize_text(j.get("text", "")); 
        if not t: keep_mask.append(False); continue
        m = (j.get("meta") or {}).copy()
        vid = (m.get("video_id") or m.get("vid") or m.get("ytid")
               or j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
        if vid: m["video_id"] = vid
        if "start" not in m and "start_sec" in m: m["start"] = m.get("start_sec")
        m["start"] = _parse_ts(m.get("start", 0))
        texts.append(t); metas.append(m)
        keep_mask.append((not candidate_vids) or (vid in candidate_vids))

    if any(keep_mask):
        texts  = [t for t, k in zip(texts, keep_mask) if k]
        metas  = [m for m, k in zip(metas, keep_mask) if k]
        idxs   = [i for i, k in zip(idxs, keep_mask) if k]
        scores0= [s for s, k in zip(scores0, keep_mask) if k]
    if not texts: return []

    doc_vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")
    order = list(range(len(texts)))
    if apply_mmr:
        sel = mmr(qv, doc_vecs, k=min(len(texts), max(8, final_k * 2)), lambda_diversity=float(mmr_lambda))
        order = sel

    now = time.time(); blended = []
    for li in order:
        i_global = idxs[li] if li < len(idxs) else None
        base = scores0[li] if li < len(scores0) else 0.0
        m = metas[li]; t = texts[li]; vid = m.get("video_id")
        rec = _recency_score(_vid_epoch(vm, vid), now, half_life_days)
        score = (1.0 - recency_weight) * float(base) + recency_weight * float(rec)
        blended.append((i_global, score, t, m))
    blended.sort(key=lambda x: -x[1])

    picked = []; seen_per_video = {}; distinct = []
    for ig, sc, tx, me in blended:
        vid = me.get("video_id", "Unknown")
        if vid not in distinct and len(distinct) >= int(max_videos): continue
        c = seen_per_video.get(vid, 0)
        if c >= int(per_video_cap): continue
        if vid not in distinct: distinct.append(vid)
        seen_per_video[vid] = c + 1
        picked.append({"i": ig, "score": float(sc), "text": tx, "meta": me})
        if len(picked) >= int(final_k): break
    return picked

# -------- Grouping --------
def group_hits_by_video(hits: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    g = {}
    for h in hits:
        vid = (h.get("meta") or {}).get("video_id") or "Unknown"
        g.setdefault(vid, []).append(h)
    return g

def build_grouped_evidence_for_prompt(hits: List[Dict[str, Any]], vm: dict, summaries: dict, max_quotes: int = 3) -> str:
    groups = group_hits_by_video(hits)
    ordered = sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)
    lines = []
    for v_idx, (vid, items) in enumerate(ordered, 1):
        info = vm.get(vid, {})
        title = info.get("title") or summaries.get(vid, {}).get("title") or vid
        creator = info.get("podcaster") or info.get("channel") or summaries.get(vid, {}).get("channel") or "Unknown"
        date = info.get("published_at") or info.get("publishedAt") or info.get("date") or summaries.get(vid, {}).get("published_at") or ""
        lines.append(f"[Video {v_idx}] {title} — {creator}" + (f" — {date}" if date else ""))
        summ = summaries.get(vid, {}).get("summary", "")
        if summ:
            lines.append(f"  • summary: {summ[:300]}{'…' if len(summ) > 300 else ''}")
        clean_items = _dedupe_passages(items, time_window_sec=8.0, min_chars=40)
        for h in clean_items[:max_quotes]:
            ts = _format_ts((h.get("meta") or {}).get("start", 0))
            q = _normalize_text(h.get("text", ""))
            if len(q) > 260: q = q[:260] + "…"
            lines.append(f"  • {ts}: “{q}”")
        lines.append("")
    return "\n".join(lines).strip()

# -------- Web fetch --------
def fetch_trusted_snippets(query: str, allowed_domains: List[str], max_snippets: int = 3, per_domain: int = 1, timeout: float = 6.0):
    if not requests or not BeautifulSoup or max_snippets <= 0: return []
    headers = {"User-Agent": "Mozilla/5.0"}
    out = []
    for domain in allowed_domains:
        try:
            resp = requests.get("https://duckduckgo.com/html/", params={"q": f"site:{domain} {query}"}, headers=headers, timeout=timeout)
            if resp.status_code != 200: continue
            soup = BeautifulSoup(resp.text, "html.parser")
            links = [a.get("href") for a in soup.select("a.result__a") if a.get("href") and domain in a.get("href")]
            links = links[:per_domain]
            for url in links:
                try:
                    r2 = requests.get(url, headers=headers, timeout=timeout)
                    if r2.status_code != 200: continue
                    s2 = BeautifulSoup(r2.text, "html.parser")
                    paras = [p.get_text(" ", strip=True) for p in s2.find_all("p")]
                    text = _normalize_text(" ".join(paras))[:2000]
                    if len(text) < 200: continue
                    out.append({"domain": domain, "url": url, "text": text})
                except: continue
            if len(out) >= max_snippets: break
        except: continue
    return out[:max_snippets]

# -------- LLM synthesis --------
def openai_answer(model_name: str, question: str, history: List[Dict[str, str]], grouped_video_block: str, web_snips: List[Dict[str, str]], no_video: bool) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ OPENAI_API_KEY is not set."

    recent = history[-6:]; convo = []
    for m in recent:
        role = m.get("role"); content = m.get("content", "")
        if role in ("user", "assistant") and content:
            label = "User" if role == "user" else "Assistant"
            convo.append(f"{label}: {content}")

    web_lines = []
    for j, s in enumerate(web_snips, 1):
        txt = " ".join((s.get("text", "")).split())[:300]
        dom = s.get("domain", "web"); url = s.get("url", "")
        web_lines.append(f"(W{j}) {dom} — {url}\n“{txt}”")
    web_block = "\n".join(web_lines) if web_lines else "None"

    fallback_line = (
        "If no suitable video evidence exists, you MAY answer from trusted web snippets alone, "
        "but begin with: 'Web-only evidence'.\n"
        if (WEB_FALLBACK and no_video) else
        "Trusted web snippets are supporting evidence.\n"
    )

    system = (
        "Answer from the provided evidence plus trusted web sources. Priority: (1) grouped VIDEO evidence from selected experts, "
        "(2) trusted WEB snippets.\n" +
        fallback_line +
        "Rules:\n"
        "• Cite every claim/step: (Video k) for videos, (DOMAIN Wj) for web.\n"
        "• Prefer human clinical data; label animal/in-vitro/mechanistic explicitly.\n"
        "• Normalize units and report numeric effect sizes when sources provide them (%, mg/dL, mmol/L, ApoB concentration). "
        "If ranges disagree, state both and indicate higher-quality evidence.\n"
        "• list therapeutic OPTIONS by class and drug names mentioned in videos and trusted sites "
        "(e.g., statins, ezetimibe, PCSK9, etc if discussed). "
        "Include mechanism and typical magnitude of change when stated; if dose not provided, write 'dose not specified'. No diagnosis.\n"
        "Structure:\n"
        "• Key summary — specific, robust, detailed, source-grounded, with numbers when available\n"
        "• Practical protocol — numbered, stepwise, actionable; include recommendations and steps\n"
        "• Safety notes — contraindications, interactions, and when to consult a clinician\n"
        "Output must be concise, uncertainty labeled, and free of speculation."
    )

    user_payload = (
        (("Recent conversation:\n" + "\n".join(convo) + "\n\n") if convo else "")
        + f"Question: {question}\n\n"
        + "Grouped Video Evidence:\n" + (grouped_video_block or "None") + "\n\n"
        + "Trusted Web Snippets:\n" + web_block + "\n\n"
        + "Write a concise, well-grounded answer."
    )

    try:
        client = OpenAI(timeout=60)
        r = client.chat.completions.create(
            model=model_name, temperature=0.2,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user_payload}]
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ Generation error: {e}"

# -------- App UI --------
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

with st.sidebar:
    show_diag = st.toggle("Show data diagnostics", value=False, help="Show file locations and last updated times.")
if show_diag:
    colA, colB, colC = st.columns([2, 3, 3])
    with colA: st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    with colB: st.caption(f"chunks.jsonl mtime: {datetime.fromtimestamp(_file_mtime(CHUNKS_PATH)).isoformat() if CHUNKS_PATH.exists() else 'missing'}")
    with colC: st.caption(f"index mtime: {datetime.fromtimestamp(_file_mtime(INDEX_PATH)).isoformat() if INDEX_PATH.exists() else 'missing'}")

with st.sidebar:
    st.header("Answer settings")

    # Tuned defaults for richer context
    initial_k = st.number_input("How many passages to scan first", 32, 5000, 512, 32,
                                help="Search more passages to avoid missing ideas. Higher can be slower.")
    final_k = st.number_input("How many passages to use", 8, 80, 30, 2,
                              help="How many of the best passages the answer can draw from.")

    st.subheader("Keep it focused")
    max_videos = st.number_input("Maximum videos to use", 1, 12, 4, 1,
                                 help="Limits how many different videos contribute to the answer.")
    per_video_cap = st.number_input("Passages per video", 1, 10, 4, 1,
                                    help="Prevents one video from taking over the answer.")

    st.subheader("Balance variety and accuracy")
    use_mmr = st.checkbox("Encourage variety (recommended)", value=True,
                          help="Reduces near-duplicate quotes so you get broader coverage.")
    mmr_lambda = st.slider("Balance: accuracy vs variety", 0.1, 0.9, 0.4, 0.05,
                           help="Move right for tighter matching. Move left for broader coverage.")

    st.subheader("Prefer newer videos")
    recency_weight = st.slider("Recency influence", 0.0, 1.0, 0.30, 0.05,
                               help="How much to favor newer videos.")
    half_life = st.slider("Recency half-life (days)", 7, 720, 180, 7,
                          help="After this many days, the recency effect halves.")

    st.subheader("Route to best videos first")
    topN_videos = st.number_input("Videos to consider before chunk search", 1, 50, 15, 1,
                                  help="First pick likely videos, then search inside them.")

    # Experts list
    vm = load_video_meta()
    def _creator_of(vid: str) -> str:
        info = vm.get(vid, {})
        return info.get("podcaster") or info.get("channel") or "Unknown"

    counts: Dict[str, int] = {}
    for vid in vm:
        c = _creator_of(vid)
        if c in EXCLUDED_CREATORS: continue
        counts[c] = counts.get(c, 0) + 1
    creators_all = sorted(counts.keys(), key=lambda x: x.lower())

    st.subheader("Experts")
    st.caption("Choose the channels/podcasters to include. All are selected by default. Uncheck to exclude.")
    selected_creators_list = []
    for i, name in enumerate(creators_all):
        label = f"{name} ({counts.get(name, 0)})"
        if st.checkbox(label, value=True, key=f"exp_{i}"):
            selected_creators_list.append(name)
    selected_creators: Set[str] = set(selected_creators_list)

    # Candidate pool
    vids_pool = [vid for vid in vm if _creator_of(vid) in selected_creators]

    # Trusted sites
    st.subheader("Trusted sites")
    st.caption("Short, reliable excerpts that support the expert videos. All are selected by default. Uncheck to exclude.")
    allow_web = st.checkbox(
        "Add supporting excerpts from trusted sites",
        value=True,
        help=("If on, short quotes from trusted sites back up the videos."
              if (requests and BeautifulSoup) else
              "Install 'requests' and 'beautifulsoup4' to enable this."),
        disabled=(requests is None or BeautifulSoup is None)
    )
    selected_domains = []
    for i, dom in enumerate(TRUSTED_DOMAINS):
        if st.checkbox(dom, value=True, key=f"site_{i}"):
            selected_domains.append(dom)

    max_web = st.slider("Max supporting excerpts", 0, 8, 3, 1,
                        help="Upper limit on short quotes pulled from trusted sites as supporting evidence.")

    model_choice = st.selectbox("OpenAI model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0,
                                help="Model used to write the final answer.")

    st.divider()
    st.subheader("Library status")
    st.checkbox("chunks.jsonl present", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("offsets built", value=OFFSETS_NPY.exists(), disabled=True)
    st.checkbox("faiss.index present", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("metas.pkl present", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("video_meta.json present", value=VIDEO_META_JSON.exists(), disabled=True)

    cent_ready = VID_CENT_NPY.exists() and VID_IDS_TXT.exists()
    st.caption("Video centroids: ready" if cent_ready else "Video centroids: not found (run scripts/precompute_video_summaries.py)")
    if cent_ready:
        newer_chunks = _file_mtime(CHUNKS_PATH) > max(_file_mtime(VID_CENT_NPY), _file_mtime(VID_IDS_TXT))
        if newer_chunks:
            st.warning("chunks.jsonl changed after centroids were built. Re-run precompute to refresh routing.", icon="⚠️")

# -------- Chat history --------
if "messages" not in st.session_state: st.session_state.messages = []
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# -------- Input --------
prompt = st.chat_input("Ask about sleep, protein timing, ApoB/LDL drugs, fasting, supplements, protocols…")
if prompt is None:
    cols = st.columns([1]*12)
    with cols[-1]:
        st.button("Clear chat", key="clear_idle", help="Start a new conversation.", on_click=_clear_chat)
    st.stop()

st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"): st.markdown(prompt)

# -------- Guardrails --------
missing = [p for p in REQUIRED if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error("Missing required files:\n" + "\n".join(f"- {p}" for p in missing))
    st.stop()

# -------- Load index + encoder --------
try:
    index, metas_from_pkl, payload = load_metas_and_model()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load index or encoder. Details:"); st.exception(e)
    st.stop()

if index is None or payload is None:
    with st.chat_message("assistant"): st.error("Index or model not available.")
    st.stop()

embedder: SentenceTransformer = payload["embedder"]
vm = load_video_meta()
C, vid_list = load_video_centroids()
summaries = load_video_summaries()

# -------- Admin diagnostics --------
if _is_admin():
    st.subheader("Diagnostics (admin)")
    status = precompute_status(payload["model_name"])
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Centroids present", "Yes" if status["centroids_present"] else "No")
        st.metric("IDs present", "Yes" if status["ids_present"] else "No")
        st.metric("Summaries present", "Yes" if status["summaries_present"] else "No")
    with col2:
        st.metric("Shapes OK", "Yes" if status.get("ok_shapes") is True else "No" if status.get("ok_shapes") is False else "n/a")
        st.metric("Norms ~1.0", "Yes" if status.get("ok_norms") is True else "No" if status.get("ok_norms") is False else "n/a")
        st.metric("Dim matches encoder", "Yes" if status.get("ok_dim") is True else "No" if status.get("ok_dim") is False else "n/a")
    with col3:
        st.caption(f"chunks.jsonl mtime: {datetime.fromtimestamp(status['chunks_mtime']).isoformat() if status['chunks_mtime'] else 'n/a'}")
        st.caption(f"centroids mtime: {datetime.fromtimestamp(status['cent_mtime']).isoformat() if status['cent_mtime'] else 'n/a'}")
        st.caption(f"ids mtime: {datetime.fromtimestamp(status['ids_mtime']).isoformat() if status['ids_mtime'] else 'n/a'}")
    if status["msg"]:
        for m in status["msg"]:
            st.warning(m)
    st.markdown("---")
    st.markdown("**Keyword coverage scan (admin)**")
    st.caption("Scans chunks.jsonl for chosen terms. For verification only.")
    default_terms = ["apob","apo-b","apo b","ldl","statin","ezetimibe","pcsk9","bempedoic","inclisiran","niacin"]
    term_input = st.text_input("Terms to scan (comma-separated)", ", ".join(default_terms))
    terms = [t.strip() for t in term_input.split(",") if t.strip()]
    if st.button("Run scan", help="Analyze transcripts for these terms."):
        with st.spinner("Scanning chunks.jsonl…"):
            scan = scan_chunks_for_terms(terms, vm, limit_examples=300)
        st.metric("Total matching chunks", scan["total_matches"])
        if scan["per_creator"]:
            st.markdown("**Matches by expert**")
            st.dataframe(
                [{"expert": k, "matching_chunks": v} for k, v in scan["per_creator"].items()],
                use_container_width=True
            )
        else:
            st.info("No matches found for these terms.")
        if scan["examples"]:
            st.markdown("**Example matches**")
            st.dataframe(scan["examples"], use_container_width=True)
            st.download_button(
                "Download examples (CSV)",
                data=("\n".join(
                    ["video_id,creator,ts,snippet"] + [
                        f"{e['video_id']},{e['creator']},{e['ts']},\"{e['snippet'].replace('\"','\"\"')}\""
                        for e in scan["examples"]
                    ])
                ).encode("utf-8"),
                file_name="diagnostic_examples.csv",
                mime="text/csv"
            )
    st.markdown("---")

# -------- Stage A: route to videos --------
routed_vids = []; candidate_vids: set[str] = set()
with st.spinner("Routing to likely videos…"):
    qv = embedder.encode([prompt], normalize_embeddings=True).astype("float32")[0]
    def _creator_of_vid(vid: str) -> str:
        info = vm.get(vid, {})
        return info.get("podcaster") or info.get("channel") or "Unknown"
    allowed_vids_all = [vid for vid in (vid_list or list(vm.keys())) if _creator_of_vid(vid) in selected_creators]
    allowed_vids = set(allowed_vids_all)
    if C is not None and vid_list is not None:
        routed_vids = stageA_route_videos(qv, C, vid_list, int(topN_videos),
                                          allowed_vids, vm, float(recency_weight), float(half_life),
                                          pin_boost=0.0, pinned=set())
        candidate_vids = set(routed_vids)
    else:
        candidate_vids = allowed_vids

# -------- Stage B: search chunks --------
with st.spinner("Searching inside selected videos…"):
    try:
        hits = stageB_search_chunks(
            prompt, index, embedder, candidate_vids,
            int(initial_k), int(final_k), int(max_videos), int(per_video_cap),
            bool(use_mmr), float(mmr_lambda),
            float(recency_weight), float(half_life), vm
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Search failed."); st.exception(e)
        st.stop()

# -------- Optional web --------
web_snips = []
if allow_web and selected_domains and requests and BeautifulSoup and int(max_web) > 0:
    with st.spinner("Fetching trusted websites…"):
        web_snips = fetch_trusted_snippets(prompt, selected_domains, max_snippets=int(max_web))

if not hits and web_snips and not WEB_FALLBACK:
    with st.chat_message("assistant"):
        st.warning("No relevant expert video evidence found. Trusted sites are supporting-only. Adjust experts or refine your query.")
    cols = st.columns([1]*12)
    with cols[-1]:
        st.button("Clear chat", key="clear_noweb", help="Start a new conversation.", on_click=_clear_chat)
    st.stop()

# -------- Build evidence and answer --------
grouped_block = build_grouped_evidence_for_prompt(hits, vm, summaries, max_quotes=3)

with st.chat_message("assistant"):
    if not hits and not web_snips:
        st.warning("No relevant evidence found.")
        st.session_state.messages.append({"role":"assistant","content":"I couldn’t find enough evidence to answer that."})
        cols = st.columns([1]*12)
        with cols[-1]:
            st.button("Clear chat", key="clear_nohits", help="Start a new conversation.", on_click=_clear_chat)
        st.stop()

    with st.spinner("Writing your answer…"):
        ans = openai_answer(model_choice, prompt, st.session_state.messages, grouped_block, web_snips, no_video=(len(hits)==0))

    st.markdown(ans)
    st.session_state.messages.append({"role":"assistant","content":ans})

    # Sources UI (deduped)
    with st.expander("Sources & timestamps", expanded=False):
        groups = group_hits_by_video(hits)
        ordered = sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)
        export_payload = {"videos": [], "web": []}

        for vid, items in ordered:
            info = vm.get(vid, {})
            title = info.get("title") or summaries.get(vid, {}).get("title") or vid
            creator = info.get("podcaster") or info.get("channel") or summaries.get(vid, {}).get("channel") or ""
            url = info.get("url") or ""
            header = f"**{title}**" + (f" — _{creator}_" if creator else "")
            st.markdown(f"- [{header}]({url})" if url else f"- {header}")

            clean_items = _dedupe_passages(items, time_window_sec=8.0, min_chars=40)
            v_entry = {"video_id": vid, "title": title, "creator": creator, "url": url, "quotes": []}
            for h in clean_items:
                ts = _format_ts((h.get("meta") or {}).get("start", 0))
                quote = _normalize_text(h.get("text", ""))
                if len(quote) > 160: quote = quote[:160] + "…"
                st.markdown(f"  • **{ts}** — “{quote}”")
                v_entry["quotes"].append({"ts": ts, "text": quote})
            export_payload["videos"].append(v_entry)

        if web_snips:
            st.markdown("**Trusted websites**")
            for j, s in enumerate(web_snips, 1):
                st.markdown(f"W{j}. [{s['domain']}]({s['url']})")
                export_payload["web"].append({"id": f"W{j}", "domain": s["domain"], "url": s["url"]})

        st.download_button(
            "Download sources as JSON",
            data=json.dumps(export_payload, ensure_ascii=False, indent=2),
            file_name="sources.json",
            mime="application/json",
            help="Save the cited sources and timestamps."
        )

# -------- Footer hint --------
st.caption(
    "If you add new videos or change chunks.jsonl, run: "
    "`DATA_DIR=/var/data python scripts/precompute_video_summaries.py` "
    "to refresh video centroids and summaries."
)

# -------- Bottom-right Clear Chat --------
cols = st.columns([1]*12)
with cols[-1]:
    st.button("Clear chat", key="clear_done", help="Start a new conversation.", on_click=_clear_chat)

# -------- Non-fatal runtime check --------
try:
    if VID_CENT_NPY.exists():
        C_tmp = np.load(VID_CENT_NPY)
        emb_dim = _load_embedder(payload["model_name"]).get_sentence_embedding_dimension()
        if C_tmp.shape[1] != emb_dim:
            st.warning("Video centroids dim != encoder dim. Re-run precompute with the same model used for the index.", icon="⚠️")
except Exception:
    pass
