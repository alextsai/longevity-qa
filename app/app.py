# app/app.py
# -*- coding: utf-8 -*-
"""
Longevity / Nutrition Q&A — experts-first RAG with robust routing, strict source gating,
reliable web snippets, and per-turn auditable sources.

Key improvements
- Embeddings: intfloat/e5-large-v2 on CPU; auto L2-normalize; centroid file re-normalization.
- Domain router (optional): LinearSVC + Platt + Mahalanobis OOD + tag overlap gating.
- Quote quality: stricter filters + semantic gate + per-video top-K; fallback uses up to 2 summary bullets.
- Source accuracy: every quote carries video_id, title, creator, ts, url; web snippets include real URLs and first paragraph context.
- Per turn sources: each reply stores its own immutable sources; earlier turns remain visible.
- Diagnostics: self-check footer (videos used, quotes used, experts covered, web selected, web trace), admin tools.

Runs without classifier or centroid files; features auto-disable cleanly.
"""

from __future__ import annotations
import os, sys, re, json, time, math, pickle, collections, warnings
from pathlib import Path
from typing import Dict, List, Any, Tuple, Set
from datetime import datetime

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Optional deps
try:
    import joblib
except Exception:
    joblib = None
try:
    import yaml
except Exception:
    yaml = None
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

# -------------------- Config --------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU

DATA_ROOT = Path(os.getenv("DATA_DIR", "/var/data")).resolve()

CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"
OFFSETS_NPY     = DATA_ROOT / "data/chunks/chunks.offsets.npy"
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"     # only for metadata; encoder ignored
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"
VID_CENT_NPY    = DATA_ROOT / "data/index/video_centroids.npy"
VID_IDS_TXT     = DATA_ROOT / "data/index/video_ids.txt"
VID_SUM_JSON    = DATA_ROOT / "data/catalog/video_summaries.json"

# Robust router assets (optional)
DOMAIN_MODEL    = DATA_ROOT / "models/domain_model.joblib"
SCALER_FILE     = DATA_ROOT / "models/scaler.joblib"
DOMAIN_YAML     = DATA_ROOT / "models/domain_probs.yaml"
EMB_CENTROID    = DATA_ROOT / "data/index/emb_centroids.npy"  # class centroids (optional)
EMB_COV         = DATA_ROOT / "data/index/emb_cov.npy"        # shared covariance (optional)

WEB_FALLBACK = os.getenv("WEB_FALLBACK", "true").lower() in {"1", "true", "yes", "on"}

TRUSTED_DOMAINS = [
    "nih.gov","medlineplus.gov","cdc.gov","mayoclinic.org","health.harvard.edu",
    "familydoctor.org","healthfinder.gov","ama-assn.org","medicalxpress.com",
    "sciencedaily.com","nejm.org","med.stanford.edu","icahn.mssm.edu"
]

ALLOWED_CREATORS = [
    "Dr. Pradip Jamnadas, MD",
    "Andrew Huberman",
    "Healthy Immune Doc",
    "Peter Attia MD",
    "The Diary of A CEO",
]
EXCLUDED_CREATORS_EXACT = {
    "they diary of a ceo and louse tomlinson",
    "dr. pradip jamnadas, md and the primal podcast",
}

REQUIRED_FILES = [INDEX_PATH, CHUNKS_PATH, VIDEO_META_JSON]

# -------------------- Utils --------------------
def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _parse_ts(v) -> float:
    if isinstance(v, (int, float)):
        try: return float(v)
        except: return 0.0
    try:
        sec = 0.0
        for p in str(v).split(":"): sec = sec*60 + float(p)
        return sec
    except: return 0.0

def _iso_to_epoch(iso: str) -> float:
    if not iso: return 0.0
    try:
        if "T" in iso: return datetime.fromisoformat(iso.replace("Z","+00:00")).timestamp()
        return datetime.fromisoformat(iso).timestamp()
    except: return 0.0

def _format_ts(sec: float) -> str:
    sec = int(max(0, float(sec))); h, r = divmod(sec, 3600); m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def _file_mtime(p: Path) -> float:
    try: return p.stat().st_mtime
    except: return 0.0

def _iso(ts: float) -> str:
    try: return datetime.fromtimestamp(ts).isoformat()
    except: return "n/a"

def _clear_chat():
    st.session_state["messages"] = []
    st.session_state["turns"] = []
    st.rerun()

# -------------------- Admin gate --------------------
def _is_admin() -> bool:
    try: qp = st.query_params
    except Exception: return False
    if qp.get("admin","0") != "1": return False
    try: expected = st.secrets["ADMIN_KEY"]
    except Exception: expected = None
    if expected is None: return True
    return qp.get("key","") == str(expected)

# -------------------- IO --------------------
@st.cache_data(show_spinner=False)
def load_video_meta() -> Dict[str, Dict[str, Any]]:
    if VIDEO_META_JSON.exists():
        try: return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

@st.cache_data(show_spinner=False)
def load_video_summaries() -> Dict[str, Any]:
    if VID_SUM_JSON.exists():
        try: return json.loads(VID_SUM_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

@st.cache_resource(show_spinner=False)
def _load_embedder() -> SentenceTransformer:
    model = "intfloat/e5-large-v2"
    emb = SentenceTransformer(model, device="cpu")
    return emb

@st.cache_resource(show_spinner=False)
def load_faiss_and_embedder():
    if not INDEX_PATH.exists():
        return None, None
    index = faiss.read_index(str(INDEX_PATH))
    embedder = _load_embedder()
    # Force FAISS to expect e5-large-v2 dimensionality
    if index.d != embedder.get_sentence_embedding_dimension():
        raise RuntimeError(f"FAISS dim {index.d} != encoder dim {embedder.get_sentence_embedding_dimension()}. Rebuild index with e5-large-v2.")
    return index, embedder

@st.cache_resource(show_spinner=False)
def load_video_centroids():
    if not (VID_CENT_NPY.exists() and VID_IDS_TXT.exists()):
        return None, None
    C = np.load(VID_CENT_NPY).astype("float32")
    # auto renormalize to ~1.0
    n = np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
    C = C / n
    np.save(VID_CENT_NPY, C)
    vids = VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
    if C.shape[0] != len(vids): return None, None
    return C, vids

# JSONL random access
def _ensure_offsets() -> np.ndarray:
    if OFFSETS_NPY.exists():
        try:
            arr = np.load(OFFSETS_NPY)
            saved = len(arr); cur = sum(1 for _ in CHUNKS_PATH.open("rb"))
            if cur <= saved: return arr
        except: pass
    pos, offs = 0, []
    with CHUNKS_PATH.open("rb") as f:
        for ln in f:
            offs.append(pos); pos += len(ln)
    arr = np.array(offs, dtype=np.int64)
    OFFSETS_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OFFSETS_NPY, arr)
    return arr

def iter_jsonl_rows(indices: List[int], limit: int|None=None):
    if not CHUNKS_PATH.exists(): return
    offs = _ensure_offsets()
    want = [i for i in indices if 0 <= i < len(offs)]
    if limit is not None: want = want[:limit]
    with CHUNKS_PATH.open("rb") as f:
        for i in want:
            f.seek(int(offs[i])); raw = f.readline()
            try: yield i, json.loads(raw)
            except: continue

# -------------------- Creator mapping --------------------
def _raw_creator_of_vid(vid: str, vm: dict) -> str:
    info = vm.get(vid, {}) or {}
    for k in ("podcaster","channel","author","uploader","owner","creator"):
        if info.get(k): return str(info[k])
    for k,v in ((kk.lower(),vv) for kk,vv in info.items()):
        if k in {"podcaster","channel","author","uploader","owner","creator"} and v:
            return str(v)
    return "Unknown"

def _canonicalize_creator(name: str) -> str | None:
    n = _normalize_text(name).lower().replace("™","").replace("®","")
    if not n: return None
    if n in EXCLUDED_CREATORS_EXACT: return None
    tokens = set(re.findall(r"[a-z0-9]+", n))
    if ("healthy" in tokens and "immune" in tokens) or "healthyimmunedoc" in tokens:
        return "Healthy Immune Doc"
    if "diary" in tokens and "ceo" in tokens:
        return "The Diary of A CEO"
    if "huberman" in tokens:
        return "Andrew Huberman"
    if "attia" in tokens:
        return "Peter Attia MD"
    if "jamnadas" in tokens:
        return "Dr. Pradip Jamnadas, MD"
    for canon in ALLOWED_CREATORS:
        if n == canon.lower(): return canon
        if re.sub(r"[^\w\s]","",n) == re.sub(r"[^\w\s]","",canon.lower()):
            return canon
    return None

def build_creator_indexes_from_chunks(vm: dict) -> tuple[dict, dict]:
    vid_to_creator: Dict[str,str] = {}
    creator_to_vids: Dict[str,set] = {}
    if CHUNKS_PATH.exists():
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            for ln in f:
                try: j = json.loads(ln)
                except: continue
                m = (j.get("meta") or {})
                vid = (m.get("video_id") or m.get("vid") or m.get("ytid") or
                       j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
                if not vid: continue
                raw = (m.get("channel") or m.get("author") or m.get("uploader") or _raw_creator_of_vid(vid, vm))
                canon = _canonicalize_creator(raw)
                if canon is None: continue
                if vid not in vid_to_creator:
                    vid_to_creator[vid] = canon
                    creator_to_vids.setdefault(canon, set()).add(vid)
    for vid in vm.keys():
        if vid in vid_to_creator: continue
        canon = _canonicalize_creator(_raw_creator_of_vid(vid, vm))
        if canon is None: continue
        vid_to_creator[vid] = canon
        creator_to_vids.setdefault(canon, set()).add(vid)
    return vid_to_creator, creator_to_vids

# -------------------- Domain classifier (optional) --------------------
@st.cache_resource(show_spinner=False)
def load_domain_router():
    if not (joblib and DOMAIN_MODEL.exists() and SCALER_FILE.exists() and yaml and DOMAIN_YAML.exists()):
        return None
    try:
        model = joblib.load(DOMAIN_MODEL)
        scaler = joblib.load(SCALER_FILE)
        priors = yaml.safe_load(DOMAIN_YAML.read_text())
        cent = np.load(EMB_CENTROID) if EMB_CENTROID.exists() else None
        cov = np.load(EMB_COV) if EMB_COV.exists() else None
        inv_cov = None
        if cov is not None:
            try:
                inv_cov = np.linalg.inv(cov + 1e-6*np.eye(cov.shape[0]))
            except Exception:
                inv_cov = None
        return {"model": model, "scaler": scaler, "priors": priors, "cent": cent, "inv_cov": inv_cov}
    except Exception:
        return None

def _mahalanobis(x: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> float:
    d = x - mu
    return float(d.T @ inv_cov @ d)

def route_domain_tags(embedder: SentenceTransformer, router: dict | None, text: str) -> tuple[str|None, float, Set[str]]:
    """Return (domain, calibrated_prob, tag_set). If router missing → (None,0,set())."""
    if router is None:
        # naive tags from query tokens, lowercase
        tags = set(t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3)
        return None, 0.0, tags
    v = embedder.encode([text], normalize_embeddings=True).astype("float32")[0]
    xs = router["scaler"].transform(v.reshape(1,-1))
    prob = float(router["model"].predict_proba(xs)[0].max())
    domain = router["model"].classes_[int(np.argmax(router["model"].predict_proba(xs)[0]))]
    # OOD check if centroids+cov available
    if router["cent"] is not None and router["inv_cov"] is not None:
        dists = np.array([_mahalanobis(v, mu, router["inv_cov"]) for mu in router["cent"]], dtype=np.float32)
        dmin = float(dists.min())
        thr = float(router["priors"].get("ood_threshold", 12.0))
        if dmin > thr:
            return None, 0.0, set(re.findall(r"[a-z0-9]+", text.lower()))
    tags = set(re.findall(r"[a-z0-9]+", text.lower()))
    return str(domain), prob, tags

# -------------------- Routing by summaries + centroids --------------------
def _recency_score(published_ts: float, now: float, half_life_days: float) -> float:
    if published_ts <= 0: return 0.0
    days = max(0.0, (now - published_ts) / 86400.0)
    return 0.5 ** (days / max(1e-6, half_life_days))

def _build_idf_over_bullets(summaries: dict) -> dict:
    DF = collections.Counter()
    for v in summaries.keys():
        for b in summaries.get(v, {}).get("bullets", []):
            for w in set(re.findall(r"[a-z0-9]+", (b.get("text","") or "").lower())):
                DF[w] += 1
    N = max(1, len(summaries))
    return {w: math.log((N+1)/(df+0.5)) for w,df in DF.items()}

def _kw_score(text: str, query: str, idf: dict) -> Tuple[float,int]:
    if not text: return 0.0, 0
    qtok = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if w]
    ttok = re.findall(r"[a-z0-9]+", (text or "").lower())
    tf = {w: ttok.count(w) for w in set(ttok)}
    overlap = len(set(qtok) & set(ttok))
    score = sum(tf.get(w,0) * idf.get(w,0.0) for w in set(qtok)) / (len(ttok)+1e-6)
    return float(score), int(overlap)

def _vid_epoch(vm: dict, vid: str) -> float:
    info = (vm or {}).get(vid, {})
    return _iso_to_epoch(info.get("published_at") or info.get("publishedAt") or info.get("date") or "")

def route_videos(query: str, qv: np.ndarray, summaries: dict, vm: dict,
                 C: np.ndarray|None, vids: List[str]|None, allowed_vids: Set[str],
                 topK: int, recency_weight: float, half_life_days: float,
                 min_kw_overlap: int, domain: str|None, domain_prob: float, tags: Set[str]) -> List[str]:
    universe = [v for v in (vids or list(vm.keys())) if (not allowed_vids or v in allowed_vids)]
    if not universe: return []
    idf = _build_idf_over_bullets(summaries)
    now = time.time()
    cent = {}
    if C is not None and vids is not None and len(vids)==C.shape[0]:
        sim = (C @ qv.reshape(-1,1)).ravel()
        cent = {vids[i]: float(sim[i]) for i in range(len(vids))}
    scored = []
    for v in universe:
        bullets = summaries.get(v, {}).get("bullets", [])
        pseudo = " ".join(b.get("text","") for b in bullets)[:2000]
        kw, overlap = _kw_score(pseudo, query, idf)
        if overlap < min_kw_overlap and pseudo:
            continue
        # optional domain/tag gating: require at least one tag hit when domain prob high
        gate_bonus = 0.0
        if domain and domain_prob >= 0.55:
            hit = any(t in pseudo.lower() for t in tags)
            if not hit: 
                continue
            gate_bonus = 0.05
        cs  = cent.get(v, 0.0)
        rec = _recency_score(_vid_epoch(vm, v), now, half_life_days)
        base = 0.6*cs + 0.3*kw + gate_bonus
        score = (1.0 - recency_weight)*base + recency_weight*(0.1*rec + 0.9*base)
        scored.append((v, score))
    scored.sort(key=lambda x:-x[1])
    return [v for v,_ in scored[:int(topK)]]

# -------------------- Quote selection --------------------
def _quote_is_valid(text: str) -> bool:
    t = _normalize_text(text)
    if len(t) < 48: return False
    return any(x in t for x in [". ","; ",": ","? ","! "])

def _score_quote_semantic(qv: np.ndarray, emb: SentenceTransformer, text: str) -> float:
    dv = emb.encode([text], normalize_embeddings=True).astype("float32")[0]
    return float(np.clip(np.dot(qv, dv), -1.0, 1.0))

def select_quotes_from_hits(query: str, hits: List[Dict[str,Any]], vm: dict, emb: SentenceTransformer,
                            per_video_cap: int, min_sim: float = 0.28) -> Dict[str, List[Dict[str,Any]]]:
    """Group by video and keep top-N quotes per video with semantic gate."""
    qv = emb.encode([query], normalize_embeddings=True).astype("float32")[0]
    groups: Dict[str, List[Dict[str,Any]]] = {}
    scored_local: Dict[str, List[Tuple[float, Dict[str,Any]]]] = {}
    for h in hits:
        vid = (h.get("meta") or {}).get("video_id") or "Unknown"
        txt = _normalize_text(h.get("text",""))
        if not _quote_is_valid(txt): continue
        sim = _score_quote_semantic(qv, emb, txt)
        if sim < min_sim: continue
        scored_local.setdefault(vid, []).append((sim, h))
    for vid, arr in scored_local.items():
        arr.sort(key=lambda x: -x[0])
        keep = [h for _,h in arr[:max(1, per_video_cap)]]
        groups[vid] = keep
    return groups

def group_hits_by_video(hits: List[Dict[str,Any]]) -> Dict[str, List[Dict[str,Any]]]:
    g: Dict[str,List[Dict[str,Any]]] = {}
    for h in hits:
        vid = (h.get("meta") or {}).get("video_id") or "Unknown"
        g.setdefault(vid, []).append(h)
    return g

# -------------------- Stage-B dense search --------------------
def mmr(qv: np.ndarray, doc_vecs: np.ndarray, k: int, lam: float = 0.45) -> List[int]:
    if doc_vecs.size == 0: return []
    sim = (doc_vecs @ qv.reshape(-1,1)).ravel()
    sel = []; cand = set(range(doc_vecs.shape[0]))
    while cand and len(sel) < k:
        if not sel:
            cl = list(cand); pick = cl[int(np.argmax(sim[cl]))]
            sel.append(pick); cand.remove(pick); continue
        sv = doc_vecs[sel]; cl = list(cand)
        max_div = (sv @ doc_vecs[cl].T).max(axis=0)
        scores = lam*sim[cl] - (1-lam)*max_div
        pick = cl[int(np.argmax(scores))]
        sel.append(pick); cand.remove(pick)
    return sel

def dense_search(query: str, index: faiss.Index, emb: SentenceTransformer,
                 candidate_vids: Set[str], initial_k: int, final_k: int, max_videos: int, per_video_cap: int,
                 use_mmr: bool, lam: float, recency_weight: float, half_life_days: float, vm: dict) -> List[Dict[str,Any]]:
    if index is None: return []
    qv = emb.encode([query], normalize_embeddings=True).astype("float32")[0]
    K = min(int(initial_k), index.ntotal if index.ntotal>0 else int(initial_k))
    D,I = index.search(qv.reshape(1,-1), K)
    idxs   = [int(x) for x in I[0] if x>=0]
    scores = [float(s) for s in D[0][:len(idxs)]]

    rows = list(iter_jsonl_rows(idxs))
    texts, metas, keep, base = [], [], [], []
    for pos, j in rows:
        t = _normalize_text(j.get("text",""))
        m = (j.get("meta") or {}).copy()
        vid = (m.get("video_id") or m.get("vid") or m.get("ytid") or
               j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
        if vid: m["video_id"] = vid
        if "start" not in m and "start_sec" in m: m["start"] = m.get("start_sec")
        m["start"] = _parse_ts(m.get("start", 0))
        if t:
            texts.append(t); metas.append(m)
            keep.append((candidate_vids is None) or (vid in candidate_vids))
    if any(keep):
        texts  = [t for t,k in zip(texts,keep) if k]
        metas  = [m for m,k in zip(metas,keep) if k]
        idxs   = [i for i,k in zip(idxs,keep) if k]
        scores = [s for s,k in zip(scores,keep) if k]
    if not texts: return []

    doc_vecs = emb.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")
    order = list(range(len(texts)))
    if use_mmr:
        order = mmr(qv, doc_vecs, k=min(len(texts), max(8, final_k*2)), lam=float(lam))
    now = time.time(); blended=[]
    for li in order:
        base = scores[li] if li < len(scores) else 0.0
        m,t = metas[li], texts[li]
        vid = m.get("video_id")
        rec = _recency_score(_vid_epoch(vm, vid), now, half_life_days)
        sc  = (1.0-recency_weight)*float(base) + recency_weight*float(rec)
        blended.append((idxs[li], sc, t, m))
    blended.sort(key=lambda x: -x[1])

    picked, seen_per, distinct = [], {}, []
    for ig, sc, tx, me in blended:
        vid = me.get("video_id","Unknown")
        if vid not in distinct and len(distinct) >= int(max_videos): continue
        if seen_per.get(vid,0) >= int(per_video_cap): continue
        if vid not in distinct: distinct.append(vid)
        seen_per[vid] = seen_per.get(vid,0)+1
        picked.append({"i": ig, "score": float(sc), "text": tx, "meta": me})
        if len(picked) >= int(final_k): break
    return picked

# -------------------- Evidence assembly --------------------
def build_grouped_evidence(query: str, hits: List[Dict[str,Any]], vm: dict, summaries: dict,
                           emb: SentenceTransformer, per_video_cap: int, fallback_bullets: int = 2) -> Tuple[str, Dict[str,Any]]:
    """
    Assemble human-readable block for LLM and a structured export. If no quotes after selection,
    fallback uses up to `fallback_bullets` summary bullets per routed video, else top raw line.
    """
    # strict per-video semantic selection
    selected = select_quotes_from_hits(query, hits, vm, emb, per_video_cap=per_video_cap, min_sim=0.28)
    groups_all = group_hits_by_video(hits)

    block_lines = []; export_v = []
    # maintain routed order by max score
    ordered = sorted(groups_all.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)
    for v_idx, (vid, items) in enumerate(ordered, 1):
        info = vm.get(vid, {})
        title = info.get("title") or summaries.get(vid,{}).get("title") or vid
        creator_raw = _raw_creator_of_vid(vid, vm)
        creator = _canonicalize_creator(creator_raw) or creator_raw
        date = info.get("published_at") or info.get("publishedAt") or info.get("date") or ""
        url  = info.get("url") or f"https://www.youtube.com/watch?v={vid}"
        block_lines.append(f"[Video {v_idx}] {title} — {creator}" + (f" — {date}" if date else ""))

        quotes = selected.get(vid, [])
        added = 0
        for h in quotes:
            ts = _format_ts((h.get("meta") or {}).get("start", 0))
            q  = _normalize_text(h.get("text",""))[:260]
            block_lines.append(f"  • {ts}: “{q}”")
            export_v.append({"video_id": vid, "title": title, "creator": creator, "url": url, "ts": ts, "text": q})
            added += 1

        if added == 0:
            # fallback: up to N bullets from summaries
            bullets = summaries.get(vid, {}).get("bullets", [])[:max(0,fallback_bullets)]
            b_used = 0
            for b in bullets:
                q = _normalize_text(b.get("text",""))
                if not _quote_is_valid(q): continue
                ts = _format_ts(float(b.get("ts", 0)))
                block_lines.append(f"  • {ts}: “{q[:260]}”")
                export_v.append({"video_id": vid, "title": title, "creator": creator, "url": url, "ts": ts, "text": q})
                b_used += 1
                if b_used >= fallback_bullets: break

            # still empty → take highest-scoring raw line
            if b_used == 0 and items:
                best = sorted(items, key=lambda r: -float(r.get("score",0)))[0]
                ts = _format_ts((best.get("meta") or {}).get("start",0))
                q  = _normalize_text(best.get("text",""))[:260]
                block_lines.append(f"  • {ts}: “{q}”")
                export_v.append({"video_id": vid, "title": title, "creator": creator, "url": url, "ts": ts, "text": q})
        block_lines.append("")

    return "\n".join(block_lines).strip(), {"videos": export_v}

# -------------------- Trusted web snippets --------------------
def _ddg(domain: str, query: str, headers: dict, timeout: float) -> List[str]:
    try:
        r = requests.get("https://duckduckgo.com/html/", params={"q": f"site:{domain} {query}"}, headers=headers, timeout=timeout)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, "html.parser")
        hrefs = [a.get("href") for a in soup.select("a.result__a") if a.get("href")]
        # normalize
        cleaned = []
        for u in hrefs:
            if u.startswith("http"): cleaned.append(u)
        return cleaned
    except Exception:
        return []

def fetch_trusted_snippets(query: str, allowed_domains: List[str], max_snippets: int = 3, per_domain: int = 1, timeout: float = 6.0):
    if not (requests and BeautifulSoup): return []
    out, trace = [], []
    headers = {"User-Agent": "Mozilla/5.0"}
    for dom in allowed_domains:
        links = _ddg(dom, query, headers, timeout)
        if not links:
            links = [f"https://{dom}"]
            trace.append(f"{dom}: fallback homepage")
        else:
            trace.append(f"{dom}: hits={len(links)}")
        for url in links[:per_domain]:
            try:
                r = requests.get(url, headers=headers, timeout=timeout)
                if r.status_code != 200:
                    trace.append(f"{dom}: {url} [{r.status_code}]"); continue
                soup = BeautifulSoup(r.text, "html.parser")
                paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                text = _normalize_text(" ".join(paras))[:1800]
                if len(text) < 200:
                    trace.append(f"{dom}: short"); continue
                out.append({"domain": dom, "url": url, "text": text})
            except Exception as e:
                trace.append(f"{dom}: fetch err")
        if len(out) >= max_snippets: break
    st.session_state["web_trace"] = "; ".join(trace) if trace else "none"
    return out[:max_snippets]

# -------------------- LLM answer --------------------
def openai_answer(model_name: str, question: str, history, grouped_block: str,
                  web_snips: list[dict], no_video: bool) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ OPENAI_API_KEY is not set."
    recent = [m for m in history[-6:] if m.get("role") in ("user","assistant")]
    convo = [("User: " if m["role"]=="user" else "Assistant: ")+m.get("content","") for m in recent]
    web_lines = [f"(W{j}) {s.get('domain','web')} — {s.get('url','')}\n“{_normalize_text(s.get('text',''))[:300]}”"
                 for j,s in enumerate(web_snips,1)]
    web_block = "\n".join(web_lines) if web_lines else "None"
    fallback_line = ("If no suitable video evidence exists, you MAY answer from trusted web snippets alone, "
                     "but begin with: 'Web-only evidence'.\n") if (WEB_FALLBACK and no_video) else \
                    "Trusted web snippets are supporting evidence.\n"
    system = (
        "Answer from the provided evidence plus trusted web sources. Priority: (1) grouped VIDEO evidence from selected experts, "
        "(2) trusted WEB snippets.\n" + fallback_line +
        "Rules:\n"
        "• Cite every claim: (Video k) for videos, (DOMAIN Wj) for web.\n"
        "• Prefer human clinical data; label animal/in-vitro/mechanistic.\n"
        "• Normalize units; include effect sizes when present.\n"
        "• Provide: Key summary • Practical protocol • Safety notes. No diagnosis.\n"
        "Use only quoted bullets/snippets; do not invent claims."
    )
    user_payload = (("Recent conversation:\n"+"\n".join(convo)+"\n\n") if convo else "") + \
                   f"Question: {question}\n\nGrouped Video Evidence:\n{grouped_block or 'None'}\n\n" + \
                   f"Trusted Web Snippets:\n{web_block}\n\nWrite a concise, source-grounded answer."
    try:
        client = OpenAI(timeout=60)
        r = client.chat.completions.create(
            model=model_name, temperature=0.2,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user_payload}]
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ Generation error: {e}"

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

if "turns" not in st.session_state: st.session_state["turns"] = []
if "messages" not in st.session_state: st.session_state["messages"] = []

with st.sidebar:
    st.markdown("**Auto Mode** · accuracy + diversity")

    vm = load_video_meta()
    vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
    counts={canon: len(creator_to_vids.get(canon,set())) for canon in ALLOWED_CREATORS}

    st.subheader("Experts")
    st.caption("All selected. Uncheck to exclude any expert.")
    selected_creators_list=[]
    for i, canon in enumerate(ALLOWED_CREATORS):
        label=f"{canon} ({counts.get(canon,0)})"
        if st.checkbox(label, value=True, key=f"exp_{i}"):
            selected_creators_list.append(canon)
    selected_creators:set[str]=set(selected_creators_list)
    st.session_state["selected_creators"]=selected_creators

    st.subheader("Trusted sites")
    st.caption("Adds short excerpts from vetted medical sites.")
    allow_web = st.checkbox("Include supporting website excerpts", value=True)
    selected_domains=[]
    for i,dom in enumerate(TRUSTED_DOMAINS):
        if st.checkbox(dom, value=True, key=f"site_{i}"):
            selected_domains.append(dom)
    max_web_auto = 3

    model_choice = st.selectbox("Answering model", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)

    with st.expander("Advanced (technical controls)", expanded=False):
        st.caption("Defaults are tuned.")
        st.number_input("Scan candidates first (K)", 128, 5000, 1024, 64, key="adv_scanK")
        st.number_input("Use top passages", 8, 120, 36, 2, key="adv_useK")
        st.number_input("Max videos", 1, 12, 5, 1, key="adv_maxvid")
        st.number_input("Passages per video cap", 1, 10, 4, 1, key="adv_cap")
        st.checkbox("Diversify with MMR", value=True, key="adv_mmr")
        st.slider("MMR balance", 0.1, 0.9, 0.45, 0.05, key="adv_lam")
        st.slider("Recency weight", 0.0, 1.0, 0.20, 0.05, key="adv_rec")
        st.slider("Recency half-life (days)", 7, 720, 270, 7, key="adv_hl")

    st.divider()
    show_diag = st.toggle("Show data diagnostics", value=False)
    st.subheader("Library status")
    st.checkbox("chunks.jsonl present", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("offsets built", value=OFFSETS_NPY.exists(), disabled=True)
    st.checkbox("faiss.index present", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("video_meta.json present", value=VIDEO_META_JSON.exists(), disabled=True)
    cent_ready = VID_CENT_NPY.exists() and VID_IDS_TXT.exists() and VID_SUM_JSON.exists()
    st.caption("Video centroids/summaries: ready" if cent_ready else "Precompute not found.")

if show_diag:
    colA,colB,colC = st.columns([2,3,3])
    with colA: st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    with colB: st.caption(f"chunks mtime: {_iso(_file_mtime(CHUNKS_PATH)) if CHUNKS_PATH.exists() else 'missing'}")
    with colC: st.caption(f"index mtime: {_iso(_file_mtime(INDEX_PATH)) if INDEX_PATH.exists() else 'missing'}")

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Admin panel
if _is_admin():
    st.subheader("Diagnostics (admin)")
    st.code(json.dumps({
        "chunks": str(CHUNKS_PATH), "index": str(INDEX_PATH),
        "video_meta": str(VIDEO_META_JSON), "centroids": str(VID_CENT_NPY),
        "ids": str(VID_IDS_TXT), "summaries": str(VID_SUM_JSON),
        "domain_model": str(DOMAIN_MODEL), "scaler": str(SCALER_FILE),
        "priors_yaml": str(DOMAIN_YAML)
    }, indent=2))
    if st.button("Clear chat"):
        _clear_chat()

# -------------- Input --------------
prompt = st.chat_input("Ask about ApoB/LDL drugs, sleep, fasting, supplements, exercise, protocols…")
if prompt is None:
    # show prior turns
    if st.session_state["turns"]:
        st.subheader("Previous replies and their sources")
        for i, t in enumerate(st.session_state["turns"], 1):
            with st.expander(f"Turn {i}: {t.get('prompt','')[:80]}"):
                st.markdown(t.get("answer",""))
                vids = t.get("videos",[])
                web  = t.get("web",[])
                if vids:
                    st.markdown("**Video quotes**")
                    for v in vids:
                        st.markdown(f"- **{v['title']}** — _{v['creator']}_ • {v['ts']} • [{v['url']}]({v['url']})")
                        st.markdown(f"  • “{_normalize_text(v['text'])[:160]}”")
                if web:
                    st.markdown("**Trusted websites**")
                    for j,s in enumerate(web,1):
                        st.markdown(f"W{j}. [{s['domain']}]({s['url']})")
    cols = st.columns([1]*12)
    with cols[-1]:
        st.button("Clear chat", key="clear_idle", on_click=_clear_chat)
    st.stop()

st.session_state["messages"].append({"role":"user","content":prompt})
with st.chat_message("user"): st.markdown(prompt)

# -------------- Guards --------------
missing = [p for p in REQUIRED_FILES if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error("Missing required files:\n" + "\n".join(f"- {p}" for p in missing))
    st.stop()

# -------------- Load assets --------------
try:
    index, embedder = load_faiss_and_embedder()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load FAISS or encoder."); st.exception(e); st.stop()

vm         = load_video_meta()
C, vid_ids = load_video_centroids()
summaries  = load_video_summaries()
router     = load_domain_router()

# Allowed videos by selected experts
vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
universe = set(vid_ids or list(vm.keys()) or list(vid_to_creator.keys()))
chosen   = st.session_state.get("selected_creators", set(ALLOWED_CREATORS))
allowed  = {vid for vid in universe if vid_to_creator.get(vid) in chosen}

# Routing query uses light follow-up context
routing_query = prompt
prev_users = [m["content"] for m in st.session_state["messages"] if m["role"]=="user"]
if len(prev_users) >= 2: routing_query = prev_users[-2] + " ; " + prompt
qv = embedder.encode([routing_query], normalize_embeddings=True).astype("float32")[0]

# Domain route + tags
domain, dprob, tags = route_domain_tags(embedder, router, routing_query)

# Stage-A routing
routed_vids = route_videos(
    routing_query, qv, summaries, vm, C, list(universe), allowed,
    topK=5, recency_weight=0.20, half_life_days=270, min_kw_overlap=2,
    domain=domain, domain_prob=dprob, tags=tags
)
candidate_vids = set(routed_vids) if routed_vids else allowed

# Stage-B dense search
K_scan = int(st.session_state.get("adv_scanK", 1024))
K_use  = int(st.session_state.get("adv_useK", 36))
max_v  = int(st.session_state.get("adv_maxvid", 5))
cap_v  = int(st.session_state.get("adv_cap", 4))
use_mmr= bool(st.session_state.get("adv_mmr", True))
lam    = float(st.session_state.get("adv_lam", 0.45))
rec_w  = float(st.session_state.get("adv_rec", 0.20))
hlife  = int(st.session_state.get("adv_hl", 270))

with st.spinner("Scanning selected videos…"):
    try:
        hits = dense_search(
            prompt, index, embedder, candidate_vids,
            initial_k=min(K_scan, index.ntotal if index is not None else K_scan),
            final_k=K_use, max_videos=max_v, per_video_cap=cap_v,
            use_mmr=use_mmr, lam=lam, recency_weight=rec_w, half_life_days=hlife, vm=vm
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Search failed."); st.exception(e); st.stop()

# Web
web_snips=[]
if allow_web and selected_domains and int(max_web_auto)>0:
    with st.spinner("Fetching trusted websites…"):
        web_snips = fetch_trusted_snippets(prompt, selected_domains, max_snippets=int(max_web_auto), per_domain=1)

# Evidence assembly (fallback bullets=2)
group_block, export_struct = build_grouped_evidence(prompt, hits, vm, summaries, embedder, per_video_cap=cap_v, fallback_bullets=2)

with st.chat_message("assistant"):
    if not export_struct["videos"] and not web_snips:
        st.warning("No relevant evidence found.")
        st.session_state["messages"].append({"role":"assistant","content":"No relevant evidence found."})
        cols = st.columns([1]*12)
        with cols[-1]:
            st.button("Clear chat", key="clear_nohits", on_click=_clear_chat)
        st.stop()

    with st.spinner("Writing your answer…"):
        ans = openai_answer(model_choice, prompt, st.session_state["messages"], group_block, web_snips, no_video=(len(export_struct["videos"])==0))

    # Answer
    st.markdown(ans)
    st.session_state["messages"].append({"role":"assistant","content":ans})

    # Persist per-turn sources
    st.session_state["turns"].append({
        "prompt": prompt,
        "answer": ans,
        "videos": export_struct["videos"],
        "web": web_snips,
        "web_trace": st.session_state.get("web_trace",""),
        "router": {"domain": domain, "prob": dprob, "tags": sorted(list(tags))[:12]}
    })

    # Sources UI for this reply
    with st.expander("Sources for this reply", expanded=True):
        vids = export_struct["videos"]
        if vids:
            st.markdown("**Video quotes**")
            for v in vids:
                st.markdown(f"- **{v['title']}** — _{v['creator']}_ • {v['ts']} • [{v['url']}]({v['url']})")
                st.markdown(f"  • “{_normalize_text(v['text'])[:220]}”")
        if web_snips:
            st.markdown("**Trusted websites**")
            for j, s in enumerate(web_snips, 1):
                st.markdown(f"W{j}. [{s['domain']}]({s['url']})")

        # Self-check (diagnostic)
        routed_creators = sorted({(_canonicalize_creator(_raw_creator_of_vid(vid, vm)) or _raw_creator_of_vid(vid, vm)) for vid in candidate_vids})
        st.caption(
            f"Self-check: videos routed={len(candidate_vids)} — experts covered={len(routed_creators)} — "
            f"quotes used={len(vids)} — web selected={len(web_snips)} — router=({domain or 'none'}, p={dprob:.2f}) — tags={', '.join(sorted(list(tags))[:10])} "
            f"— web trace: {st.session_state.get('web_trace','none')}"
        )

# Footer / clear
cols = st.columns([1]*12)
with cols[-1]:
    st.button("Clear chat", key="clear_done", on_click=_clear_chat)
