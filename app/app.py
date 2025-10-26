# app/app.py
# -*- coding: utf-8 -*-
"""
Longevity / Nutrition Q&A — experts-first RAG with trusted web support.

Core:
- Stage A routing: pick ~5 best videos via centroid + summary bullets + recency.
- Stage B recall: scan ALL chunks from the routed videos only, not random K.
- Quote filters: minimum length, ends cleanly, contains query tokens or passes a semantic threshold.
- Follow-ups: each user turn gets its own answer + its own Sources block; earlier sources remain below.
- Trusted sites: short vetted snippets are added as supporting evidence.
- Admin: precompute freshness, rebuild, repair norms, fallback summaries, creator inventory, term scan.

Assumptions:
- DATA_DIR layout:
  /data/chunks/chunks.jsonl
  /data/index/faiss.index
  /data/index/metas.pkl
  /data/index/video_centroids.npy
  /data/index/video_ids.txt
  /data/catalog/video_meta.json
  /data/catalog/video_summaries.json

This file is drop-in safe.
"""

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS","1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")

from pathlib import Path
import sys, json, pickle, time, re, math, collections
from typing import List, Dict, Any, Set, Tuple
from datetime import datetime

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Optional web fetch
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests=None; BeautifulSoup=None

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(os.getenv("DATA_DIR","/var/data")).resolve()
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
WEB_FALLBACK = os.getenv("WEB_FALLBACK","true").strip().lower() in {"1","true","yes","on"}

# ---------------- Trusted domains ----------------
TRUSTED_DOMAINS = [
    "nih.gov","medlineplus.gov","cdc.gov","mayoclinic.org","health.harvard.edu",
    "familydoctor.org","healthfinder.gov","ama-assn.org","medicalxpress.com",
    "sciencedaily.com","nejm.org","med.stanford.edu","icahn.mssm.edu"
]

# ---------------- Experts allow-list + synonyms ----------------
ALLOWED_CREATORS = [
    "Dr. Pradip Jamnadas, MD",
    "Andrew Huberman",
    "Healthy Immune Doc",
    "Peter Attia MD",
    "The Diary of A CEO",
]
# remove these exact combined labels if they show up in metadata
EXCLUDED_CREATORS_EXACT = {
    "they diary of a ceo and louse tomlinson",
    "dr. pradip jamnadas, md and the primal podcast",
}
CREATOR_SYNONYMS = {
    # normalize common typos/variants
    "heathy immune doc": "Healthy Immune Doc",
    "heathly immune doc": "Healthy Immune Doc",
    "healthy immune doc": "Healthy Immune Doc",
    "healthy  immune  doc": "Healthy Immune Doc",
    "healthy immune doc youtube": "Healthy Immune Doc",
    "healthy immune doc ": "Healthy Immune Doc",
    "the diary of a ceo": "The Diary of A CEO",
}

# ---------------- Small utils ----------------
def _normalize_text(s:str)->str:
    return re.sub(r"\s+"," ",(s or "").strip())

def _parse_ts(v)->float:
    if isinstance(v,(int,float)):
        try:return float(v)
        except:return 0.0
    try:
        sec=0.0
        for p in str(v).split(":"): sec=sec*60+float(p)
        return sec
    except: return 0.0

def _iso_to_epoch(iso:str)->float:
    if not iso: return 0.0
    try:
        if "T" in iso: return datetime.fromisoformat(iso.replace("Z","+00:00")).timestamp()
        return datetime.fromisoformat(iso).timestamp()
    except: return 0.0

def _format_ts(sec:float)->str:
    sec = int(max(0,float(sec))); h,r=divmod(sec,3600); m,s=divmod(r,60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def _file_mtime(p:Path)->float:
    try:return p.stat().st_mtime
    except:return 0.0

def _iso(ts: float) -> str:
    try: return datetime.fromtimestamp(ts).isoformat()
    except Exception: return "n/a"

def _clear_chat():
    st.session_state["messages"]=[]
    st.session_state["turns"]=[]
    st.rerun()

# ---------------- Admin gate ----------------
def _is_admin()->bool:
    # enable with ?admin=1; optional ?key=... if ADMIN_KEY present in secrets
    try: qp = st.query_params
    except Exception: return False
    if qp.get("admin","0")!="1": return False
    try: expected = st.secrets["ADMIN_KEY"]
    except Exception: expected = None
    if expected is None: return True
    return qp.get("key","")==str(expected)

# ---------------- Creator normalization ----------------
def canonicalize_creator(name: str) -> str | None:
    n = (name or "").strip()
    if not n: return None
    low = re.sub(r"\s+", " ", n.lower()).strip()
    low = low.replace("™","").replace("®","").strip()
    if low in EXCLUDED_CREATORS_EXACT: return None
    low = CREATOR_SYNONYMS.get(low, low)
    for canon in ALLOWED_CREATORS:
        if low == canon.lower(): return canon
    strip_punct = re.sub(r"[^\w\s]", "", low)
    for canon in ALLOWED_CREATORS:
        if strip_punct == re.sub(r"[^\w\s]", "", canon.lower()):
            return canon
    return None

def _raw_creator_of_vid(vid:str, vm:dict)->str:
    info = vm.get(vid, {}) or {}
    for k in ("podcaster","channel","author","uploader","owner","creator"):
        if k in info and info[k]: return str(info[k])
    for k,v in ((kk.lower(), vv) for kk,vv in info.items()):
        if k in {"podcaster","channel","author","uploader","owner","creator"} and v:
            return str(v)
    return "Unknown"

def _vid_epoch(vm:dict, vid:str)->float:
    info=(vm or {}).get(vid,{})
    return _iso_to_epoch(info.get("published_at") or info.get("publishedAt") or info.get("date") or "")

# ---------------- Offsets + quick line index ----------------
@st.cache_data(show_spinner=False, hash_funcs={Path:_file_mtime})
def _ensure_offsets()->np.ndarray:
    # builds a seek index for chunks.jsonl so we can random-access lines by FAISS ids
    if OFFSETS_NPY.exists():
        try:
            arr=np.load(OFFSETS_NPY)
            saved=len(arr); cur=sum(1 for _ in CHUNKS_PATH.open("rb"))
            if cur<=saved: return arr
        except: pass
    pos=0; offs=[]
    with CHUNKS_PATH.open("rb") as f:
        for ln in f:
            offs.append(pos); pos+=len(ln)
    arr=np.array(offs,dtype=np.int64)
    OFFSETS_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(OFFSETS_NPY, arr)
    return arr

def iter_jsonl_rows(indices:List[int], limit:int|None=None):
    if not CHUNKS_PATH.exists(): return
    offs=_ensure_offsets()
    want=[i for i in indices if 0<=i<len(offs)]
    if limit is not None: want=want[:limit]
    with CHUNKS_PATH.open("rb") as f:
        for i in want:
            f.seek(int(offs[i])); raw=f.readline()
            try: yield i, json.loads(raw)
            except: continue

@st.cache_data(show_spinner=False, hash_funcs={Path:_file_mtime})
def build_line_index_by_video() -> Dict[str, List[int]]:
    """
    One-time pass over chunks.jsonl to build mapping: video_id -> list of line indices.
    Enables scanning ALL chunks from routed videos quickly.
    """
    mapping: Dict[str,List[int]] = {}
    if not CHUNKS_PATH.exists(): return mapping
    # offsets array length equals number of lines
    offs = _ensure_offsets()
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f):
            try: j=json.loads(raw)
            except: continue
            m = (j.get("meta") or {})
            vid=(m.get("video_id") or m.get("vid") or m.get("ytid") or
                 j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
            if not vid: continue
            mapping.setdefault(vid, []).append(line_idx)
    return mapping

# ---------------- Model + FAISS ----------------
@st.cache_resource(show_spinner=False)
def _load_embedder(model_name:str)->SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")

@st.cache_resource(show_spinner=False)
def load_metas_and_model(index_path:Path=INDEX_PATH, metas_path:Path=METAS_PKL):
    if not index_path.exists() or not metas_path.exists(): return None, None, None
    index = faiss.read_index(str(index_path))
    with metas_path.open("rb") as f:
        payload=pickle.load(f)
    model_name = payload.get("model","sentence-transformers/all-MiniLM-L6-v2")
    local_dir = DATA_ROOT/"models"/"all-MiniLM-L6-v2"
    try_name = str(local_dir) if (local_dir/"config.json").exists() else model_name
    embedder = _load_embedder(try_name)
    if index.d != embedder.get_sentence_embedding_dimension():
        raise RuntimeError(f"Embedding dim mismatch: FAISS={index.d} vs Encoder={embedder.get_sentence_embedding_dimension()}. Rebuild.")
    return index, payload.get("metas",[]), {"model_name":try_name, "embedder":embedder}

@st.cache_resource(show_spinner=False)
def load_video_centroids():
    if not (VID_CENT_NPY.exists() and VID_IDS_TXT.exists()): return None, None
    C = np.load(VID_CENT_NPY).astype("float32")
    vids = VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
    if C.shape[0]!=len(vids): return None, None
    return C, vids

@st.cache_data(show_spinner=False)
def load_video_meta()->Dict[str,Dict[str,Any]]:
    if VIDEO_META_JSON.exists():
        try:return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except:return {}
    return {}

@st.cache_data(show_spinner=False)
def load_video_summaries():
    if VID_SUM_JSON.exists():
        try:return json.loads(VID_SUM_JSON.read_text(encoding="utf-8"))
        except:return {}
    return {}

# ---------------- Diversity (MMR) ----------------
def mmr(qv:np.ndarray, doc_vecs:np.ndarray, k:int, lambda_diversity:float=0.45)->List[int]:
    if doc_vecs.size==0: return []
    sim=(doc_vecs @ qv.reshape(-1,1)).ravel()
    sel=[]; cand=set(range(doc_vecs.shape[0]))
    while cand and len(sel)<k:
        if not sel:
            cl=list(cand); pick=cl[int(np.argmax(sim[cl]))]
            sel.append(pick); cand.remove(pick); continue
        sv=doc_vecs[sel]; cl=list(cand)
        max_div=(sv @ doc_vecs[cl].T).max(axis=0)
        scores=lambda_diversity*sim[cl] - (1-lambda_diversity)*max_div
        pick=cl[int(np.argmax(scores))]
        sel.append(pick); cand.remove(pick)
    return sel

def _dedupe_passages(items:List[Dict[str,Any]], time_window_sec:float=8.0, min_chars:int=60):
    # remove near-duplicate quotes from the same timestamp neighborhood
    out=[]; seen=[]
    for h in sorted(items, key=lambda r: float((r.get("meta") or {}).get("start",0))):
        ts=float((h.get("meta") or {}).get("start",0))
        txt=_normalize_text(h.get("text",""))
        if len(txt)<min_chars: continue
        if any(abs(ts - float((s.get("meta") or {}).get("start",0)))<=time_window_sec and _normalize_text(s.get("text",""))==txt for s in seen):
            continue
        seen.append(h); out.append(h)
    return out

# ---------------- Summary-aware routing ----------------
def _build_idf_over_bullets(summaries: dict) -> dict:
    DF = collections.Counter()
    vids = list(summaries.keys())
    for v in vids:
        for b in summaries.get(v, {}).get("bullets", []):
            toks = set((b.get("text","") or "").lower().split())
            for w in toks: DF[w] += 1
    N = max(1, len(vids))
    return {w: math.log((N+1)/(df+0.5)) for w,df in DF.items()}

def _kw_score(text: str, query: str, idf: dict) -> float:
    if not text: return 0.0
    q = [w for w in (query or "").lower().split() if w]
    t = (text or "").lower().split()
    tf = {w: t.count(w) for w in set(t)}
    return sum(tf.get(w,0) * idf.get(w,0.0) for w in set(q)) / (len(t)+1e-6)

def _recency_score(published_ts:float, now:float, half_life_days:float)->float:
    if published_ts<=0: return 0.0
    days=max(0.0,(now-published_ts)/86400.0)
    return 0.5 ** (days / max(1e-6, half_life_days))

def route_videos_by_summary(
    query: str, qv: np.ndarray,
    summaries: dict, vm: dict,
    C: np.ndarray | None, vids: list[str] | None,
    allowed_vids: set[str],
    topK: int, recency_weight: float, half_life_days: float
) -> list[str]:
    # Candidate universe limited to selected experts
    universe = [v for v in (vids or list(vm.keys())) if (not allowed_vids or v in allowed_vids)]
    if not universe: return []
    # Optional centroid similarity if precomputed vector per video exists
    cent = {}
    if C is not None and vids is not None and len(vids) == C.shape[0]:
        sim = (C @ qv.reshape(-1,1)).ravel()
        cent = {vids[i]: float(sim[i]) for i in range(len(vids))}
    # Keyword score over summary bullets
    idf = _build_idf_over_bullets(summaries)
    now = time.time()
    scored = []
    for v in universe:
        bullets = summaries.get(v, {}).get("bullets", [])
        pseudo = " ".join(b.get("text","") for b in bullets)[:2000]
        kw = _kw_score(pseudo, query, idf)
        cs = cent.get(v, 0.0)
        rec = _recency_score(_vid_epoch(vm, v), now, half_life_days)
        base = 0.6*cs + 0.3*kw
        score = (1.0 - recency_weight)*base + recency_weight*(0.1*rec + 0.9*base)
        scored.append((v, score))
    scored.sort(key=lambda x:-x[1])
    return [v for v,_ in scored[:int(topK)]]

# ---------------- Creator inventory from chunks ----------------
def build_creator_indexes_from_chunks(vm: dict) -> tuple[dict, dict]:
    """Return (vid_to_creator, creator_to_vids) using chunks.jsonl first, else vm."""
    vid_to_creator: Dict[str,str] = {}
    creator_to_vids: Dict[str,set] = {}
    if CHUNKS_PATH.exists():
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            for ln in f:
                try: j=json.loads(ln)
                except: continue
                m = (j.get("meta") or {})
                vid = (m.get("video_id") or m.get("vid") or m.get("ytid") or
                       j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
                if not vid: continue
                raw = (m.get("channel") or m.get("author") or m.get("uploader") or
                       _raw_creator_of_vid(vid, vm))
                canon = canonicalize_creator(raw)
                if canon is None: continue
                if vid not in vid_to_creator:
                    vid_to_creator[vid] = canon
                    creator_to_vids.setdefault(canon, set()).add(vid)
    # fill any missing from vm
    for vid in vm.keys():
        if vid in vid_to_creator: continue
        canon = canonicalize_creator(_raw_creator_of_vid(vid, vm))
        if canon is None: continue
        vid_to_creator[vid]=canon
        creator_to_vids.setdefault(canon,set()).add(vid)
    return vid_to_creator, creator_to_vids

# ---------------- Quote quality filters ----------------
def _token_overlap_ok(text:str, query:str)->bool:
    qt = [w for w in re.findall(r"[A-Za-z0-9\-]+", (query or "").lower()) if len(w)>=3]
    if not qt: return False
    t = (text or "").lower()
    return any(w in t for w in qt)

def _ends_cleanly(text:str)->bool:
    return bool(re.search(r"[\.!?…]\"?$", text.strip()))

def _semantic_ok(qv:np.ndarray, tv:np.ndarray, thresh:float=0.35)->bool:
    s = float(np.dot(qv, tv))
    return s >= thresh

def quote_is_valid(text:str, query:str, qv:np.ndarray, tv:np.ndarray)->bool:
    txt=_normalize_text(text)
    if len(txt) < 80: return False
    if not _ends_cleanly(txt): return False
    if _token_overlap_ok(txt, query): return True
    return _semantic_ok(qv, tv)

# ---------------- Stage B: scan ALL chunks from routed videos ----------------
def scan_all_chunks_in_videos(
    query:str,
    embedder:SentenceTransformer,
    candidate_vids:Set[str],
    max_videos:int,
    per_video_cap:int,
    recency_weight:float,
    half_life_days:float,
    vm:dict)->List[Dict[str,Any]]:

    if not candidate_vids: return []
    line_index = build_line_index_by_video()
    # order videos by simple recency to start
    now=time.time()
    vids_ordered = sorted(list(candidate_vids),
                          key=lambda v: _recency_score(_vid_epoch(vm,v), now, half_life_days),
                          reverse=True)[:max_videos]

    # collect all rows for these videos
    indices=[]
    for vid in vids_ordered:
        indices.extend(line_index.get(vid, []))
    if not indices: return []

    # read rows
    texts=[]; metas=[]; vids=[]
    for _, j in iter_jsonl_rows(indices):
        t=_normalize_text(j.get("text",""))
        if not t: continue
        m=(j.get("meta") or {}).copy()
        vid=(m.get("video_id") or m.get("vid") or m.get("ytid") or
             j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
        if not vid or vid not in candidate_vids: continue
        if "start" not in m and "start_sec" in m: m["start"]=m.get("start_sec")
        m["start"]=_parse_ts(m.get("start",0))
        m["video_id"]=vid
        texts.append(t); metas.append(m); vids.append(vid)

    if not texts: return []
    # embed all passages for validity and MMR selection
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
    doc_vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")

    # initial scoring = semantic sim + small recency boost
    now=time.time()
    base_sim = (doc_vecs @ qv.reshape(-1,1)).ravel()
    rec = np.array([_recency_score(_vid_epoch(vm, v), now, half_life_days) for v in vids], dtype="float32")
    score = 0.9*base_sim + 0.1*rec

    # filter by quote quality
    keep=[]
    for i,(t,m,vv) in enumerate(zip(texts, metas, vids)):
        if quote_is_valid(t, query, qv, doc_vecs[i]):
            keep.append((i, float(score[i])))
    if not keep: return []

    # per-video cap using best-first within each video
    by_vid: Dict[str, List[Tuple[int,float]]] = {}
    for i, sc in keep:
        by_vid.setdefault(vids[i], []).append((i, sc))
    for v in by_vid:
        by_vid[v].sort(key=lambda x:-x[1])

    picked=[]
    # round-robin to balance quotes across videos
    for round_idx in range(per_video_cap):
        for v in vids_ordered:
            lst = by_vid.get(v, [])
            if round_idx < len(lst):
                i, sc = lst[round_idx]
                picked.append({"i":indices[i],
                               "score":float(sc),
                               "text":texts[i],
                               "meta":metas[i]})
    return picked

# ---------------- Grouped evidence block for LLM ----------------
def group_hits_by_video(hits:List[Dict[str,Any]])->Dict[str,List[Dict[str,Any]]]:
    g={}
    for h in hits:
        vid=(h.get("meta") or {}).get("video_id") or "Unknown"
        g.setdefault(vid,[]).append(h)
    return g

def build_grouped_evidence_for_prompt(hits:List[Dict[str,Any]], vm:dict, summaries:dict, max_quotes:int=3)->str:
    groups=group_hits_by_video(hits)
    ordered = sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)
    lines=[]
    for v_idx,(vid,items) in enumerate(ordered,1):
        info=vm.get(vid,{})
        title=info.get("title") or summaries.get(vid,{}).get("title") or vid
        creator_raw=_raw_creator_of_vid(vid, vm)
        creator=canonicalize_creator(creator_raw) or creator_raw
        date=info.get("published_at") or info.get("publishedAt") or info.get("date") or summaries.get(vid,{}).get("published_at") or ""
        lines.append(f"[Video {v_idx}] {title} — {creator}" + (f" — {date}" if date else ""))

        # prefer summary bullets; backfill with best quotes
        bullets = summaries.get(vid,{}).get("bullets", [])
        for b in bullets[:max_quotes]:
            ts=_format_ts(b.get("ts",0))
            q=_normalize_text(b.get("text","")); q=q[:260]+"…" if len(q)>260 else q
            lines.append(f"  • {ts}: “{q}”")

        need = max(0, max_quotes - len(bullets))
        if need>0:
            clean=_dedupe_passages(items, time_window_sec=8.0, min_chars=60)
            for h in clean[:need]:
                ts=_format_ts((h.get("meta") or {}).get("start",0))
                q=_normalize_text(h.get("text","")); q=q[:260]+"…" if len(q)>260 else q
                lines.append(f"  • {ts}: “{q}”")
        lines.append("")
    return "\n".join(lines).strip()

# ---------------- Web fetch ----------------
def _ddg_domain_search(domain:str, query:str, headers:dict, timeout:float):
    try:
        r=requests.get("https://duckduckgo.com/html/", params={"q":f"site:{domain} {query}"}, headers=headers, timeout=timeout)
        if r.status_code!=200: return []
        soup=BeautifulSoup(r.text,"html.parser")
        return [a.get("href") for a in soup.select("a.result__a") if a.get("href") and domain in a.get("href")]
    except Exception:
        return []

def fetch_trusted_snippets(query:str, allowed_domains:List[str], max_snippets:int=3, per_domain:int=1, timeout:float=6.0):
    if not requests or not BeautifulSoup or max_snippets<=0: return []
    headers={"User-Agent":"Mozilla/5.0"}
    out=[]; seen=set()
    for domain in allowed_domains:
        links=_ddg_domain_search(domain, query, headers, timeout)
        if not links: links=[f"https://{domain}"]  # minimal fallback
        links=links[:per_domain]
        for url in links:
            if url in seen: continue
            try:
                r=requests.get(url, headers=headers, timeout=timeout)
                if r.status_code!=200: continue
                soup=BeautifulSoup(r.text,"html.parser")
                paras=[p.get_text(" ",strip=True) for p in soup.find_all("p")]
                text=_normalize_text(" ".join(paras))[:2000]
                if len(text)<200: continue
                out.append({"domain":domain,"url":url,"text":text}); seen.add(url)
            except: continue
        if len(out)>=max_snippets: break
    return out[:max_snippets]

# ---------------- LLM ----------------
def openai_answer(model_name: str, question: str, history, grouped_video_block: str,
                  web_snips: list[dict], no_video: bool) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ OPENAI_API_KEY is not set."
    # last 2 Q&A pairs for follow-up context
    recent = [m for m in history[-6:] if m.get("role") in ("user","assistant")]
    convo = [("User: " if m["role"]=="user" else "Assistant: ")+m.get("content","") for m in recent]

    web_lines = [
        f"(W{j}) {s.get('domain','web')} — {s.get('url','')}\n“{' '.join((s.get('text','')).split())[:300]}”"
        for j,s in enumerate(web_snips,1)
    ]
    web_block = "\n".join(web_lines) if web_lines else "None"

    fallback_line = ("If no suitable video evidence exists, you MAY answer from trusted web snippets alone, "
                     "but begin with: 'Web-only evidence'.\n") if (WEB_FALLBACK and no_video) else \
                    "Trusted web snippets are supporting evidence.\n"

    system = (
        "Answer from the provided evidence plus trusted web sources. Priority: (1) grouped VIDEO evidence from selected experts, "
        "(2) trusted WEB snippets.\n" + fallback_line +
        "Rules:\n"
        "• Cite every claim/step: (Video k) for videos, (DOMAIN Wj) for web.\n"
        "• Prefer human clinical data; label animal/in-vitro/mechanistic explicitly.\n"
        "• Normalize units and report numeric effect sizes when provided (%, mg/dL, mmol/L, ApoB).\n"
        "• List therapeutic OPTIONS by class and drug names; include mechanism and typical magnitude; "
        "if dose missing, write 'dose not specified'. No diagnosis.\n"
        "Structure: Key summary • Practical protocol • Safety notes. Be concise and source-grounded.\n"
        "Use only quoted bullets and chunk quotes; do not invent claims."
    )

    user_payload = (("Recent conversation:\n"+"\n".join(convo)+"\n\n") if convo else "") + \
                   f"Question: {question}\n\n" + \
                   "Grouped Video Evidence:\n" + (grouped_video_block or "None") + "\n\n" + \
                   "Trusted Web Snippets:\n" + web_block + "\n\n" + \
                   "Write a concise, well-grounded answer."

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

# ---------------- Precompute helpers (admin) ----------------
def _run_precompute_inline()->str:
    try:
        try:
            from scripts import precompute_video_summaries as pvs  # type: ignore
            pvs.main()
            return "ok: rebuilt via package import"
        except Exception:
            import importlib.util
            p = Path(__file__).resolve().parents[1] / "scripts" / "precompute_video_summaries.py"
            spec = importlib.util.spec_from_file_location("pvs_fallback", str(p))
            if spec is None or spec.loader is None:
                return f"precompute error: cannot load {p}"
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            mod.main()
            return "ok: rebuilt via file loader"
    except Exception as e:
        return f"precompute error: {e}"

def _repair_centroids_in_place()->str:
    try:
        if not VID_CENT_NPY.exists(): return "centroids file missing"
        C = np.load(VID_CENT_NPY).astype("float32")
        if C.ndim != 2 or C.size == 0: return "centroids shape invalid"
        n = np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        C = C / n
        np.save(VID_CENT_NPY, C)
        return "ok: renormalized"
    except Exception as e:
        return f"repair error: {e}"

def _build_summaries_fallback(max_lines_per_video: int = 800) -> str:
    """Create video_summaries.json directly from chunks.jsonl (light TF-IDF bullets)."""
    try:
        if not CHUNKS_PATH.exists(): return "chunks.jsonl missing"
        vm = load_video_meta()

        texts_by_vid: Dict[str, List[str]] = {}
        ts_by_vid: Dict[str, List[float]] = {}
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            for ln in f:
                try: j = json.loads(ln)
                except: continue
                t = _normalize_text(j.get("text",""))
                m = (j.get("meta") or {})
                vid = (m.get("video_id") or m.get("vid") or m.get("ytid") or
                       j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
                if not vid or not t: continue
                ts = _parse_ts(m.get("start", m.get("start_sec", 0)))
                texts_by_vid.setdefault(vid, []).append(t)
                ts_by_vid.setdefault(vid, []).append(float(ts))

        vids = list(texts_by_vid.keys())
        if not vids: return "no videos detected in chunks.jsonl"

        DF = collections.Counter()
        for vid in vids:
            seen=set()
            for t in texts_by_vid[vid]:
                for w in set(t.lower().split()):
                    if w not in seen:
                        DF[w]+=1; seen.add(w)
        N = max(1,len(vids))
        def score_line(t:str)->float:
            words=t.lower().split()
            tf=collections.Counter(words)
            return sum(tf[w]*math.log((N+1)/(DF.get(w,1)+0.5)) for w in tf)/(len(words)+1e-6)

        summaries={}
        for vid in vids:
            lines=texts_by_vid[vid][:max_lines_per_video]
            times=ts_by_vid[vid][:max_lines_per_video]
            idx_scores=[(i,score_line(t)) for i,t in enumerate(lines)]
            top=sorted([i for i,_ in sorted(idx_scores,key=lambda x:-x[1])[:12]])[:10]
            bullets=[{"ts":float(times[i]),
                      "text":(lines[i][:280]+"…" if len(lines[i])>280 else lines[i])} for i in top[:6]]
            summary=" ".join(lines[i] for i in top[:6])
            if len(summary)>1200: summary=summary[:1200]+"…"
            info=vm.get(vid,{})
            summaries[vid]={"title":info.get("title",""),
                            "channel":info.get("channel",""),
                            "published_at":info.get("published_at") or info.get("publishedAt") or info.get("date") or "",
                            "bullets":bullets,"summary":summary}

        VID_SUM_JSON.parent.mkdir(parents=True, exist_ok=True)
        VID_SUM_JSON.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"ok: wrote {VID_SUM_JSON}"
    except Exception as e:
        return f"summaries fallback error: {e}"

def _path_exists_report()->Dict[str, Any]:
    return {
        "DATA_DIR": str(DATA_ROOT),
        "chunks": str(CHUNKS_PATH), "chunks_exists": CHUNKS_PATH.exists(),
        "centroids": str(VID_CENT_NPY), "centroids_exists": VID_CENT_NPY.exists(),
        "ids": str(VID_IDS_TXT), "ids_exists": VID_IDS_TXT.exists(),
        "summaries": str(VID_SUM_JSON), "summaries_exists": VID_SUM_JSON.exists(),
    }

# ---------------- Verification helper (admin) ----------------
def scan_chunks_for_terms(terms:List[str], vm:Dict[str,Dict[str,Any]], limit_examples:int=200):
    if not CHUNKS_PATH.exists():
        return {"total_matches":0,"per_creator":{},"examples":[]}
    pat = re.compile(r"("+"|".join([re.escape(t) for t in terms]) + r")", re.IGNORECASE)
    total=0; per_creator={}; examples=[]
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for ln in f:
            try:j=json.loads(ln)
            except:continue
            t=j.get("text","") or ""
            if not pat.search(t): continue
            m=(j.get("meta") or {})
            vid=m.get("video_id") or m.get("vid") or m.get("ytid") or j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id") or "Unknown"
            st_sec=_parse_ts(m.get("start", m.get("start_sec",0)))
            creator=_raw_creator_of_vid(vid, vm)
            canon = canonicalize_creator(creator)
            label = canon if canon else creator
            per_creator[label]=per_creator.get(label,0)+1
            total+=1
            if len(examples)<int(limit_examples):
                sn=_normalize_text(t); sn=sn[:260]+"…" if len(sn)>260 else sn
                examples.append({"video_id":vid,"creator":label,"ts":_format_ts(st_sec),"snippet":sn})
    per_creator=dict(sorted(per_creator.items(), key=lambda kv:-kv[1]))
    return {"total_matches":total,"per_creator":per_creator,"examples":examples}

# ================= UI =================
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

with st.sidebar:
    st.markdown("**Auto Mode** · optimized for accuracy and diversity")

    # Experts: vertical checklist with video counts
    vm = load_video_meta()
    vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
    counts={canon: len(creator_to_vids.get(canon,set())) for canon in ALLOWED_CREATORS}

    st.subheader("Experts")
    st.caption("Select which expert channels to include. All are on by default.")
    selected_creators_list=[]
    for i, canon in enumerate(ALLOWED_CREATORS):
        label=f"{canon} ({counts.get(canon,0)})"
        if st.checkbox(label, value=True, key=f"exp_{i}",
                       help="Uncheck to exclude this expert from being used as video evidence. The number shows how many indexed videos are available."):
            selected_creators_list.append(canon)
    selected_creators:set[str]=set(selected_creators_list)
    st.session_state["selected_creators"]=selected_creators  # persist for follow-ups

    # Trusted sites: vertical checklist, all on by default
    st.subheader("Trusted sites")
    st.caption("Short excerpts from vetted medical sites are added as supporting evidence. They should confirm or contextualize the video claims.")
    allow_web = st.checkbox(
        "Include supporting website excerpts", value=True,
        help="When enabled, the app fetches brief paragraphs from the checked medical domains and blends them with expert videos."
    )
    selected_domains=[]
    for i,dom in enumerate(TRUSTED_DOMAINS):
        if st.checkbox(dom, value=True, key=f"site_{i}",
                       help="Keep checked to allow short corroborating excerpts from this domain."):
            selected_domains.append(dom)

    # Model choice
    model_choice = st.selectbox(
        "Answering model", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0,
        help="Controls how the final answer is written from the evidence."
    )

    # Advanced knobs hidden unless needed
    with st.expander("Advanced (optional)"):
        st.caption("Defaults are tuned. Adjust only if needed.")
        # routing
        recency_weight_adv = st.slider("Routing: recency weight", 0.0, 1.0, 0.20, 0.05)
        half_life_adv = st.slider("Routing: half-life (days)", 7, 720, 270, 7)
        topK_route_adv = st.slider("Routing: videos to consider", 3, 8, 5, 1)
        # recall caps
        max_videos_adv = st.slider("Evidence: max videos", 3, 8, 5, 1)
        per_video_cap_adv = st.slider("Evidence: passages per video", 2, 6, 4, 1)
        mmr_lambda_adv = st.slider("Diversity: MMR balance", 0.1, 0.9, 0.45, 0.05)
        use_mmr_adv = st.checkbox("Use MMR", True)

    st.divider()
    show_diag = st.toggle(
        "Show data diagnostics", value=False,
        help="Check file presence, modification times, and index health without a shell."
    )

    st.subheader("Library status")
    st.checkbox("chunks.jsonl present", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("offsets built", value=OFFSETS_NPY.exists(), disabled=True)
    st.checkbox("faiss.index present", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("metas.pkl present", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("video_meta.json present", value=VIDEO_META_JSON.exists(), disabled=True)
    cent_ready = VID_CENT_NPY.exists() and VID_IDS_TXT.exists() and VID_SUM_JSON.exists()
    st.checkbox("centroids & summaries present", value=cent_ready, disabled=True)

# Diagnostics area
if show_diag:
    colA,colB,colC = st.columns([2,3,3])
    with colA: st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    with colB: st.caption(f"chunks mtime: {_iso(_file_mtime(CHUNKS_PATH)) if CHUNKS_PATH.exists() else 'missing'}")
    with colC: st.caption(f"index mtime: {_iso(_file_mtime(INDEX_PATH)) if INDEX_PATH.exists() else 'missing'}")

# Keep multi-turn message list and per-turn sources
if "messages" not in st.session_state: st.session_state.messages=[]
if "turns" not in st.session_state: st.session_state.turns=[]  # each turn: {"q": str, "a": str, "sources": export_json}

# Render prior conversation with their own sources blocks
for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["a"])
        with st.expander("Sources & timestamps", expanded=False):
            src = turn.get("sources", {})
            # videos
            for v in src.get("videos", []):
                header = f"- **{v.get('title','')}**" + (f" — _{v.get('creator','')}_ " if v.get("creator") else "")
                url = v.get("url") or ""
                st.markdown(f"{header} [{'(link)'}]({url})" if url else f"{header}")
                for q in v.get("quotes", []):
                    st.markdown(f"  • **{q.get('ts','')}** — “{q.get('text','')}”")
            # web
            if src.get("web"):
                st.markdown("**Trusted websites**")
                for w in src["web"]:
                    st.markdown(f"{w.get('id','')} [{w.get('domain','')}]({w.get('url','')})")
            st.download_button(
                "Download sources as JSON",
                data=json.dumps(src, ensure_ascii=False, indent=2),
                file_name="sources.json",
                mime="application/json",
            )

# Admin diagnostics pane
if _is_admin():
    st.subheader("Diagnostics (admin)")
    try:
        _,_,payload = load_metas_and_model()
        emb_model_name = payload["model_name"]
        # minimal status
        C_ok = VID_CENT_NPY.exists()
        I_ok = VID_IDS_TXT.exists()
        S_ok = VID_SUM_JSON.exists()
        st.metric("Centroids", "Yes" if C_ok else "No")
        st.metric("IDs", "Yes" if I_ok else "No")
        st.metric("Summaries", "Yes" if S_ok else "No")
    except Exception as e:
        st.error(f"Load error: {e}")

    st.markdown("---")
    st.subheader("Files on disk")
    st.code(json.dumps(_path_exists_report(), indent=2))

    cols_dbg = st.columns(3)
    with cols_dbg[0]:
        if st.button("Rebuild precompute (admin)"):
            with st.spinner("Building centroids and summaries…"):
                msg=_run_precompute_inline()
            st.success(str(msg)); st.cache_resource.clear(); st.cache_data.clear(); st.rerun()
    with cols_dbg[1]:
        if st.button("Repair centroids norms"):
            with st.spinner("Renormalizing centroids…"):
                msg=_repair_centroids_in_place()
            st.success(msg); st.cache_resource.clear(); st.cache_data.clear(); st.rerun()
    with cols_dbg[2]:
        st.caption(f"summaries path: {VID_SUM_JSON}")
        if st.button("Build summaries now (fallback)"):
            with st.spinner("Generating summaries from chunks.jsonl…"):
                msg=_build_summaries_fallback()
            st.success(msg); st.cache_resource.clear(); st.cache_data.clear(); st.rerun()

    # Creator inventory
    with st.expander("Creator inventory (from chunks.jsonl)"):
        inv = sorted(((c, len(vs)) for c,vs in build_creator_indexes_from_chunks(load_video_meta())[1].items()),
                     key=lambda x: -x[1])
        st.dataframe([{"creator": c, "videos": n} for c,n in inv], use_container_width=True)

    st.markdown("---")
    st.caption("Keyword coverage scan (verification only).")
    default_terms = "apob, apo-b, ldl, statin, ezetimibe, pcsk9, bempedoic, inclisiran, niacin"
    term_input = st.text_input("Terms (comma-separated)", default_terms)
    if st.button("Run scan"):
        terms=[t.strip() for t in term_input.split(",") if t.strip()]
        with st.spinner("Scanning transcripts…"):
            scan=scan_chunks_for_terms(terms=terms, vm=load_video_meta(), limit_examples=300)
        st.metric("Total matching chunks", scan["total_matches"])
        if scan["per_creator"]: st.dataframe([{"expert":k,"matching_chunks":v} for k,v in scan["per_creator"].items()], use_container_width=True)
        if scan["examples"]: st.dataframe(scan["examples"], use_container_width=True)
    st.markdown("---")

# Prompt input (new turn)
prompt = st.chat_input("Ask about ApoB/LDL drugs, sleep, protein timing, fasting, supplements, protocols…")
if prompt is None:
    cols=st.columns([1]*12)
    with cols[-1]:
        st.button("Clear chat", key="clear_idle", help="Reset conversation.", on_click=_clear_chat)
    st.stop()

# Display user message for current turn
with st.chat_message("user"): st.markdown(prompt)

# Guardrails
missing=[p for p in REQUIRED if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error("Missing required files:\n" + "\n".join(f"- {p}" for p in missing))
    st.stop()

# Load FAISS + encoder + precompute artifacts
try:
    index, _, payload = load_metas_and_model()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load index or encoder."); st.exception(e); st.stop()
embedder: SentenceTransformer = payload["embedder"]
vm = load_video_meta()
C, vid_list = load_video_centroids()
summaries = load_video_summaries()

# Selected experts → allowed video universe
vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
universe = set(vid_list or list(vm.keys()) or list(vid_to_creator.keys()))
chosen = st.session_state.get("selected_creators", set(ALLOWED_CREATORS))
allowed_vids = {vid for vid in universe if vid_to_creator.get(vid) in chosen}

# -------- Stage A: routing
# include previous user message lightly for follow-ups
routing_query = prompt
prev_users = [m["content"] for m in st.session_state.messages if m.get("role")=="user"]
if len(prev_users) >= 2:
    routing_query = prev_users[-2] + " ; " + prompt

embed_model = embedder
qv_route = embed_model.encode([routing_query], normalize_embeddings=True).astype("float32")[0]

# Defaults; allow Advanced overrides if user opened it
recency_weight_auto = float(st.session_state.get("Routing: recency weight", 0.20))
half_life_auto = int(st.session_state.get("Routing: half-life (days)", 270))
topK_route = int(st.session_state.get("Routing: videos to consider", 5))

routed_vids = route_videos_by_summary(
    routing_query, qv_route, summaries, vm, C, list(universe), allowed_vids,
    topK=topK_route, recency_weight=recency_weight_auto, half_life_days=half_life_auto
)
candidate_vids = set(routed_vids) if routed_vids else allowed_vids

# -------- Stage B: scan ALL chunks from routed videos, then enforce caps
max_videos = int(st.session_state.get("Evidence: max videos", 5))
per_video_cap = int(st.session_state.get("Evidence: passages per video", 4))
use_mmr = bool(st.session_state.get("Use MMR", True))
mmr_lambda = float(st.session_state.get("Diversity: MMR balance", 0.45))

with st.spinner("Scanning selected videos…"):
    hits_all = scan_all_chunks_in_videos(
        prompt, embedder, candidate_vids, max_videos=max_videos, per_video_cap=per_video_cap,
        recency_weight=recency_weight_auto, half_life_days=half_life_auto, vm=vm
    )

# MMR diversification across picked quotes if enabled
if use_mmr and hits_all:
    texts=[_normalize_text(h["text"]) for h in hits_all]
    qv = embedder.encode([prompt], normalize_embeddings=True).astype("float32")[0]
    doc_vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")
    order = mmr(qv, doc_vecs, k=min(len(texts), max_videos*per_video_cap), lambda_diversity=mmr_lambda)
    hits = [hits_all[i] for i in order]
else:
    hits = hits_all

# Optional web support
web_snips=[]
if allow_web and selected_domains and requests and BeautifulSoup:
    with st.spinner("Fetching trusted websites…"):
        web_snips = fetch_trusted_snippets(prompt, selected_domains, max_snippets=3, per_domain=1)

# Build grouped evidence and answer
grouped_block = build_grouped_evidence_for_prompt(hits, vm, summaries, max_quotes=3)

with st.chat_message("assistant"):
    if not hits and not web_snips:
        st.warning("No relevant evidence found.")
        # Record empty turn so the conversation still shows the question
        st.session_state.turns.append({"q": prompt, "a": "I couldn’t find enough evidence to answer that.", "sources": {"videos":[],"web":[]}})
        cols=st.columns([1]*12)
        with cols[-1]:
            st.button("Clear chat", key="clear_nohits", on_click=_clear_chat)
        st.stop()

    with st.spinner("Writing your answer…"):
        ans=openai_answer(model_choice, prompt, st.session_state.messages + [{"role":"user","content":prompt}], grouped_block, web_snips, no_video=(len(hits)==0))

    # Show answer
    st.markdown(ans)

    # Build per-turn accurate sources JSON from actual hits shown
    groups = group_hits_by_video(hits)
    ordered = sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)
    export = {"videos": [], "web": []}

    for vid, items in ordered:
        info = vm.get(vid, {})
        title = info.get("title") or summaries.get(vid, {}).get("title") or vid
        creator_raw = _raw_creator_of_vid(vid, vm)
        creator = canonicalize_creator(creator_raw) or creator_raw
        url = info.get("url") or ""
        clean = _dedupe_passages(items, time_window_sec=8.0, min_chars=60)

        v = {"video_id": vid, "title": title, "creator": creator, "url": url, "quotes": []}
        for h in clean:
            ts = _format_ts((h.get("meta") or {}).get("start", 0))
            q = _normalize_text(h.get("text", ""))
            if len(q) > 200: q = q[:200] + "…"
            v["quotes"].append({"ts": ts, "text": q})
        export["videos"].append(v)

    if web_snips:
        for j, s in enumerate(web_snips, 1):
            export["web"].append({"id": f"W{j}", "domain": s["domain"], "url": s["url"]})

    # Persist this turn so follow-ups keep earlier sources visible
    st.session_state.turns.append({"q": prompt, "a": ans, "sources": export})

    # Also render its sources block immediately (per your requirement)
    with st.expander("Sources & timestamps", expanded=False):
        st.caption("Each bullet shows a timestamped quote from a video. Websites appear as supporting references.")
        for v in export["videos"]:
            header = f"- **{v.get('title','')}**" + (f" — _{v.get('creator','')}_ " if v.get("creator") else "")
            url = v.get("url") or ""
            st.markdown(f"{header} [{'(link)'}]({url})" if url else f"{header}")
            for q in v.get("quotes", []):
                st.markdown(f"  • **{q.get('ts','')}** — “{q.get('text','')}”")
        if export["web"]:
            st.markdown("**Trusted websites**")
            for w in export["web"]:
                st.markdown(f"{w.get('id','')} [{w.get('domain','')}]({w.get('url','')})")
        st.download_button(
            "Download sources as JSON",
            data=json.dumps(export, ensure_ascii=False, indent=2),
            file_name="sources.json",
            mime="application/json",
        )

# Update global message log for follow-ups
st.session_state.messages.append({"role":"user","content":prompt})
st.session_state.messages.append({"role":"assistant","content":st.session_state.turns[-1]["a"]})

# Footer + Clear chat
st.caption("Routing selects top videos; all chunks from those videos are scanned. Quotes are filtered for completeness and relevance, then fused with trusted medical sites.")
cols=st.columns([1]*12)
with cols[-1]:
    st.button("Clear chat", key="clear_done", on_click=_clear_chat)
