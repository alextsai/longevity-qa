# app/app.py
# -*- coding: utf-8 -*-
"""
Longevity / Nutrition Q&A — experts-first RAG + trusted web support.

What changed (high level)
- Robust creator mapping: heuristics map variants like "Healthy Immune Doc" and "Diary of a CEO".
- Routing guardrails: hybrid score = centroid + summary-keywords; also require minimal keyword overlap.
- Recall that actually yields quotes: softer quote filter + guaranteed ≥1 quote per routed video.
- Per-turn sources: each assistant reply stores its OWN sources; earlier turn sources are kept.
- Web snippets that actually show: dual DuckDuckGo modes + homepage fallback + explicit debug trace.
- Follow-ups: last 2 Q/As kept for LLM context and routing keyword expansion.
- Admin: inventory table, freshness checks, rebuild, renorm, summaries-fallback.
- Comments: every critical function documents assumptions and failure modes.

This file is self-contained and safe to drop in.
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

# Optional web fetch (safe if not installed)
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

# Precompute outputs produced by your offline pipeline
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

# ---------------- Experts allow-list ----------------
ALLOWED_CREATORS = [
    "Dr. Pradip Jamnadas, MD",
    "Andrew Huberman",
    "Healthy Immune Doc",
    "Peter Attia MD",
    "The Diary of A CEO",
]
# Exact labels to drop if they appear as weird combined metadata
EXCLUDED_CREATORS_EXACT = {
    "they diary of a ceo and louse tomlinson",
    "dr. pradip jamnadas, md and the primal podcast",
}

# ---------------- Small utils ----------------
def _normalize_text(s:str)->str:
    return re.sub(r"\s+"," ",(s or "").strip())

def _parse_ts(v)->float:
    """Parse HH:MM:SS or seconds to float seconds."""
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
    st.session_state["messages"]=[]; st.session_state["turns"]=[]; st.rerun()


# ---------------- Admin gate ----------------
def _is_admin()->bool:
    """Enable admin with ?admin=1 and optional ?key=... if ADMIN_KEY set in secrets."""
    try: qp = st.query_params
    except Exception: return False
    if qp.get("admin","0")!="1": return False
    try: expected = st.secrets["ADMIN_KEY"]
    except Exception: expected = None
    if expected is None: return True
    return qp.get("key","")==str(expected)


# ---------------- Creator mapping ----------------
def _raw_creator_of_vid(vid:str, vm:dict)->str:
    """Extract a creator-ish field from video_meta.json for a given video."""
    info = vm.get(vid, {}) or {}
    for k in ("podcaster","channel","author","uploader","owner","creator"):
        if k in info and info[k]: return str(info[k])
    # try case-insensitive keys, some dumps are inconsistent
    for k,v in ((kk.lower(), vv) for kk,vv in info.items()):
        if k in {"podcaster","channel","author","uploader","owner","creator"} and v:
            return str(v)
    return "Unknown"

def _canonicalize_creator(name: str) -> str | None:
    """
    Robust canonicalization.
    - Heuristics instead of fragile synonym tables.
    - Returns one of ALLOWED_CREATORS or None if excluded.
    """
    n = _normalize_text(name).lower().replace("™","").replace("®","")
    if not n: return None
    if n in EXCLUDED_CREATORS_EXACT: return None

    # Heuristic buckets by token presence
    tokens = set(re.findall(r"[a-z0-9]+", n))

    # Healthy Immune Doc
    if ("healthy" in tokens and "immune" in tokens) or "healthyimmunedoc" in tokens:
        return "Healthy Immune Doc"
    # Diary of a CEO
    if "diary" in tokens and "ceo" in tokens:
        return "The Diary of A CEO"
    # Huberman
    if "huberman" in tokens:
        return "Andrew Huberman"
    # Attia
    if "attia" in tokens:
        return "Peter Attia MD"
    # Jamnadas
    if "jamnadas" in tokens:
        return "Dr. Pradip Jamnadas, MD"

    # Exact match fallback after punctuation strip
    for canon in ALLOWED_CREATORS:
        if n == canon.lower(): return canon
        if re.sub(r"[^\w\s]","",n) == re.sub(r"[^\w\s]","",canon.lower()):
            return canon
    return None


# ---------------- LLM answerer ----------------
def openai_answer(model_name: str, question: str, history, grouped_video_block: str,
                  web_snips: list[dict], no_video: bool) -> str:
    """Compose answer from grouped video quotes + trusted sites. History provides follow-up context."""
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ OPENAI_API_KEY is not set."

    # carry last 2 Q&A pairs for follow-ups context
    recent = [m for m in history[-6:] if m.get("role") in ("user","assistant")]
    convo = [("User: " if m["role"]=="user" else "Assistant: ")+m.get("content","") for m in recent]

    web_lines = [
        f"(W{j}) {s.get('domain','web')} — {s.get('url','')}\n“{_normalize_text(s.get('text',''))[:300]}”"
        for j,s in enumerate(web_snips,1)
    ]
    web_block = "\n".join(web_lines) if web_lines else "None"

    # use web-only if no relevant video
    fallback_line = ("If no suitable video evidence exists, you MAY answer from trusted web snippets alone, "
                     "but begin with: 'Web-only evidence'.\n") if (WEB_FALLBACK and no_video) else \
                    "Trusted web snippets are supporting evidence.\n"

    system = (
        "Answer from the provided evidence plus trusted web sources. Priority: (1) grouped VIDEO evidence from selected experts, "
        "(2) trusted WEB snippets.\n" + fallback_line +
        "Rules:\n"
        "• Cite every claim/step: (Video k) for videos, (DOMAIN Wj) for web.\n"
        "• Prefer human clinical data; label animal/in-vitro/mechanistic explicitly.\n"
        "• Normalize units; include numeric effect sizes when present.\n"
        "• List therapeutic OPTIONS by class + drugs; include mechanism and typical magnitude; use 'dose not specified' if missing. No diagnosis.\n"
        "Structure: Key summary • Practical protocol • Safety notes. Be concise and source-grounded.\n"
        "Use only the provided quotes/snippets; do not invent claims."
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


# ---------------- Loaders ----------------
@st.cache_data(show_spinner=False)
def load_video_meta()->Dict[str,Dict[str,Any]]:
    """video_meta.json keyed by video_id."""
    if VIDEO_META_JSON.exists():
        try:return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except:return {}
    return {}

def _vid_epoch(vm:dict, vid:str)->float:
    info=(vm or {}).get(vid,{})
    return _iso_to_epoch(info.get("published_at") or info.get("publishedAt") or info.get("date") or "")

def _recency_score(published_ts:float, now:float, half_life_days:float)->float:
    if published_ts<=0: return 0.0
    days=max(0.0,(now-published_ts)/86400.0)
    return 0.5 ** (days / max(1e-6, half_life_days))

def precompute_status(embedder_model:str)->Dict[str,Any]:
    """Sanity checks for centroids/ids shapes and norms."""
    status = {
        "centroids_present": VID_CENT_NPY.exists(),
        "ids_present": VID_IDS_TXT.exists(),
        "summaries_present": VID_SUM_JSON.exists(),
        "chunks_mtime": _file_mtime(CHUNKS_PATH),
        "cent_mtime": _file_mtime(VID_CENT_NPY),
        "ids_mtime": _file_mtime(VID_IDS_TXT),
        "ok_shapes": None, "ok_norms": None, "ok_dim": None, "msg":[]
    }
    try:
        if status["centroids_present"] and status["ids_present"]:
            C = np.load(VID_CENT_NPY)
            vids = VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
            status["ok_shapes"] = (C.ndim==2 and C.shape[0]==len(vids))
            n = np.linalg.norm(C,axis=1) if C.ndim==2 else np.array([])
            status["ok_norms"] = bool(len(n) and n.min()>0.90 and n.max()<1.10)
            emb_dim = _load_embedder(embedder_model).get_sentence_embedding_dimension()
            status["ok_dim"] = (C.shape[1]==emb_dim)
            if status["chunks_mtime"] > max(status["cent_mtime"], status["ids_mtime"]):
                status["msg"].append("chunks.jsonl is newer than centroids/ids → run precompute.")
        else:
            status["msg"].append("Centroids/IDs missing → run precompute.")
    except Exception as e:
        status["msg"].append(f"precompute check error: {e}")
    return status


# ---------------- JSONL offsets for random access ----------------
def _ensure_offsets()->np.ndarray:
    """Build a seek index for chunks.jsonl so we can random-access by FAISS ids."""
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
    """Yield rows by numeric line indices from chunks.jsonl."""
    if not CHUNKS_PATH.exists(): return
    offs=_ensure_offsets()
    want=[i for i in indices if 0<=i<len(offs)]
    if limit is not None: want=want[:limit]
    with CHUNKS_PATH.open("rb") as f:
        for i in want:
            f.seek(int(offs[i])); raw=f.readline()
            try: yield i, json.loads(raw)
            except: continue


# ---------------- Model + FAISS ----------------
@st.cache_resource(show_spinner=False)
def _load_embedder(model_name:str)->SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")

@st.cache_resource(show_spinner=False)
def load_metas_and_model(index_path:Path=INDEX_PATH, metas_path:Path=METAS_PKL):
    """Load FAISS index + encoder chosen in metas.pkl. Enforce dim match."""
    if not index_path.exists() or not metas_path.exists(): return None, None, None
    index = faiss.read_index(str(index_path))
    with metas_path.open("rb") as f:
        payload=pickle.load(f)
    metas_from_pkl = payload.get("metas",[])
    model_name = payload.get("model","sentence-transformers/all-MiniLM-L6-v2")
    local_dir = DATA_ROOT/"models"/"all-MiniLM-L6-v2"
    try_name = str(local_dir) if (local_dir/"config.json").exists() else model_name
    embedder = _load_embedder(try_name)
    if index.d != embedder.get_sentence_embedding_dimension():
        raise RuntimeError(f"Embedding dim mismatch: FAISS={index.d} vs Encoder={embedder.get_sentence_embedding_dimension()}. Rebuild.")
    return index, metas_from_pkl, {"model_name":try_name, "embedder":embedder}

@st.cache_resource(show_spinner=False)
def load_video_centroids():
    """Return per-video centroid matrix and aligned video_id list."""
    if not (VID_CENT_NPY.exists() and VID_IDS_TXT.exists()): return None, None
    C = np.load(VID_CENT_NPY).astype("float32")
    vids = VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
    if C.shape[0]!=len(vids): return None, None
    return C, vids

@st.cache_data(show_spinner=False)
def load_video_summaries():
    """Light extractive bullets per video; improves routing and quotes quality."""
    if VID_SUM_JSON.exists():
        try:return json.loads(VID_SUM_JSON.read_text(encoding="utf-8"))
        except:return {}
    return {}


# ---------------- Diversity (MMR) ----------------
def mmr(qv:np.ndarray, doc_vecs:np.ndarray, k:int, lambda_diversity:float=0.45)->List[int]:
    """Maximal Marginal Relevance to avoid near-duplicate quotes."""
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

def _quote_is_valid(text:str)->bool:
    """
    Soft filter so we don't throw away everything:
    - at least 40 characters
    - contains a sentence boundary or a colon (structured bullet)
    """
    t=_normalize_text(text)
    if len(t) < 40: return False
    return any(x in t for x in [". ","; ",": ","? ","! "])

def _dedupe_passages(items:List[Dict[str,Any]], time_window_sec:float=8.0, min_chars:int=40):
    """Remove near-duplicate lines around same timestamp and keep only substantial text."""
    out=[]; seen=[]
    for h in sorted(items, key=lambda r: float((r.get("meta") or {}).get("start",0))):
        ts=float((h.get("meta") or {}).get("start",0))
        txt=_normalize_text(h.get("text",""))
        if len(txt)<min_chars or not _quote_is_valid(txt): continue
        if any(abs(ts - float((s.get("meta") or {}).get("start",0)))<=time_window_sec and _normalize_text(s.get("text",""))==txt for s in seen):
            continue
        seen.append(h); out.append(h)
    return out


# ---------------- Summary-aware routing ----------------
def _build_idf_over_bullets(summaries: dict) -> dict:
    """Global IDF over summary bullets supports keyword score."""
    DF = collections.Counter()
    vids = list(summaries.keys())
    for v in vids:
        for b in summaries.get(v, {}).get("bullets", []):
            toks = set((b.get("text","") or "").lower().split())
            for w in toks: DF[w] += 1
    N = max(1, len(vids))
    return {w: math.log((N+1)/(df+0.5)) for w,df in DF.items()}

def _kw_score(text: str, query: str, idf: dict) -> Tuple[float,int]:
    """
    Keyword score + count of overlapping unique query terms.
    We use the overlap count as an extra guard to avoid irrelevant videos.
    """
    if not text: return 0.0, 0
    qtok = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if w]
    t = re.findall(r"[a-z0-9]+", (text or "").lower())
    tf = {w: t.count(w) for w in set(t)}
    overlap=len(set(qtok) & set(t))
    score = sum(tf.get(w,0) * idf.get(w,0.0) for w in set(qtok)) / (len(t)+1e-6)
    return score, overlap

def route_videos_by_summary(
    query: str, qv: np.ndarray,
    summaries: dict, vm: dict,
    C: np.ndarray | None, vids: list[str] | None,
    allowed_vids: set[str],
    topK: int, recency_weight: float, half_life_days: float,
    min_kw_overlap:int=2
) -> list[str]:
    """
    Score videos by centroid similarity + bullet-keyword score + recency.
    **New guard**: require at least `min_kw_overlap` keyword overlaps to reduce spurious picks.
    """
    universe = [v for v in (vids or list(vm.keys())) if (not allowed_vids or v in allowed_vids)]
    if not universe: return []
    cent = {}
    if C is not None and vids is not None and len(vids) == C.shape[0]:
        sim = (C @ qv.reshape(-1,1)).ravel()
        cent = {vids[i]: float(sim[i]) for i in range(len(vids))}
    idf = _build_idf_over_bullets(summaries)
    now = time.time()
    scored = []
    for v in universe:
        bullets = summaries.get(v, {}).get("bullets", [])
        pseudo = " ".join(b.get("text","") for b in bullets)[:2000]
        kw, overlap = _kw_score(pseudo, query, idf)
        if overlap < min_kw_overlap and len(pseudo)>0:
            continue  # skip videos with near-zero topical overlap
        cs = cent.get(v, 0.0)
        rec = _recency_score(_vid_epoch(vm, v), now, half_life_days)
        base = 0.6*cs + 0.3*kw
        score = (1.0 - recency_weight)*base + recency_weight*(0.1*rec + 0.9*base)
        scored.append((v, score))
    scored.sort(key=lambda x:-x[1])
    return [v for v,_ in scored[:int(topK)]]


# ---------------- Stage B search ----------------
def stageB_search_chunks(query:str,
    index:faiss.Index, embedder:SentenceTransformer,
    candidate_vids:Set[str] | None,
    initial_k:int, final_k:int, max_videos:int, per_video_cap:int,
    apply_mmr:bool, mmr_lambda:float,
    recency_weight:float, half_life_days:float, vm:dict)->List[Dict[str,Any]]:

    """Dense search -> re-embed -> MMR diversify -> recency blend -> per-video capping."""
    if index is None: return []
    qv=embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
    K = min(int(initial_k), index.ntotal if index.ntotal>0 else int(initial_k))
    D,I=index.search(qv.reshape(1,-1), K)
    idxs=[int(x) for x in I[0] if x>=0]
    scores0=[float(s) for s in D[0][:len(idxs)]]

    rows=list(iter_jsonl_rows(idxs))
    texts=[]; metas=[]; keep=[]
    for _,j in rows:
        t=_normalize_text(j.get("text",""))
        m=(j.get("meta") or {}).copy()
        vid=(m.get("video_id") or m.get("vid") or m.get("ytid") or
             j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
        if vid: m["video_id"]=vid
        if "start" not in m and "start_sec" in m: m["start"]=m.get("start_sec")
        m["start"]=_parse_ts(m.get("start",0))
        if t:
            texts.append(t); metas.append(m)
            keep.append((candidate_vids is None) or (vid in candidate_vids))

    if any(keep):
        texts=[t for t,k in zip(texts,keep) if k]
        metas=[m for m,k in zip(metas,keep) if k]
        idxs=[i for i,k in zip(idxs,keep) if k]
        scores0=[s for s,k in zip(scores0,keep) if k]
    if not texts: return []

    doc_vecs=embedder.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")
    order=list(range(len(texts)))
    if apply_mmr:
        order=mmr(qv, doc_vecs, k=min(len(texts), max(8, final_k*2)), lambda_diversity=float(mmr_lambda))

    now=time.time(); blended=[]
    for li in order:
        i_global=idxs[li] if li<len(idxs) else None
        base=scores0[li] if li<len(scores0) else 0.0
        m=metas[li]; t=texts[li]; vid=m.get("video_id")
        rec=_recency_score(_vid_epoch(vm,vid), now, half_life_days)
        score=(1.0-recency_weight)*float(base)+recency_weight*float(rec)
        blended.append((i_global,score,t,m))
    blended.sort(key=lambda x:-x[1])

    # Enforce per-video caps and total cap
    picked=[]; seen_per_video={}; distinct=[]
    for ig,sc,tx,me in blended:
        vid=me.get("video_id","Unknown")
        if vid not in distinct and len(distinct)>=int(max_videos): continue
        if seen_per_video.get(vid,0)>=int(per_video_cap): continue
        if vid not in distinct: distinct.append(vid)
        seen_per_video[vid]=seen_per_video.get(vid,0)+1
        picked.append({"i":ig,"score":float(sc),"text":tx,"meta":me})
        if len(picked)>=int(final_k): break
    return picked


# ---------------- Grouped evidence ----------------
def group_hits_by_video(hits:List[Dict[str,Any]])->Dict[str,List[Dict[str,Any]]]:
    g={}
    for h in hits:
        vid=(h.get("meta") or {}).get("video_id") or "Unknown"
        g.setdefault(vid,[]).append(h)
    return g

def build_grouped_evidence_for_prompt(hits:List[Dict[str,Any]], vm:dict, summaries:dict, max_quotes:int=3)->Tuple[str,Dict[str,Any]]:
    """
    Build a human-readable block for the LLM **and** a structured export with per-quote timestamps.
    Guarantees at least one quote per routed video by falling back to raw chunks if no bullets.
    """
    groups=group_hits_by_video(hits)
    ordered = sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)
    lines=[]; export=[]
    for v_idx,(vid,items) in enumerate(ordered,1):
        info=vm.get(vid,{})
        title=info.get("title") or summaries.get(vid,{}).get("title") or vid
        creator_raw=_raw_creator_of_vid(vid, vm)
        creator=_canonicalize_creator(creator_raw) or creator_raw
        date=info.get("published_at") or info.get("publishedAt") or info.get("date") or summaries.get(vid,{}).get("published_at") or ""
        lines.append(f"[Video {v_idx}] {title} — {creator}" + (f" — {date}" if date else ""))

        # 1) prefer summary bullets
        bullets = summaries.get(vid,{}).get("bullets", [])[:max_quotes]
        added=0
        for b in bullets:
            q=_normalize_text(b.get("text",""))
            if not _quote_is_valid(q): continue
            ts=_format_ts(b.get("ts",0))
            lines.append(f"  • {ts}: “{q[:260]}”")
            export.append({"video_id":vid,"title":title,"creator":creator,"ts":ts,"text":q})
            added+=1

        # 2) top raw chunks to ensure >=1 quote survives
        if added < max_quotes:
            clean=_dedupe_passages(items, time_window_sec=8.0, min_chars=40)
            for h in clean[:max(1, max_quotes - added)]:
                ts=_format_ts((h.get("meta") or {}).get("start",0))
                q=_normalize_text(h.get("text",""))
                lines.append(f"  • {ts}: “{q[:260]}”")
                export.append({"video_id":vid,"title":title,"creator":creator,"ts":ts,"text":q})
        lines.append("")
    return "\n".join(lines).strip(), {"videos": export}


# ---------------- Web fetch ----------------
def _ddg_html(domain:str, query:str, headers:dict, timeout:float)->List[str]:
    """DuckDuckGo HTML endpoint; returns result hrefs or []."""
    try:
        r=requests.get("https://duckduckgo.com/html/", params={"q":f"site:{domain} {query}"}, headers=headers, timeout=timeout)
        if r.status_code!=200: return []
        soup=BeautifulSoup(r.text,"html.parser")
        return [a.get("href") for a in soup.select("a.result__a") if a.get("href")]
    except Exception:
        return []

def _ddg_lite(domain:str, query:str, headers:dict, timeout:float)->List[str]:
    """Very simple lite endpoint that sometimes works when HTML throttles."""
    try:
        r=requests.get("https://duckduckgo.com/lite/", params={"q":f"site:{domain} {query}"}, headers=headers, timeout=timeout)
        if r.status_code!=200: return []
        soup=BeautifulSoup(r.text,"html.parser")
        return [a.get("href") for a in soup.find_all("a") if a.get("href","").startswith("http")]
    except Exception:
        return []

def fetch_trusted_snippets(query:str, allowed_domains:List[str], max_snippets:int=3, per_domain:int=1, timeout:float=6.0):
    """
    Try HTML → Lite → homepage fallback. Returns list of {domain,url,text} and
    writes a debug trace into st.session_state['web_trace'] for transparency.
    """
    trace=[]
    out=[]
    if not requests or not BeautifulSoup or max_snippets<=0:
        st.session_state["web_trace"] = "requests/bs4 missing or disabled."
        return []

    headers={"User-Agent":"Mozilla/5.0"}
    for domain in allowed_domains:
        links=_ddg_html(domain, query, headers, timeout)
        if not links:
            links=_ddg_lite(domain, query, headers, timeout)
            trace.append(f"{domain}: lite mode links={len(links)}")
        else:
            trace.append(f"{domain}: html mode links={len(links)}")

        if not links:
            # homepage fallback lets at least show something non-empty
            links=[f"https://{domain}"]
            trace.append(f"{domain}: fallback homepage")

        links=links[:per_domain]
        for url in links:
            try:
                r=requests.get(url, headers=headers, timeout=timeout)
                if r.status_code!=200: 
                    trace.append(f"{domain}: {url} status {r.status_code}")
                    continue
                soup=BeautifulSoup(r.text,"html.parser")
                paras=[p.get_text(" ",strip=True) for p in soup.find_all("p")]
                text=_normalize_text(" ".join(paras))[:1800]
                if len(text)<200:
                    trace.append(f"{domain}: {url} too short")
                    continue
                out.append({"domain":domain,"url":url,"text":text})
            except Exception as e:
                trace.append(f"{domain}: fetch error {e}")
        if len(out)>=max_snippets: break

    st.session_state["web_trace"]="; ".join(trace) if trace else "no trace"
    return out[:max_snippets]


# ---------------- Precompute runners + repairs ----------------
def _run_precompute_inline()->str:
    """Attempt to run scripts/precompute_video_summaries.py in-process."""
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
    """Safely renormalize centroids to unit length."""
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
    """
    Create video_summaries.json directly from chunks.jsonl (light TF-IDF bullets).
    This is a last-resort path if your upstream summary job didn't run.
    """
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
                for w in set(re.findall(r"[a-z0-9]+",t.lower())):
                    if w not in seen:
                        DF[w]+=1; seen.add(w)
        N = max(1,len(vids))
        def score_line(t:str)->float:
            words=re.findall(r"[a-z0-9]+",t.lower())
            tf=collections.Counter(words)
            return sum(tf[w]*math.log((N+1)/(DF.get(w,1)+0.5)) for w in tf)/(len(words)+1e-6)

        summaries={}
        for vid in vids:
            lines=texts_by_vid[vid][:max_lines_per_video]
            times=ts_by_vid[vid][:max_lines_per_video]
            idx_scores=[(i,score_line(t)) for i,t in enumerate(lines)]
            top=sorted([i for i,_ in sorted(idx_scores,key=lambda x:-x[1])[:12]])[:10]
            bullets=[{"ts":float(times[i]), "text":(lines[i][:280]+"…" if len(lines[i])>280 else lines[i])} for i in top[:6]]
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


# ---------------- Creator index from chunks ----------------
def build_creator_indexes_from_chunks(vm: dict) -> tuple[dict, dict]:
    """
    Build (vid->creator, creator->set(vid)) using chunks.jsonl first, then fill gaps from video_meta.
    Uses robust heuristic canonicalization so variants still map correctly.
    """
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
                canon = _canonicalize_creator(raw)
                if canon is None: continue
                if vid not in vid_to_creator:
                    vid_to_creator[vid] = canon
                    creator_to_vids.setdefault(canon, set()).add(vid)
    for vid in vm.keys():
        if vid in vid_to_creator: continue
        canon = _canonicalize_creator(_raw_creator_of_vid(vid, vm))
        if canon is None: continue
        vid_to_creator[vid]=canon
        creator_to_vids.setdefault(canon,set()).add(vid)
    return vid_to_creator, creator_to_vids


# ---------------- Verification helper (admin) ----------------
def scan_chunks_for_terms(terms:List[str], vm:Dict[str,Dict[str,Any]], limit_examples:int=200):
    """Quick coverage scan to see which experts mention the terms."""
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
            creator=_canonicalize_creator(_raw_creator_of_vid(vid, vm)) or _raw_creator_of_vid(vid, vm)
            per_creator[creator]=per_creator.get(creator,0)+1
            total+=1
            if len(examples)<int(limit_examples):
                sn=_normalize_text(t); sn=sn[:260]+"…" if len(sn)>260 else sn
                examples.append({"video_id":vid,"creator":creator,"ts":_format_ts(st_sec),"snippet":sn})
    per_creator=dict(sorted(per_creator.items(), key=lambda kv:-kv[1]))
    return {"total_matches":total,"per_creator":per_creator,"examples":examples}


# ================= UI =================
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

# Persist per-turn sources
if "turns" not in st.session_state: st.session_state["turns"]=[]
if "messages" not in st.session_state: st.session_state["messages"]=[]

with st.sidebar:
    st.markdown("**Auto Mode** · tuned for accuracy + diversity")

    # Experts checklist with counts derived from chunks + video_meta
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
    st.session_state["selected_creators"]=selected_creators  # persist for follow-ups

    # Trusted sites
    st.subheader("Trusted sites")
    st.caption("Short excerpts from vetted medical sites are added as supporting evidence.")
    allow_web = st.checkbox(
        "Include supporting website excerpts", value=True,
        help="Fetches brief paragraphs from the checked sites and blends them with expert videos."
    )
    selected_domains=[]
    for i,dom in enumerate(TRUSTED_DOMAINS):
        if st.checkbox(dom, value=True, key=f"site_{i}"):
            selected_domains.append(dom)
    max_web_auto = 3  # keep compact and readable

    # Model choice
    model_choice = st.selectbox(
        "Answering model", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0,
        help="Chooses how the final answer is written from the evidence."
    )

    # Advanced
    with st.expander("Advanced (technical controls)", expanded=False):
        st.caption("Defaults are tuned. Adjust only if needed.")
        initial_k_adv = st.number_input("Scan candidates first (K)", 128, 5000, 768, 64, key="adv_scanK")
        final_k_adv   = st.number_input("Use top passages", 8, 120, 36, 2, key="adv_useK")
        max_videos_adv = st.number_input("Max videos", 1, 12, 5, 1, key="adv_maxvid")
        per_video_cap_adv = st.number_input("Passages per video cap", 1, 10, 4, 1, key="adv_cap")
        use_mmr_adv = st.checkbox("Diversify with MMR", value=True, key="adv_mmr")
        mmr_lambda_adv = st.slider("MMR balance", 0.1, 0.9, 0.45, 0.05, key="adv_lam")
        recency_weight_adv = st.slider("Recency weight", 0.0, 1.0, 0.20, 0.05, key="adv_rec")
        half_life_adv = st.slider("Recency half-life (days)", 7, 720, 270, 7, key="adv_hl")

    st.divider()
    show_diag = st.toggle(
        "Show data diagnostics", value=False,
        help="Check file locations, modification times, and index health."
    )
    st.subheader("Library status")
    st.checkbox("chunks.jsonl present", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("offsets built", value=OFFSETS_NPY.exists(), disabled=True)
    st.checkbox("faiss.index present", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("metas.pkl present", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("video_meta.json present", value=VIDEO_META_JSON.exists(), disabled=True)
    cent_ready = VID_CENT_NPY.exists() and VID_IDS_TXT.exists() and VID_SUM_JSON.exists()
    st.caption("Video centroids/summaries: ready" if cent_ready else "Precompute not found. Use admin tools.")


# Diagnostics area
if show_diag:
    colA,colB,colC = st.columns([2,3,3])
    with colA: st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    with colB: st.caption(f"chunks mtime: {_iso(_file_mtime(CHUNKS_PATH)) if CHUNKS_PATH.exists() else 'missing'}")
    with colC: st.caption(f"index mtime: {_iso(_file_mtime(INDEX_PATH)) if INDEX_PATH.exists() else 'missing'}")

# Render chat history briefly
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Admin diagnostics
if _is_admin():
    st.subheader("Diagnostics (admin)")
    try:
        _,_,payload = load_metas_and_model()
        status = precompute_status(payload["model_name"])
    except Exception as e:
        status = {"msg":[f"load error: {e}"]}

    col1,col2,col3 = st.columns(3)
    with col1:
        st.metric("Centroids present", "Yes" if status.get("centroids_present") else "No")
        st.metric("IDs present", "Yes" if status.get("ids_present") else "No")
        st.metric("Summaries present", "Yes" if status.get("summaries_present") else "No")
    with col2:
        st.metric("Shapes OK", "Yes" if status.get("ok_shapes") else "No")
        st.metric("Norms ~1.0", "Yes" if status.get("ok_norms") else "No")
        st.metric("Dim matches encoder", "Yes" if status.get("ok_dim") else "No")
    with col3:
        st.caption(f"chunks mtime: {_iso(status.get('chunks_mtime',0))}")
        st.caption(f"centroids mtime: {_iso(status.get('cent_mtime',0))}")
        st.caption(f"ids mtime: {_iso(status.get('ids_mtime',0))}")

    # Freshness check
    cent_m   = _file_mtime(VID_CENT_NPY)
    ids_m    = _file_mtime(VID_IDS_TXT)
    chunks_m = _file_mtime(CHUNKS_PATH)
    ok_cent = bool(cent_m and chunks_m and cent_m > chunks_m)
    ok_ids  = bool(ids_m  and chunks_m and ids_m  > chunks_m)

    st.subheader("Precompute freshness check")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("centroids newer than chunks", "Yes" if ok_cent else "No")
        st.caption(f"centroids: {_iso(cent_m)}  |  chunks: {_iso(chunks_m)}")
    with c2:
        st.metric("ids newer than chunks", "Yes" if ok_ids else "No")
        st.caption(f"ids: {_iso(ids_m)}  |  chunks: {_iso(chunks_m)}")
    if not (ok_cent and ok_ids):
        st.warning("Centroids/IDs are older than chunks.jsonl. Use rebuild below.")

    st.markdown("---")
    st.subheader("Files on disk (debug)")
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
        if not VID_SUM_JSON.exists():
            st.warning("Summaries missing. Use rebuild or fallback.")

    with st.expander("Creator inventory (from chunks.jsonl)"):
        inv = sorted(((c, len(vs)) for c,vs in build_creator_indexes_from_chunks(load_video_meta())[1].items()),
                     key=lambda x: -x[1])
        st.dataframe([{"creator": c, "videos": n} for c,n in inv], use_container_width=True)

    st.markdown("---")
    st.caption("Keyword coverage scan (verification only).")
    default_terms = "apob, apo-b, ldl, statin, ezetimibe, pcsk9, bempedoic, inclisiran, niacin, sleep, magnesium, melatonin"
    term_input = st.text_input("Terms (comma-separated)", default_terms)
    if st.button("Run scan"):
        terms=[t.strip() for t in term_input.split(",") if t.strip()]
        with st.spinner("Scanning transcripts…"):
            scan=scan_chunks_for_terms(terms=terms, vm=load_video_meta(), limit_examples=300)
        st.metric("Total matching chunks", scan["total_matches"])
        if scan["per_creator"]: st.dataframe([{"expert":k,"matching_chunks":v} for k,v in scan["per_creator"].items()], use_container_width=True)
        if scan["examples"]: st.dataframe(scan["examples"], use_container_width=True)
    st.markdown("---")


# --------------- Prompt input ---------------
prompt = st.chat_input("Ask about ApoB/LDL drugs, sleep, protein timing, fasting, supplements, protocols…")
if prompt is None:
    # show history sources per turn for transparency
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
                        info = load_video_meta().get(v.get("video_id",""),{}) or {}
                        url = info.get("url","")
                        hdr = f"- **{v.get('title','')}** — _{v.get('creator','')}_"
                        st.markdown(f"{hdr}" + (f" — [{url}]({url})" if url else ""))
                        st.markdown(f"  • **{v.get('ts','')}** — “{_normalize_text(v.get('text',''))[:160]}”")
                if web:
                    st.markdown("**Trusted websites**")
                    for j, s in enumerate(web, 1):
                        st.markdown(f"W{j}. [{s['domain']}]({s['url']})")
    cols=st.columns([1]*12)
    with cols[-1]:
        st.button("Clear chat", key="clear_idle", help="Start a new conversation.", on_click=_clear_chat)
    st.stop()

# store user message
st.session_state.messages.append({"role":"user","content":prompt})
with st.chat_message("user"): st.markdown(prompt)

# Guardrails
missing=[p for p in REQUIRED if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error("Missing required files:\n" + "\n".join(f"- {p}" for p in missing))
    st.stop()

# Load FAISS + encoder + summaries
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

# Stage A routing with light follow-up context
routing_query = prompt
# include previous user message lightly to help follow-ups without polluting too much
prev_users = [m["content"] for m in st.session_state.messages if m["role"]=="user"]
if len(prev_users) >= 2:
    routing_query = prev_users[-2] + " ; " + prompt

qv = embedder.encode([routing_query], normalize_embeddings=True).astype("float32")[0]
topK_route = 5
recency_weight_auto = 0.20
half_life_auto = 270

routed_vids = route_videos_by_summary(
    routing_query, qv, summaries, vm, C, list(universe), allowed_vids,
    topK=topK_route, recency_weight=recency_weight_auto, half_life_days=half_life_auto,
    min_kw_overlap=2
)
candidate_vids = set(routed_vids) if routed_vids else allowed_vids

# Auto recall defaults (overridable in Advanced)
K_scan = st.session_state.get("adv_scanK", 768)
K_use  = st.session_state.get("adv_useK", 36)
max_videos = st.session_state.get("adv_maxvid", 5)
per_video_cap = st.session_state.get("adv_cap", 4)
use_mmr = st.session_state.get("adv_mmr", True)
mmr_lambda = st.session_state.get("adv_lam", 0.45)
recency_weight = st.session_state.get("adv_rec", recency_weight_auto)
half_life = st.session_state.get("adv_hl", half_life_auto)

# Stage B: search chunks only in routed videos
with st.spinner("Scanning selected videos…"):
    try:
        hits = stageB_search_chunks(
            prompt, index, embedder, candidate_vids,
            initial_k=min(int(K_scan), index.ntotal if index is not None else int(K_scan)),
            final_k=int(K_use), max_videos=int(max_videos), per_video_cap=int(per_video_cap),
            apply_mmr=bool(use_mmr), mmr_lambda=float(mmr_lambda),
            recency_weight=float(recency_weight), half_life_days=float(half_life), vm=vm
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Search failed."); st.exception(e); st.stop()

# Web support (up to 3, 1 per domain)
web_snips=[]
if allow_web and selected_domains and requests and BeautifulSoup and int(max_web_auto)>0:
    with st.spinner("Fetching trusted websites…"):
        web_snips = fetch_trusted_snippets(prompt, selected_domains, max_snippets=int(max_web_auto), per_domain=1)

# Build evidence and answer
grouped_block, export_struct = build_grouped_evidence_for_prompt(hits, vm, summaries, max_quotes=3)

with st.chat_message("assistant"):
    if not export_struct["videos"] and not web_snips:
        st.warning("Insufficient expert video evidence after relevance filtering.")
        st.session_state.messages.append({"role":"assistant","content":"No relevant evidence found."})
        cols=st.columns([1]*12)
        with cols[-1]:
            st.button("Clear chat", key="clear_nohits", on_click=_clear_chat)
        st.stop()

    with st.spinner("Writing your answer…"):
        ans=openai_answer(model_choice, prompt, st.session_state.messages, grouped_block, web_snips, no_video=(len(export_struct["videos"])==0))

    # Show the answer first
    st.markdown(ans)
    st.session_state.messages.append({"role":"assistant","content":ans})

    # Persist sources for THIS reply only
    st.session_state["turns"].append({
        "prompt": prompt,
        "answer": ans,
        "videos": export_struct["videos"],
        "web": web_snips,
        "web_trace": st.session_state.get("web_trace","")
    })

    # Sources UI for this reply
    with st.expander("Sources for this reply", expanded=False):
        st.caption("Each bullet shows a timestamped quote pulled from a routed video. Websites are supporting references.")
        vids = export_struct["videos"]
        if vids:
            # group by video_id for neat display
            by_vid={}; 
            for v in vids: by_vid.setdefault(v["video_id"], []).append(v)
            order = list(by_vid.keys())
            for vid in order:
                info = vm.get(vid, {})
                title = info.get("title") or by_vid[vid][0].get("title","")
                creator = by_vid[vid][0].get("creator","")
                url = info.get("url") or ""
                header = f"- **{title}**" + (f" — _{creator}_" if creator else "")
                st.markdown(f"{header}" + (f" — [{url}]({url})" if url else ""))
                for q in by_vid[vid]:
                    ts=q.get("ts",""); qt=_normalize_text(q.get("text",""))
                    if len(qt)>160: qt=qt[:160]+"…"
                    st.markdown(f"  • **{ts}** — “{qt}”")
        if web_snips:
            st.markdown("**Trusted websites**")
            for j, s in enumerate(web_snips, 1):
                st.markdown(f"W{j}. [{s['domain']}]({s['url']})")
        if st.session_state.get("web_trace"):
            st.caption(f"Web fetch: enabled={bool(allow_web)} trace={st.session_state.get('web_trace')}")

# Footer + Clear chat
st.caption("Routing uses summary bullets + centroids with keyword overlap; diversified quotes are fused with trusted medical sites.")
cols=st.columns([1]*12)
with cols[-1]:
    st.button("Clear chat", key="clear_done", on_click=_clear_chat)
