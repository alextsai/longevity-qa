# -*- coding: utf-8 -*-
"""
Health | Nutrition Q&A — experts-first RAG + domain routing + trusted web + streaming.

Fixes and changes
- Quote accuracy: keyword gating, domain+summary routing, creator allow-list, MMR, min-sim guard, per-video dedupe.
- Relevance: require query-token overlap at routing and quote-pick time.
- Trusted web: visible links + excerpts under collapsed section; fetch trace available.
- Usability: page renamed, sources collapsed by default, streaming answer, faster cache and deferred inventory.
- Domain artifacts: auto-bootstrap from repo to mounted volume; admin verify tool.

DATA_DIR env points to base dir that contains data/{chunks,index,catalog,domain}.
"""

from __future__ import annotations
import os, sys, re, json, yaml, time, math, pickle, hashlib, collections
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
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
    requests = None
    BeautifulSoup = None

# Optional domain model
try:
    import joblib
except Exception:
    joblib = None

# ---------- Constants ----------
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS","1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(os.getenv("DATA_DIR", "/var/data/data")).resolve()

# Core artifacts
CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"
OFFSETS_NPY     = DATA_ROOT / "data/chunks/chunks.offsets.npy"
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"
VID_CENT_NPY    = DATA_ROOT / "data/index/video_centroids.npy"
VID_IDS_TXT     = DATA_ROOT / "data/index/video_ids.txt"
VID_SUM_JSON    = DATA_ROOT / "data/catalog/video_summaries.json"

# Domain artifacts
DOMAIN_DIR        = DATA_ROOT / "data/domain"
DOMAIN_MODEL      = DOMAIN_DIR / "domain_model.joblib"
DOMAIN_PROBS_JSON = DOMAIN_DIR / "domain_probs.json"
DOMAIN_PROBS_YAML = DOMAIN_DIR / "domain_probs.yaml"

REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]
WEB_FALLBACK = os.getenv("WEB_FALLBACK","true").strip().lower() in {"1","true","yes","on"}

TRUSTED_DOMAINS = [
    "nih.gov","medlineplus.gov","cdc.gov","mayoclinic.org","health.harvard.edu",
    "familydoctor.org","healthfinder.gov","ama-assn.org","nejm.org",
    "med.stanford.edu","icahn.mssm.edu","who.int"
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

# ---------- Utils ----------
def _normalize_text(s:str)->str:
    return re.sub(r"\s+"," ", (s or "").strip())

def _parse_ts(v)->float:
    if isinstance(v,(int,float)): 
        try: return float(v)
        except: return 0.0
    try:
        sec=0.0
        for p in str(v).split(":"): sec=sec*60+float(p)
        return sec
    except: return 0.0

def _yt_ts_url(url:str, start_sec:float)->str:
    if not url: return ""
    s = int(max(0, round(start_sec)))
    join = "&" if "?" in url else "?"
    return f"{url}{join}t={s}s"

def _iso_to_epoch(iso:str)->float:
    if not iso: return 0.0
    try:
        if "T" in iso: return datetime.fromisoformat(iso.replace("Z","+00:00")).timestamp()
        return datetime.fromisoformat(iso).timestamp()
    except: return 0.0

def _recency_score(published_ts:float, now:float, half_life_days:float)->float:
    if published_ts<=0: return 0.0
    days=max(0.0,(now-published_ts)/86400.0)
    return 0.5 ** (days / max(1e-6, half_life_days))

def _vid_epoch(vm:dict, vid:str)->float:
    info=(vm or {}).get(vid,{})
    return _iso_to_epoch(info.get("published_at") or info.get("publishedAt") or info.get("date") or "")

def _file_mtime(p:Path)->float:
    try: return p.stat().st_mtime
    except: return 0.0

def _sha256(p: Path) -> str:
    try:
        h=hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1<<20), b""): h.update(chunk)
        return h.hexdigest()
    except: return ""

def _is_admin()->bool:
    try: qp = st.query_params
    except Exception: return False
    if qp.get("admin","0")!="1": return False
    try: expected = st.secrets["ADMIN_KEY"]
    except Exception: expected = None
    if expected is None: return True
    return qp.get("key","")==str(expected)

def _clear_chat():
    st.session_state["messages"]=[]; st.session_state["turns"]=[]; st.rerun()

# ---------- Domain artifact bootstrap ----------
from shutil import copy2
def _ensure_domain_artifacts():
    repo_src = ROOT / "data" / "domain"
    DOMAIN_DIR.mkdir(parents=True, exist_ok=True)
    for fn in ["domain_model.joblib","domain_probs.json","domain_probs.yaml"]:
        s = repo_src / fn
        d = DOMAIN_DIR / fn
        if s.exists() and not d.exists():
            try: copy2(s, d)
            except: pass
_ensure_domain_artifacts()

def verify_domain_artifacts()->Dict[str,Any]:
    repo = ROOT / "data" / "domain"
    vol  = DOMAIN_DIR
    files = ["domain_model.joblib","domain_probs.json","domain_probs.yaml"]
    rows=[]
    for fn in files:
        pr = repo / fn
        pv = vol  / fn
        rows.append({
            "file": fn,
            "repo_exists": pr.exists(), "repo_size": pr.stat().st_size if pr.exists() else 0, "repo_sha256": _sha256(pr) if pr.exists() else "",
            "vol_exists":  pv.exists(), "vol_size":  pv.stat().st_size if pv.exists() else 0, "vol_sha256":  _sha256(pv) if pv.exists() else "",
        })
    load_ok=None; classes=[]
    try:
        model_path = DOMAIN_MODEL if DOMAIN_MODEL.exists() else (repo/"domain_model.joblib")
        if model_path.exists() and joblib:
            pipe = joblib.load(model_path)
            classes = list(getattr(pipe,"classes_",[])); load_ok=True
        else:
            load_ok=False
    except Exception:
        load_ok=False
    return {"DATA_DIR": str(DATA_ROOT), "rows": rows, "model_load_ok": load_ok, "classes": classes}

# ---------- Caching loaders ----------
@st.cache_data(show_spinner=False)
def load_video_meta()->Dict[str,Dict[str,Any]]:
    if VIDEO_META_JSON.exists():
        try: return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

@st.cache_resource(show_spinner=False)
def _load_embedder(model_name:str)->SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")

@st.cache_resource(show_spinner=False)
def load_metas_and_model(index_path:Path=INDEX_PATH, metas_path:Path=METAS_PKL):
    if not index_path.exists() or not metas_path.exists(): return None, None, None
    index = faiss.read_index(str(index_path))
    with metas_path.open("rb") as f: payload = pickle.load(f)
    model_name = payload.get("model","sentence-transformers/all-MiniLM-L6-v2")
    local_dir = DATA_ROOT/"models"/"all-MiniLM-L6-v2"
    try_name = str(local_dir) if (local_dir/"config.json").exists() else model_name
    embedder = _load_embedder(try_name)
    if index.d != embedder.get_sentence_embedding_dimension():
        raise RuntimeError(f"Embedding dim mismatch: FAISS={index.d} vs Encoder={embedder.get_sentence_embedding_dimension()}. Rebuild index.")
    return index, payload.get("metas",[]), {"model_name":try_name, "embedder":embedder}

@st.cache_resource(show_spinner=False)
def load_video_centroids():
    if not (VID_CENT_NPY.exists() and VID_IDS_TXT.exists()): return None, None
    C = np.load(VID_CENT_NPY).astype("float32")
    vids = VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
    if C.ndim!=2 or C.shape[0]!=len(vids): return None, None
    n = np.linalg.norm(C,axis=1,keepdims=True)+1e-12
    C = C / n
    return C, vids

@st.cache_data(show_spinner=False)
def load_video_summaries():
    if VID_SUM_JSON.exists():
        try: return json.loads(VID_SUM_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

# Build offsets for random access
def _ensure_offsets()->np.ndarray:
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

# ---------- Creator canonicalization ----------
def _raw_creator_of_vid(vid:str, vm:dict)->str:
    info = vm.get(vid, {}) or {}
    for k in ("podcaster","channel","author","uploader","owner","creator"):
        if k in info and info[k]: return str(info[k])
    for k,v in ((kk.lower(), vv) for kk,vv in info.items()):
        if k in {"podcaster","channel","author","uploader","owner","creator"} and v:
            return str(v)
    return "Unknown"

def _canonicalize_creator(name: str) -> str | None:
    n = _normalize_text(name).lower().replace("™","").replace("®","")
    if not n: return None
    if n in EXCLUDED_CREATORS_EXACT: return None
    toks = set(re.findall(r"[a-z0-9]+", n))
    if ("healthy" in toks and "immune" in toks) or "healthyimmunedoc" in toks: return "Healthy Immune Doc"
    if "diary" in toks and "ceo" in toks: return "The Diary of A CEO"
    if "huberman" in toks: return "Andrew Huberman"
    if "attia" in toks: return "Peter Attia MD"
    if "jamnadas" in toks: return "Dr. Pradip Jamnadas, MD"
    for canon in ALLOWED_CREATORS:
        if n == canon.lower(): return canon
        if re.sub(r"[^\w\s]","",n) == re.sub(r"[^\w\s]","",canon.lower()): return canon
    return None

def build_creator_indexes_from_chunks(vm: dict) -> tuple[dict, dict]:
    vid_to_creator: Dict[str,str] = {}
    creator_to_vids: Dict[str,set] = {}
    if CHUNKS_PATH.exists():
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            for ln in f:
                try: j=json.loads(ln)
                except: continue
                m=(j.get("meta") or {})
                vid=(m.get("video_id") or m.get("vid") or m.get("ytid") or
                     j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
                if not vid: continue
                raw=(m.get("channel") or m.get("author") or m.get("uploader") or _raw_creator_of_vid(vid, vm))
                canon=_canonicalize_creator(raw)
                if canon is None: continue
                if vid not in vid_to_creator:
                    vid_to_creator[vid]=canon
                    creator_to_vids.setdefault(canon,set()).add(vid)
    for vid in vm.keys():
        if vid in vid_to_creator: continue
        canon=_canonicalize_creator(_raw_creator_of_vid(vid, vm))
        if canon is None: continue
        vid_to_creator[vid]=canon
        creator_to_vids.setdefault(canon,set()).add(vid)
    return vid_to_creator, creator_to_vids

# ---------- Domain model + per-video probs ----------
@st.cache_resource(show_spinner=False)
def load_domain_model():
    if not DOMAIN_MODEL.exists() or joblib is None:
        return None, {}, set()
    try:
        model = joblib.load(DOMAIN_MODEL)
    except Exception:
        model = None

    probs_raw=[]
    if DOMAIN_PROBS_JSON.exists():
        try: probs_raw=json.loads(DOMAIN_PROBS_JSON.read_text(encoding="utf-8"))
        except: probs_raw=[]
    elif DOMAIN_PROBS_YAML.exists():
        try: probs_raw=yaml.safe_load(DOMAIN_PROBS_YAML.read_text(encoding="utf-8")) or []
        except: probs_raw=[]

    per_video: Dict[str, Dict[str,float]] = {}
    all_domains=set()
    for row in probs_raw:
        vid=row.get("video_id"); 
        if not vid: continue
        per_video.setdefault(vid, {})
        for key,val in row.items():
            if key=="video_id": continue
            try: p=float(val)
            except: continue
            for d in [t.strip() for t in key.split(";") if t.strip()]:
                all_domains.add(d)
                per_video[vid][d]=max(per_video[vid].get(d,0.0), p)
    return model, per_video, all_domains

def classify_query_domains(model, query:str, top_k:int=3)->List[str]:
    if model is None: return []
    try:
        proba = model.predict_proba([query])[0]
        classes = list(model.classes_)
        ranked = sorted(zip(classes, proba), key=lambda x:-x[1])
        return [d for d,_ in ranked[:max(1, top_k)]]
    except Exception:
        return []

# ---------- Retrieval helpers ----------
def _tokenize(s:str)->List[str]:
    return re.findall(r"[a-z0-9]+", (s or "").lower())

STOP = set("""a an the and or but of in on for to with by from into over under about as at is are be was were been being it this that these those his her their our your my i you he she they we them us me""".split())

def _overlap_score(a:List[str], b:List[str])->int:
    return len((set(a)-STOP) & (set(b)-STOP))

def _quote_is_valid(text:str)->bool:
    t=_normalize_text(text)
    if len(t)<40: return False
    return any(x in t for x in [". ","; ",": ","? ","! "])

def _dedupe_passages(items:List[Dict[str,Any]], time_window_sec:float=8.0, min_chars:int=40):
    out=[]; seen=[]
    for h in sorted(items, key=lambda r: float((r.get("meta") or {}).get("start",0))):
        ts=float((h.get("meta") or {}).get("start",0))
        txt=_normalize_text(h.get("text",""))
        if len(txt)<min_chars or not _quote_is_valid(txt): continue
        if any(abs(ts - float((s.get("meta") or {}).get("start",0)))<=time_window_sec and _normalize_text(s.get("text",""))==txt for s in seen):
            continue
        seen.append(h); out.append(h)
    return out

def _build_idf_over_bullets(summaries: dict) -> dict:
    DF = collections.Counter()
    vids = list(summaries.keys())
    for v in vids:
        for b in summaries.get(v, {}).get("bullets", []):
            toks = set(_tokenize(b.get("text","")))
            for w in toks: DF[w]+=1
    N = max(1,len(vids))
    return {w: math.log((N+1)/(df+0.5)) for w,df in DF.items()}

def _kw_score(text: str, query: str, idf: dict) -> Tuple[float,int]:
    if not text: return 0.0, 0
    qtok=_tokenize(query); t=_tokenize(text)
    tf = {w: t.count(w) for w in set(t)}
    overlap = _overlap_score(qtok, t)
    score = sum(tf.get(w,0) * idf.get(w,0.0) for w in set(qtok)) / (len(t)+1e-6)
    return score, overlap

# ---------- Routing ----------
def route_videos_by_summary(
    query: str, qv: np.ndarray,
    summaries: dict, vm: dict,
    C: np.ndarray | None, vids: list[str] | None,
    allowed_vids: set[str],
    topK: int, recency_weight: float, half_life_days: float,
    min_kw_overlap:int=2
) -> list[str]:
    universe = [v for v in (vids or list(vm.keys())) if (not allowed_vids or v in allowed_vids)]
    if not universe: return []
    cent = {}
    if C is not None and vids is not None and len(vids) == C.shape[0]:
        sim = (C @ qv.reshape(1,-1).T).ravel()
        cent = {vids[i]: float(sim[i]) for i in range(len(vids))}
    idf = _build_idf_over_bullets(summaries)
    now = time.time()
    scored=[]
    for v in universe:
        bullets = summaries.get(v, {}).get("bullets", [])
        pseudo = " ".join(b.get("text","") for b in bullets)[:3000]
        kw, overlap = _kw_score(pseudo, query, idf)
        if overlap < min_kw_overlap and pseudo:
            continue
        cs = cent.get(v, 0.0)
        rec = _recency_score(_vid_epoch(vm, v), now, half_life_days)
        base = 0.6*cs + 0.3*kw
        score = (1.0 - recency_weight)*base + recency_weight*(0.1*rec + 0.9*base)
        scored.append((v, score))
    scored.sort(key=lambda x:-x[1])
    return [v for v,_ in scored[:int(topK)]]

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

# ---------- Stage B search with guards ----------
def stageB_search_chunks(query:str,
    index:faiss.Index, embedder:SentenceTransformer,
    candidate_vids:Set[str] | None,
    initial_k:int, final_k:int, max_videos:int, per_video_cap:int,
    apply_mmr:bool, mmr_lambda:float,
    recency_weight:float, half_life_days:float, vm:dict)->List[Dict[str,Any]]:
    if index is None: return []
    qtok = _tokenize(query)
    qv=embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
    K = min(int(initial_k), index.ntotal if index.ntotal>0 else int(initial_k))
    D,I=index.search(qv.reshape(1,-1), K)
    idxs=[int(x) for x in I[0] if x>=0]
    base_sims=[float(s) for s in D[0][:len(idxs)]]

    rows=list(iter_jsonl_rows(idxs))
    texts=[]; metas=[]; keep=[]
    for _,j in rows:
        t=_normalize_text(j.get("text",""))
        m=(j.get("meta") or {}).copy()
        vid=(m.get("video_id") or m.get("vid") or m.get("ytid") or
             j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
        if not vid or not t: continue
        if candidate_vids is not None and vid not in candidate_vids: 
            continue
        if "start" not in m and "start_sec" in m: m["start"]=m.get("start_sec")
        m["start"]=_parse_ts(m.get("start",0))
        # gate by keyword overlap to prevent "peanut vs pork" errors
        if _overlap_score(qtok, _tokenize(t)) < 2:
            continue
        texts.append(t); metas.append({"video_id":vid, **m})

    if not texts: return []
    doc_vecs=embedder.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")

    # filter by cosine sim threshold vs query to remove tail matches
    sims=(doc_vecs @ qv.reshape(-1,1)).ravel()
    keep_idx=[i for i,s in enumerate(sims) if s>=0.22]  # conservative floor
    if not keep_idx: return []
    texts=[texts[i] for i in keep_idx]; metas=[metas[i] for i in keep_idx]
    doc_vecs=doc_vecs[keep_idx]; sims=sims[keep_idx]

    order=list(range(len(texts)))
    if apply_mmr:
        order=mmr(qv, doc_vecs, k=min(len(texts), max(8, final_k*2)), lambda_diversity=float(mmr_lambda))

    now=time.time(); blended=[]
    for li in order:
        m=metas[li]; vid=m.get("video_id")
        rec=_recency_score(_vid_epoch(vm,vid), now, half_life_days)
        score=(1.0-recency_weight)*float(sims[li]) + recency_weight*float(rec)
        blended.append((score, texts[li], m))
    blended.sort(key=lambda x:-x[0], reverse=True)

    picked=[]; seen_per_video={}; distinct=[]
    for sc,tx,me in blended:
        vid=me.get("video_id","Unknown")
        if vid not in distinct and len(distinct)>=int(max_videos): continue
        if seen_per_video.get(vid,0)>=int(per_video_cap): continue
        if vid not in distinct: distinct.append(vid)
        seen_per_video[vid]=seen_per_video.get(vid,0)+1
        picked.append({"score":float(sc),"text":tx,"meta":me})
        if len(picked)>=int(final_k): break
    return picked

# ---------- Evidence builder ----------
def group_hits_by_video(hits:List[Dict[str,Any]])->Dict[str,List[Dict[str,Any]]]:
    g={}
    for h in hits:
        vid=(h.get("meta") or {}).get("video_id") or "Unknown"
        g.setdefault(vid,[]).append(h)
    return g

def _first_bullets(vid:str, summaries:dict, k:int=2)->List[dict]:
    out=[]
    for b in summaries.get(vid,{}).get("bullets", []):
        q=_normalize_text(b.get("text",""))
        if _quote_is_valid(q):
            out.append({"text":q, "ts":b.get("ts",0.0)})
            if len(out)>=k: break
    return out

def build_grouped_evidence_for_prompt(
    hits:List[Dict[str,Any]], vm:dict, summaries:dict,
    routed_vids:List[str], max_quotes:int=3
)->Tuple[str,Dict[str,Any]]:
    groups=group_hits_by_video(hits)
    order_vids=[]; seen=set()
    for v in routed_vids:
        if v not in seen: order_vids.append(v); seen.add(v)
    for v in groups.keys():
        if v not in seen: order_vids.append(v); seen.add(v)

    lines=[]; export=[]
    for v_idx, vid in enumerate(order_vids, 1):
        info=vm.get(vid,{})
        if not info and vid not in groups: 
            continue
        title=info.get("title") or summaries.get(vid,{}).get("title") or vid
        creator_raw=_raw_creator_of_vid(vid, vm)
        creator=_canonicalize_creator(creator_raw) or creator_raw
        date=info.get("published_at") or info.get("publishedAt") or info.get("date") or ""
        url=info.get("url") or ""
        lines.append(f"[Video {v_idx}] {title} — {creator}" + (f" — {date}" if date else "") + (f" — {url}" if url else ""))

        items = groups.get(vid, [])
        clean=_dedupe_passages(items, time_window_sec=8.0, min_chars=40)
        used=0
        for h in clean[:max_quotes]:
            t0 = float((h.get("meta") or {}).get("start",0))
            ts = int(max(0, round(t0)))
            q=_normalize_text(h.get("text",""))
            lines.append(f"  • {datetime.utcfromtimestamp(ts).strftime('%-M:%S') if ts<3600 else datetime.utcfromtimestamp(ts).strftime('%-H:%M:%S')}: “{q[:260]}”")
            export.append({"video_id":vid,"title":title,"creator":creator,"ts":ts,"text":q,"url": _yt_ts_url(url, t0) if url else url})
            used+=1

        if used==0:
            for b in _first_bullets(vid, summaries, k=min(2,max_quotes)):
                t0=float(b["ts"] or 0.0); ts=int(max(0,round(t0)))
                q=_normalize_text(b["text"])
                lines.append(f"  • {datetime.utcfromtimestamp(ts).strftime('%-M:%S') if ts<3600 else datetime.utcfromtimestamp(ts).strftime('%-H:%M:%S')}: “{q[:260]}”")
                export.append({"video_id":vid,"title":title,"creator":creator,"ts":ts,"text":q,"url": _yt_ts_url(url, t0) if url else url})
                used+=1
                if used>=max_quotes: break

        if used==0 and items:
            h=items[0]; t0=float((h.get("meta") or {}).get("start",0)); ts=int(max(0,round(t0)))
            q=_normalize_text(h.get("text",""))
            lines.append(f"  • {datetime.utcfromtimestamp(ts).strftime('%-M:%S') if ts<3600 else datetime.utcfromtimestamp(ts).strftime('%-H:%M:%S')}: “{q[:260]}”")
            export.append({"video_id":vid,"title":title,"creator":creator,"ts":ts,"text":q,"url": _yt_ts_url(url, t0) if url else url})

        lines.append("")
    return "\n".join(lines).strip(), {"videos": export}

# ---------- Trusted web ----------
def _ddg_html(domain:str, query:str, headers:dict, timeout:float)->List[str]:
    try:
        r=requests.get("https://duckduckgo.com/html/", params={"q":f"site:{domain} {query}"}, headers=headers, timeout=timeout)
        if r.status_code!=200: return []
        soup=BeautifulSoup(r.text,"html.parser")
        return [a.get("href") for a in soup.select("a.result__a") if a.get("href")]
    except Exception:
        return []

def _ddg_lite(domain:str, query:str, headers:dict, timeout:float)->List[str]:
    try:
        r=requests.get("https://duckduckgo.com/lite/", params={"q":f"site:{domain} {query}"}, headers=headers, timeout=timeout)
        if r.status_code!=200: return []
        soup=BeautifulSoup(r.text,"html.parser")
        return [a.get("href") for a in soup.find_all("a") if a.get("href","").startswith("http")]
    except Exception:
        return []

def fetch_trusted_snippets(query:str, allowed_domains:List[str], max_snippets:int=3, per_domain:int=1, timeout:float=6.0):
    trace=[]
    out=[]
    if not requests or not BeautifulSoup or max_snippets<=0:
        st.session_state["web_trace"]="requests/bs4 unavailable."
        return []
    headers={"User-Agent":"Mozilla/5.0"}
    for domain in allowed_domains:
        links=_ddg_html(domain, query, headers, timeout)
        if not links:
            links=_ddg_lite(domain, query, headers, timeout)
            trace.append(f"{domain}: lite({len(links)})")
        else:
            trace.append(f"{domain}: html({len(links)})")
        if not links:
            links=[f"https://{domain}"]
            trace.append(f"{domain}: homepage")
        links=links[:per_domain]
        for url in links:
            try:
                r=requests.get(url, headers=headers, timeout=timeout)
                if r.status_code!=200:
                    trace.append(f"{domain}: {url} -> {r.status_code}")
                    continue
                soup=BeautifulSoup(r.text,"html.parser")
                paras=[p.get_text(" ",strip=True) for p in soup.find_all("p")]
                text=_normalize_text(" ".join(paras))[:1800]
                if len(text)<200:
                    trace.append(f"{domain}: short {url}")
                    continue
                out.append({"domain":domain,"url":url,"text":text})
            except Exception as e:
                trace.append(f"{domain}: err {e}")
        if len(out)>=max_snippets: break
    st.session_state["web_trace"]="; ".join(trace) if trace else "no trace"
    return out[:max_snippets]

# ---------- LLM with streaming ----------
def stream_answer(model_name: str, question: str, history, grouped_video_block: str,
                  web_snips: list[dict], no_video: bool):
    if not os.getenv("OPENAI_API_KEY"):
        yield "⚠️ OPENAI_API_KEY is not set."; return
    recent = [m for m in history[-6:] if m.get("role") in ("user","assistant")]
    convo = [("User: " if m["role"]=="user" else "Assistant: ")+m.get("content","") for m in recent]

    web_lines = [
        f"(W{j}) {s.get('domain','web')} — {s.get('url','')}\n“{_normalize_text(s.get('text',''))[:300]}”"
        for j,s in enumerate(web_snips,1)
    ]
    web_block = "\n".join(web_lines) if web_lines else "None"
    fallback_line = ("If no suitable video evidence exists, answer from trusted web snippets alone, "
                     "and begin with: 'Web-only evidence'.\n") if (WEB_FALLBACK and no_video) else \
                    "Trusted web snippets are supporting evidence.\n"

    system = (
        "Answer only from the provided VIDEO quotes and trusted WEB snippets. "
        "Priority: (1) grouped VIDEO quotes from allowed experts, (2) trusted WEB.\n"
        + fallback_line +
        "Rules:\n"
        "• Cite every claim: (Video k) and (DOMAIN Wj).\n"
        "• Flag animal/in-vitro vs human clinical.\n"
        "• Normalize units and include numeric effects if present.\n"
        "• Give a practical protocol and safety notes; no diagnosis."
    )

    user_payload = (("Recent conversation:\n"+"\n".join(convo)+"\n\n") if convo else "") + \
                   f"Question: {question}\n\n" + \
                   "Grouped Video Evidence:\n" + (grouped_video_block or "None") + "\n\n" + \
                   "Trusted Web Snippets:\n" + web_block + "\n\n" + \
                   "Write a detailed, well-grounded answer with explicit citations."

    try:
        client = OpenAI(timeout=60)
        stream = client.chat.completions.create(
            model=model_name, temperature=0.2, stream=True,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user_payload}]
        )
        for chunk in stream:
            delta = getattr(getattr(chunk,"choices",[None])[0], "delta", None)
            if not delta: continue
            part = getattr(delta, "content", None)
            if part: yield part
    except Exception as e:
        yield f"\n\n⚠️ Generation error: {e}"

# ---------- UI ----------
st.set_page_config(page_title="Health | Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Health | Nutrition Q&A")

if "turns" not in st.session_state: st.session_state["turns"]=[]
if "messages" not in st.session_state: st.session_state["messages"]=[]

with st.sidebar:
    st.markdown("**Auto Mode** · accuracy and diversity enabled")
    vm_lazy_slot = st.empty()
    # Trusted sites
    st.subheader("Trusted sites")
    st.caption("Short excerpts from vetted medical sites are added as supporting evidence.")
    allow_web = st.checkbox("Include supporting website excerpts", value=True)
    selected_domains=[]
    for i,dom in enumerate(TRUSTED_DOMAINS):
        if st.checkbox(dom, value=True, key=f"site_{i}"):
            selected_domains.append(dom)
    max_web_auto = 3

    # Model choice
    model_choice = st.selectbox(
        "Answering model", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0
    )

    # Advanced
    with st.expander("Advanced (technical controls)", expanded=False):
        st.caption("Defaults are tuned.")
        st.number_input("Scan candidates first (K)", 128, 5000, 768, 64, key="adv_scanK")
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
    st.checkbox("offsets built", value= OFFSETS_NPY.exists(), disabled=True)
    st.checkbox("faiss.index present", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("metas.pkl present", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("video_meta.json present", value=VIDEO_META_JSON.exists(), disabled=True)
    cent_ready = VID_CENT_NPY.exists() and VID_IDS_TXT.exists() and VID_SUM_JSON.exists()
    st.checkbox("centroids+ids+summaries present", value=cent_ready, disabled=True)

# Lazy creator inventory to speed first paint
with st.sidebar:
    if st.button("Show experts inventory"):
        vm_now = load_video_meta()
        vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm_now)
        counts={canon: len(creator_to_vids.get(canon,set())) for canon in ALLOWED_CREATORS}
        st.subheader("Experts")
        st.caption("Uncheck to exclude.")
        selected_creators_list=[]
        for i, canon in enumerate(ALLOWED_CREATORS):
            label=f"{canon} ({counts.get(canon,0)})"
            if st.checkbox(label, value=True, key=f"exp_{i}"):
                selected_creators_list.append(canon)
        st.session_state["selected_creators"]=set(selected_creators_list)

# Diagnostics area
if show_diag:
    colA,colB,colC = st.columns([2,3,3])
    with colA: st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    with colB: st.caption(f"chunks mtime: {datetime.fromtimestamp(_file_mtime(CHUNKS_PATH)).isoformat() if CHUNKS_PATH.exists() else 'missing'}")
    with colC: st.caption(f"index mtime: {datetime.fromtimestamp(_file_mtime(INDEX_PATH)).isoformat() if INDEX_PATH.exists() else 'missing'}")
    if _is_admin():
        st.code(json.dumps(verify_domain_artifacts(), indent=2))

# Render prior chat
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Admin maintenance
if _is_admin():
    st.subheader("Diagnostics (admin)")
    cols_dbg = st.columns(3)
    def _run_precompute_inline()->str:
        try:
            from scripts import precompute_video_summaries as pvs  # type: ignore
            pvs.main(); return "ok: rebuilt via import"
        except Exception as e:
            return f"precompute error: {e}"
    with cols_dbg[0]:
        if st.button("Rebuild precompute (admin)"):
            with st.spinner("Building centroids and summaries…"):
                msg=_run_precompute_inline()
            st.success(str(msg)); st.cache_resource.clear(); st.cache_data.clear(); st.rerun()
    with cols_dbg[1]:
        if st.button("Repair centroid norms"):
            try:
                C=np.load(VID_CENT_NPY).astype("float32")
                C=C/(np.linalg.norm(C,axis=1,keepdims=True)+1e-12)
                np.save(VID_CENT_NPY, C)
                st.success("ok: renormalized and saved")
            except Exception as e:
                st.error(f"repair error: {e}")
    with cols_dbg[2]:
        st.caption(f"summaries path: {VID_SUM_JSON}")

# --------------- Prompt input ---------------
prompt = st.chat_input("Ask about ApoB/LDL drugs, sleep, protein timing, fasting, supplements…")
if prompt is None:
    if st.session_state["turns"]:
        st.subheader("Previous replies and their sources")
        for i, t in enumerate(st.session_state["turns"], 1):
            with st.expander(f"Turn {i}: {t.get('prompt','')[:80]}", expanded=False):
                st.markdown(t.get("answer",""))
                vids = t.get("videos",[])
                web  = t.get("web",[])
                if vids:
                    with st.expander("Video quotes", expanded=False):
                        cur_vid=None; vm=load_video_meta()
                        for v in vids:
                            if v.get("video_id")!=cur_vid:
                                meta = vm.get(v.get("video_id",""),{})
                                link = meta.get("url","")
                                hdr = f"- **{v.get('title','')}** — _{v.get('creator','')}_"
                                st.markdown(f"{hdr}" + (f"  •  [{link}]({link})" if link else ""))
                                cur_vid = v.get("video_id")
                            if v.get("url"):
                                st.markdown(f"  • **{int(v.get('ts',0))}s** — “{_normalize_text(v.get('text',''))[:180]}”  ·  [{v.get('url')}]({v.get('url')})")
                            else:
                                st.markdown(f"  • **{int(v.get('ts',0))}s** — “{_normalize_text(v.get('text',''))[:180]}”")
                if web:
                    with st.expander("Trusted websites", expanded=False):
                        for j, s in enumerate(web, 1):
                            st.markdown(f"- W{j}: [{s['domain']}]({s['url']}) — {_normalize_text(s['text'])[:220]}…")
                        if t.get("web_trace"):
                            st.caption(t["web_trace"])
    st.button("Clear chat", on_click=_clear_chat)
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

# Load core
try:
    index, _, payload = load_metas_and_model()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load index or encoder."); st.exception(e); st.stop()
embedder: SentenceTransformer = payload["embedder"]
vm = load_video_meta()
C, vid_list = load_video_centroids()
summaries = load_video_summaries()
domain_model, per_video_domain_probs, _ = load_domain_model()

# Allowed universe by experts (if selected)
vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
universe = set(vid_list or list(vm.keys()) or list(vid_to_creator.keys()))
selected = st.session_state.get("selected_creators", set(ALLOWED_CREATORS))
allowed_vids_all = {vid for vid in universe if vid_to_creator.get(vid) in selected} or universe

# Domain routing
domain_top = classify_query_domains(domain_model, prompt, top_k=3)

def _score_vid_by_domains(vid:str, domain_top:List[str])->float:
    if vid not in per_video_domain_probs or not domain_top: return 0.0
    dv = per_video_domain_probs.get(vid, {})
    return float(sum(dv.get(d,0.0) for d in domain_top) / max(1,len(domain_top)))

qv = embedder.encode([prompt], normalize_embeddings=True).astype("float32")[0]
topK_route = 5
recency_weight_auto = st.session_state.get("adv_rec", 0.20)
half_life_auto = st.session_state.get("adv_hl", 270)

if domain_top and per_video_domain_probs:
    scored = [(vid, _score_vid_by_domains(vid, domain_top)) for vid in allowed_vids_all]
    scored.sort(key=lambda x:-x[1])
    slice_set = {vid for vid,sc in scored if sc>0.0}
    allowed_for_route = slice_set if slice_set else allowed_vids_all
else:
    allowed_for_route = allowed_vids_all

# Summary+centroid routing
routed_vids = route_videos_by_summary(
    prompt, qv, summaries, vm, C, list(vid_list) if vid_list else list(universe),
    allowed_for_route, topK=topK_route,
    recency_weight=recency_weight_auto, half_life_days=half_life_auto,
    min_kw_overlap=2
)
if not routed_vids and domain_top and per_video_domain_probs and allowed_for_route:
    routed_vids = sorted(list(allowed_for_route), key=lambda v: -_score_vid_by_domains(v, domain_top))[:topK_route]

candidate_vids = set(routed_vids) if routed_vids else allowed_vids_all

# Stage B search
K_scan = st.session_state.get("adv_scanK", 768)
K_use  = st.session_state.get("adv_useK", 36)
max_videos = st.session_state.get("adv_maxvid", 5)
per_video_cap = st.session_state.get("adv_cap", 4)
use_mmr = st.session_state.get("adv_mmr", True)
mmr_lambda = st.session_state.get("adv_lam", 0.45)

with st.spinner("Scanning selected videos…"):
    try:
        hits = stageB_search_chunks(
            prompt, index, embedder, candidate_vids,
            initial_k=min(int(K_scan), index.ntotal if index is not None else int(K_scan)),
            final_k=int(K_use), max_videos=int(max_videos), per_video_cap=int(per_video_cap),
            apply_mmr=bool(use_mmr), mmr_lambda=float(mmr_lambda),
            recency_weight=float(recency_weight_auto), half_life_days=float(half_life_auto), vm=vm
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Search failed."); st.exception(e); st.stop()

# Web support
web_snips=[]
if allow_web and selected_domains and requests and BeautifulSoup and int(max_web_auto)>0:
    with st.spinner("Fetching trusted websites…"):
        web_snips = fetch_trusted_snippets(prompt, selected_domains, max_snippets=int(max_web_auto), per_domain=1)

# Build evidence
grouped_block, export_struct = build_grouped_evidence_for_prompt(hits, vm, summaries, routed_vids=routed_vids, max_quotes=3)

# Answer with streaming
with st.chat_message("assistant"):
    if not export_struct["videos"] and not web_snips:
        st.warning("No usable expert quotes. Showing web-only if available.")
        st.session_state.messages.append({"role":"assistant","content":"No relevant video evidence found."})
        st.button("Clear chat", on_click=_clear_chat)
        st.stop()

    holder = st.empty()
    text_buf=[]
    for chunk in stream_answer(model_choice, prompt, st.session_state.messages, grouped_block, web_snips, no_video=(len(export_struct["videos"])==0)):
        text_buf.append(chunk)
        holder.markdown("".join(text_buf))
    ans = "".join(text_buf)
    st.session_state.messages.append({"role":"assistant","content":ans})

    # Persist sources for THIS reply only
    this_turn = {
        "prompt": prompt,
        "answer": ans,
        "videos": export_struct["videos"],
        "web": web_snips,
        "web_trace": st.session_state.get("web_trace","")
    }
    st.session_state["turns"].append(this_turn)

    # Sources UI for this reply (collapsed)
    st.markdown("---")
    with st.expander("Sources for this reply", expanded=False):
        vids = this_turn["videos"]; web = this_turn["web"]
        if vids:
            with st.expander("Video quotes", expanded=False):
                cur_vid=None; vm_now=load_video_meta()
                for v in vids:
                    if v.get("video_id")!=cur_vid:
                        meta = vm_now.get(v.get("video_id",""),{})
                        link = meta.get("url","")
                        hdr = f"- **{v.get('title','')}** — _{v.get('creator','')}_"
                        st.markdown(f"{hdr}" + (f"  •  [{link}]({link})" if link else ""))
                        cur_vid = v.get("video_id")
                    if v.get("url"):
                        st.markdown(f"  • **{int(v.get('ts',0))}s** — “{_normalize_text(v.get('text',''))[:180]}”  ·  [{v.get('url')}]({v.get('url')})")
                    else:
                        st.markdown(f"  • **{int(v.get('ts',0))}s** — “{_normalize_text(v.get('text',''))[:180]}”")
        if web:
            with st.expander("Trusted websites", expanded=False):
                for j, s in enumerate(web, 1):
                    st.markdown(f"- W{j}: [{s['domain']}]({s['url']}) — {_normalize_text(s['text'])[:220]}…")
                if st.session_state.get("web_trace"):
                    st.caption(st.session_state["web_trace"])

# Footer
st.button("Clear chat", on_click=_clear_chat)
