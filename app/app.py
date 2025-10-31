# -*- coding: utf-8 -*-
"""
Health | Nutrition Q&A — experts-first RAG + domain routing + trusted web (Phase 2)

Key fixes in this build
- Precision routing: domain-aligned slice + summary/centroid routing with keyword overlap.
- Quote quality: keyword overlap + cosine gate; guarantee ≥2 quotes per routed video via bullets fallback.
- Source accuracy: timestamped YouTube links and titled trusted web links; per-turn immutable sources.
- Web reliability: dual DuckDuckGo modes, title extraction, homepage fallback, and trace logging.
- Robustness: automatic centroid renormalization, offsets persistence, turn snapshots, admin rebuilds.
- Embedder upgrade guard: accepts EMBEDDER_MODEL env (e.g., bge-base-en-v1.5, e5-base-v2) but stays compatible with the current FAISS index.
- UI: removed nested expanders; use tabs under source sections to avoid StreamlitAPIException.
"""

from __future__ import annotations
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS","1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")

from pathlib import Path
import sys, json, yaml, pickle, time, re, math, collections, uuid, hashlib
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

# Optional domain model
try:
    import joblib
except Exception:
    joblib=None

# ---------- Paths and constants ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(os.getenv("DATA_DIR","/var/data")).resolve()
DATA_DIR = DATA_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH     = DATA_DIR / "chunks/chunks.jsonl"
OFFSETS_NPY     = DATA_DIR / "chunks/chunks.offsets.npy"
INDEX_PATH      = DATA_DIR / "index/faiss.index"
METAS_PKL       = DATA_DIR / "index/metas.pkl"
VIDEO_META_JSON = DATA_DIR / "catalog/video_meta.json"
VID_CENT_NPY    = DATA_DIR / "index/video_centroids.npy"
VID_IDS_TXT     = DATA_DIR / "index/video_ids.txt"
VID_SUM_JSON    = DATA_DIR / "catalog/video_summaries.json"

DOMAIN_DIR        = DATA_DIR / "domain"
DOMAIN_MODEL      = DOMAIN_DIR / "domain_model.joblib"
DOMAIN_PROBS_JSON = DOMAIN_DIR / "domain_probs.json"
DOMAIN_PROBS_YAML = DOMAIN_DIR / "domain_probs.yaml"

REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]
WEB_FALLBACK = os.getenv("WEB_FALLBACK","true").strip().lower() in {"1","true","yes","on"}

SESSION_SNAP = DATA_ROOT / "session_last.json"
SESSIONS_FILE = DATA_ROOT / "sessions.json"

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

# ---------- Small utils ----------
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

def _yt_ts_url(url:str, start_sec:float)->str:
    if not url:
        return ""
    s = int(max(0, round(start_sec)))
    join = "&" if "?" in url else "?"
    return f"{url}{join}t={s}s"

def _sha256(p: Path) -> str:
    try:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1<<20), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

def _clear_chat():
    st.session_state["messages"]=[]; st.session_state["turns"]=[]; st.rerun()

# ---------- Admin gate ----------
def _is_admin()->bool:
    try: qp = st.query_params
    except Exception: return False
    if qp.get("admin","0")!="1": return False
    try: expected = st.secrets["ADMIN_KEY"]
    except Exception: expected = None
    if expected is None: return True
    return qp.get("key","")==str(expected)

# ---------- Domain artifacts bootstrap ----------
from shutil import copy2
def _ensure_domain_artifacts():
    repo_src = ROOT / "data" / "domain"
    DOMAIN_DIR.mkdir(parents=True, exist_ok=True)
    for fn in ["domain_model.joblib", "domain_probs.json", "domain_probs.yaml"]:
        s = repo_src / fn
        d = DOMAIN_DIR / fn
        if s.exists() and not d.exists():
            try: copy2(s, d)
            except Exception: pass
_ensure_domain_artifacts()

def verify_domain_artifacts():
    repo = ROOT / "data" / "domain"
    vol  = DOMAIN_DIR
    files = ["domain_model.joblib", "domain_probs.json", "domain_probs.yaml"]
    rows = []
    for fn in files:
        pr = repo / fn
        pv = vol  / fn
        rows.append({
            "file": fn,
            "repo_exists": pr.exists(), "repo_size": pr.stat().st_size if pr.exists() else 0, "repo_sha256": _sha256(pr) if pr.exists() else "",
            "vol_exists":  pv.exists(), "vol_size":  pv.stat().st_size if pv.exists() else 0, "vol_sha256":  _sha256(pv) if pv.exists() else "",
        })
    load_ok = None
    classes = []
    if joblib and (DOMAIN_MODEL.exists() or (repo/"domain_model.joblib").exists()):
        try:
            model_path = DOMAIN_MODEL if DOMAIN_MODEL.exists() else (repo/"domain_model.joblib")
            pipe = joblib.load(model_path)
            classes = list(getattr(pipe, "classes_", []))
            load_ok = True
        except Exception:
            load_ok = False
    return {"DATA_DIR": str(DATA_ROOT), "rows": rows, "model_load_ok": load_ok, "classes": classes}

# ---------- Creator mapping ----------
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
    if ("healthy" in toks and "immune" in toks) or "healthyimmunedoc" in toks:
        return "Healthy Immune Doc"
    if "diary" in toks and "ceo" in toks:
        return "The Diary of A CEO"
    if "huberman" in toks:
        return "Andrew Huberman"
    if "attia" in toks:
        return "Peter Attia MD"
    if "jamnadas" in toks:
        return "Dr. Pradip Jamnadas, MD"

    for canon in ALLOWED_CREATORS:
        if n == canon.lower(): return canon
        if re.sub(r"[^\w\s]","",n) == re.sub(r"[^\w\s]","",canon.lower()):
            return canon
    return None

# ---------- LLM answer streaming ----------
def stream_openai_answer(model_name: str, question: str, history, grouped_video_block: str,
                         web_snips: list[dict], no_video: bool):
    if not os.getenv("OPENAI_API_KEY"):
        yield "⚠️ OPENAI_API_KEY is not set."
        return

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
        "• Cite every claim: (Video k) per video, (DOMAIN Wj) for web.\n"
        "• Flag animal/in-vitro/mechanistic vs human clinical.\n"
        "• Normalize units; include numeric effect sizes if present.\n"
        "• Give practical options with mechanisms; 'dose not specified' if absent. No diagnosis.\n"
        "Structure: Key summary • Practical protocol • Safety notes.\n"
        "Do not invent beyond the evidence."
    )

    user_payload = (("Recent conversation:\n"+"\n".join(convo)+"\n\n") if convo else "") + \
                   f"Question: {question}\n\n" + \
                   "Grouped Video Evidence:\n" + (grouped_video_block or "None") + "\n\n" + \
                   "Trusted Web Snippets:\n" + web_block + "\n\n" + \
                   "Write a concise, well-grounded answer with explicit citations."

    client = OpenAI(timeout=60)
    stream = client.chat.completions.create(
        model=model_name, temperature=0.2, stream=True,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user_payload}]
    )
    for chunk in stream:
        ch = chunk.choices[0]
        delta = getattr(ch, "delta", None)
        if delta and getattr(delta, "content", None):
            yield delta.content

# ---------- Loaders ----------
@st.cache_data(show_spinner=False)
def load_video_meta()->Dict[str,Dict[str,Any]]:
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

# ---------- JSONL offsets ----------
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

# ---------- Model + FAISS ----------
@st.cache_resource(show_spinner=False)
def _load_embedder(model_name:str)->SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")

@st.cache_resource(show_spinner=False)
def load_metas_and_model(index_path:Path=INDEX_PATH, metas_path:Path=METAS_PKL):
    if not index_path.exists() or not metas_path.exists(): return None, None, None
    index = faiss.read_index(str(index_path))
    with metas_path.open("rb") as f:
        payload=pickle.load(f)
    idx_dim = index.d

    env_override = os.getenv("EMBEDDER_MODEL", "").strip()
    model_from_meta = payload.get("model","sentence-transformers/all-MiniLM-L6-v2")

    def _try_model(name:str):
        try:
            emb = _load_embedder(name)
            return emb, emb.get_sentence_embedding_dimension()
        except Exception:
            return None, None

    if env_override:
        emb, dim = _try_model(env_override)
        if emb and dim == idx_dim:
            return index, payload.get("metas",[]), {"model_name":env_override, "embedder":emb}

    emb, dim = _try_model(model_from_meta)
    if emb and dim == idx_dim:
        return index, payload.get("metas",[]), {"model_name":model_from_meta, "embedder":emb}

    local_dir = DATA_DIR/"models"/"all-MiniLM-L6-v2"
    try_name = str(local_dir) if (local_dir/"config.json").exists() else model_from_meta
    emb, dim = _try_model(try_name)
    if not emb or dim != idx_dim:
        raise RuntimeError(f"Embedding dim mismatch. FAISS={idx_dim}. Override EMBEDDER_MODEL or rebuild index.")
    return index, payload.get("metas",[]), {"model_name":try_name, "embedder":emb}

@st.cache_resource(show_spinner=False)
def load_video_centroids():
    if not (VID_CENT_NPY.exists() and VID_IDS_TXT.exists()): return None, None
    C = np.load(VID_CENT_NPY).astype("float32")
    if C.ndim==2 and C.size>0:
        n = np.linalg.norm(C,axis=1,keepdims=True) + 1e-12
        C = C / n
    vids = VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
    if C.ndim!=2 or C.shape[0]!=len(vids): return None, None
    return C, vids

@st.cache_data(show_spinner=False)
def load_video_summaries():
    if VID_SUM_JSON.exists():
        try:return json.loads(VID_SUM_JSON.read_text(encoding="utf-8"))
        except:return {}
    return {}

# ---------- Domain routing ----------
@st.cache_resource(show_spinner=False)
def load_domain_model():
    if not DOMAIN_MODEL.exists() or joblib is None:
        return None, {}, set()
    try:
        model = joblib.load(DOMAIN_MODEL)
    except Exception:
        model = None

    probs_raw = []
    if DOMAIN_PROBS_JSON.exists():
        try:
            probs_raw = json.loads(DOMAIN_PROBS_JSON.read_text(encoding="utf-8"))
        except Exception:
            probs_raw = []
    elif DOMAIN_PROBS_YAML.exists():
        try:
            probs_raw = yaml.safe_load(DOMAIN_PROBS_YAML.read_text(encoding="utf-8")) or []
        except Exception:
            probs_raw = []

    per_video: Dict[str, Dict[str,float]] = {}
    all_domains=set()
    for row in probs_raw:
        vid=row.get("video_id"); 
        if not vid: continue
        per_video.setdefault(vid, {})
        for key,val in row.items():
            if key=="video_id": continue
            try: p=float(val)
            except: 
                continue
            parts=[t.strip() for t in key.split(";") if t.strip()]
            for d in parts:
                all_domains.add(d)
                per_video[vid][d]=max(per_video[vid].get(d,0.0), p)

    return model, per_video, all_domains

def classify_query_domains(model, query:str, top_k:int=3, min_keep:int=1)->List[str]:
    if model is None: return []
    try:
        proba = model.predict_proba([query])[0]
        classes = list(model.classes_)
        ranked = sorted(zip(classes, proba), key=lambda x:-x[1])
        keep = [d for d,_ in ranked[:max(min_keep, top_k)]]
        return keep
    except Exception:
        return []

# ---------- MMR + quote filters ----------
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

def _kw_overlap(a:str, b:str)->int:
    A=set(re.findall(r"[a-z0-9]+", (a or "").lower()))
    B=set(re.findall(r"[a-z0-9]+", (b or "").lower()))
    return len(A & B)

def _quote_is_valid(text:str)->bool:
    t=_normalize_text(text)
    if len(t) < 40: return False
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

def _quote_relevance_ok(query:str, quote:str, qv:np.ndarray, embedder:SentenceTransformer,
                        min_overlap:int=3, min_cos:float=0.35)->bool:
    if _kw_overlap(query, quote) < min_overlap:
        return False
    try:
        t_vec = embedder.encode([quote], normalize_embeddings=True).astype("float32")[0]
        cos = float(np.dot(qv, t_vec))
        return cos >= min_cos
    except Exception:
        return True

# ---------- Summary-aware routing ----------
def _build_idf_over_bullets(summaries: dict) -> dict:
    DF = collections.Counter()
    vids = list(summaries.keys())
    for v in vids:
        for b in summaries.get(v, {}).get("bullets", []):
            toks = set(re.findall(r"[a-z0-9]+",(b.get("text","") or "").lower()))
            for w in toks: DF[w] += 1
    N = max(1, len(vids))
    return {w: math.log((N+1)/(df+0.5)) for w,df in DF.items()}

def _kw_score(text: str, query: str, idf: dict) -> Tuple[float,int]:
    if not text: return 0.0, 0
    qtok = re.findall(r"[a-z0-9]+", (query or "").lower())
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
    min_kw_overlap:int=1
) -> list[str]:
    universe = [v for v in (vids or list(vm.keys())) if (not allowed_vids or v in allowed_vids)]
    if not universe: return []
    cent = {}
    if C is not None and vids is not None and len(vids) == C.shape[0]:
        sim = (C @ qv.reshape(-1,1)).ravel()
        cent = {vids[i]: float(sim[i]) for i in range(len(vids))}
    idf = _build_idf_over_bullets(summaries or {})
    now = time.time()
    scored = []
    for v in universe:
        bullets = (summaries or {}).get(v, {}).get("bullets", [])
        pseudo = " ".join(b.get("text","") for b in bullets)[:3000]
        kw, overlap = _kw_score(pseudo, query, idf) if pseudo else (0.0, 0)
        if pseudo and overlap < min_kw_overlap:
            continue
        cs = cent.get(v, 0.0)
        rec = _recency_score(_vid_epoch(vm, v), now, half_life_days)
        base = 0.6*cs + 0.3*kw
        score = (1.0 - recency_weight)*base + recency_weight*(0.1*rec + 0.9*base)
        scored.append((v, score))
    scored.sort(key=lambda x:-x[1])
    return [v for v,_ in scored[:int(topK)]]

# ---------- Stage B recall ----------
def stageB_search_chunks(query:str,
    index:faiss.Index, embedder:SentenceTransformer,
    candidate_vids:Set[str] | None,
    initial_k:int, final_k:int, max_videos:int, per_video_cap:int,
    apply_mmr:bool, mmr_lambda:float,
    recency_weight:float, half_life_days:float, vm:dict)->List[Dict[str,Any]]:
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

# ---------- Evidence builder + fallbacks ----------
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
            out.append({"text":q, "ts":_format_ts(b.get("ts",0))})
            if len(out)>=k: break
    return out

def build_grouped_evidence_for_prompt(
    hits: List[Dict[str,Any]], vm:dict, summaries:dict,
    routed_vids: List[str], max_quotes:int=3,
    qv: np.ndarray | None = None, embedder: SentenceTransformer | None = None
) -> Tuple[str, Dict[str, Any]]:
    groups = group_hits_by_video(hits)
    order_vids, seen = [], set()
    for v in routed_vids:
        if v not in seen: order_vids.append(v); seen.add(v)
    for v in groups.keys():
        if v not in seen: order_vids.append(v); seen.add(v)

    lines, export = [], []
    query_txt = st.session_state.get("_last_query", "")

    for v_idx, vid in enumerate(order_vids, 1):
        info=vm.get(vid,{}) or {}
        if (vid not in groups) and not (summaries or {}).get(vid):
            continue
        title=info.get("title") or (summaries or {}).get(vid,{}).get("title") or vid
        creator_raw=_raw_creator_of_vid(vid, vm)
        creator=_canonicalize_creator(creator_raw) or creator_raw
        date=info.get("published_at") or info.get("publishedAt") or info.get("date") \
             or (summaries or {}).get(vid,{}).get("published_at") or ""
        base_url = info.get("url") or f"https://www.youtube.com/watch?v={vid}"

        lines.append(f"[Video {v_idx}] {title} — {creator}" 
                     + (f" — {date}" if date else "")
                     + (f" — {base_url}" if base_url else ""))

        items = groups.get(vid, [])
        clean=_dedupe_passages(items, time_window_sec=8.0, min_chars=40)

        kept=[]
        if qv is not None and embedder is not None:
            for h in clean:
                q = _normalize_text(h.get("text",""))
                if _quote_relevance_ok(query_txt, q, qv, embedder, min_overlap=3, min_cos=0.35):
                    kept.append(h)
        else:
            kept = clean

        used=0
        for h in kept[:max_quotes]:
            t0=float((h.get("meta") or {}).get("start",0))
            ts=_format_ts(t0)
            q=_normalize_text(h.get("text",""))
            lines.append(f"  • {ts}: “{q[:260]}”")
            export.append({"video_id":vid,"title":title,"creator":creator,"ts":ts,"text":q,"url": _yt_ts_url(base_url, t0)})
            used+=1

        need = max(0, max(2, min(2, max_quotes)) - used)
        if need > 0:
            for b in _first_bullets(vid, summaries or {}, k=need+1):
                ts_txt = b["ts"]
                try_sec = 0.0
                if ts_txt:
                    parts = [float(x) for x in ts_txt.split(":")]
                    try_sec = sum(p * (60 ** i) for i, p in enumerate(reversed(parts)))
                lines.append(f"  • {ts_txt}: “{b['text'][:260]}”")
                export.append({"video_id":vid,"title":title,"creator":creator,"ts":ts_txt,"text":b["text"],"url": _yt_ts_url(base_url, try_sec)})
                used+=1
                if used>=max_quotes: break

        if used==0 and items:
            h=items[0]
            t0=float((h.get("meta") or {}).get("start",0))
            ts=_format_ts(t0)
            q=_normalize_text(h.get("text",""))
            lines.append(f"  • {ts}: “{q[:260]}”")
            export.append({"video_id":vid,"title":title,"creator":creator,"ts":ts,"text":q,"url": _yt_ts_url(base_url, t0)})

        lines.append("")

    return "\n".join(lines).strip(), {"videos": export}

# ---------- Web fetch with titles ----------
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

def fetch_trusted_snippets(query: str, allowed_domains: List[str],
                           max_snippets: int = 3, per_domain: int = 1, timeout: float = 6.0):
    trace, out = [], []
    if not requests or not BeautifulSoup or max_snippets <= 0:
        st.session_state["web_trace"] = "requests/bs4 unavailable."
        return []

    headers = {"User-Agent": "Mozilla/5.0"}

    for domain in allowed_domains:
        links = _ddg_html(domain, query, headers, timeout)
        if not links:
            links=_ddg_lite(domain, query, headers, timeout)
            trace.append(f"{domain}: lite links={len(links)}")
        else:
            trace.append(f"{domain}: html links={len(links)}")

        if not links:
            links=[f"https://{domain}"]
            trace.append(f"{domain}: homepage fallback")

        for url in links[:per_domain]:
            try:
                r = requests.get(url, headers=headers, timeout=timeout)
                if r.status_code != 200:
                    trace.append(f"{domain}: {url} status {r.status_code}")
                    continue
                soup = BeautifulSoup(r.text, "html.parser")
                title = _normalize_text(soup.title.get_text(strip=True) if soup.title else "")
                paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
                text = _normalize_text(" ".join(paras))[:1800]
                if len(text) < 200:
                    trace.append(f"{domain}: {url} too short")
                    continue
                out.append({"domain": domain, "url": url, "title": title, "text": text})
            except Exception as e:
                trace.append(f"{domain}: fetch error {e}")
        if len(out) >= max_snippets: break

    st.session_state["web_trace"] = "; ".join(trace) if trace else "no trace"
    return out[:max_snippets]

# ---------- Precompute + repairs (admin only) ----------
def _run_precompute_inline()->str:
    try:
        try:
            from scripts import precompute_video_summaries as pvs  # type: ignore
            pvs.main()
            return "ok: rebuilt via package import"
        except Exception:
            import importlib.util
            p = ROOT / "scripts" / "precompute_video_summaries.py"
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
        if not VID_CENT_NPY.exists(): return "centroids missing"
        C = np.load(VID_CENT_NPY).astype("float32")
        n = np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        C = C / n
        np.save(VID_CENT_NPY, C)
        return "ok: renormalized and saved"
    except Exception as e:
        return f"repair error: {e}"

def _build_summaries_fallback(max_lines_per_video: int = 800) -> str:
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
        "domain_model": str(DOMAIN_MODEL), "domain_model_exists": DOMAIN_MODEL.exists(),
        "domain_probs_json": str(DOMAIN_PROBS_JSON), "domain_probs_json_exists": DOMAIN_PROBS_JSON.exists(),
        "domain_probs_yaml": str(DOMAIN_PROBS_YAML), "domain_probs_yaml_exists": DOMAIN_PROBS_YAML.exists(),
    }

# ---------- Sessions ----------
def _get_or_make_session_id()->str:
    if "_session_uuid" not in st.session_state:
        st.session_state["_session_uuid"]=str(uuid.uuid4())
    return st.session_state["_session_uuid"]

def _record_session_hit():
    try:
        sid=_get_or_make_session_id()
        if SESSIONS_FILE.exists():
            d=json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        else:
            d={}
        now=int(time.time())
        if sid in d:
            d[sid]["last"]=now
            d[sid]["hits"]=d[sid].get("hits",0)+1
        else:
            d[sid]={"first":now,"last":now,"hits":1}
        SESSIONS_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass

def _unique_sessions_count()->int:
    try:
        if not SESSIONS_FILE.exists(): return 0
        d=json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        return len(d.keys())
    except Exception:
        return 0

def _save_turn_snapshot(turn:dict):
    try:
        SESSION_SNAP.write_text(json.dumps(turn, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

# ---------- Creator indexes ----------
def build_creator_indexes_from_chunks(vm: dict) -> tuple[dict, dict]:
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

# ======================= UI =======================

st.set_page_config(page_title="Health | Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Health | Nutrition Q&A")

if "turns" not in st.session_state: st.session_state["turns"]=[]
if "messages" not in st.session_state: st.session_state["messages"]=[]

_record_session_hit()

with st.sidebar:
    st.markdown("**Auto Mode** · accuracy and diversity enabled")

    vm = load_video_meta()
    vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
    counts={canon: len(creator_to_vids.get(canon,set())) for canon in ALLOWED_CREATORS}

    st.subheader("Experts")
    st.caption("Select which experts to include in search and answers.")
    selected_creators_list=[]
    for i, canon in enumerate(ALLOWED_CREATORS):
        label=f"{canon} ({counts.get(canon,0)})"
        if st.checkbox(label, value=True, key=f"exp_{i}", help="Include or exclude content from this expert."):
            selected_creators_list.append(canon)
    selected_creators:set[str]=set(selected_creators_list)
    st.session_state["selected_creators"]=selected_creators

    st.subheader("Trusted sites")
    st.caption("Short excerpts from vetted medical sites are added as supporting evidence.")
    allow_web = st.checkbox("Include supporting website excerpts", value=True,
                            help="Adds brief excerpts from reputable medical sites to supplement the video evidence.")
    selected_domains=[]
    for i,dom in enumerate(TRUSTED_DOMAINS):
        if st.checkbox(dom, value=True, key=f"site_{i}", help="Allow fetching supporting evidence from this domain."):
            selected_domains.append(dom)
    max_web_auto = 3

    model_choice = st.selectbox(
        "Answering model",
        ["gpt-4o","gpt-4o-mini","gpt-4.1-mini"], index=0,
        help="Model used to compose the final answer from the selected evidence."
    )

    with st.expander("Advanced (technical controls)", expanded=False):
        st.caption("Defaults are tuned.")
        st.number_input("Scan candidates first (K)", 128, 5000, 768, 64, key="adv_scanK")
        st.number_input("Use top passages", 8, 200, 48, 2, key="adv_useK")
        st.number_input("Max videos", 3, 20, 8, 1, key="adv_maxvid")
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
    st.checkbox("metas.pkl present", value=METAS_PKL.exists(), disabled=True)
    cent_ready = VID_CENT_NPY.exists() and VID_IDS_TXT.exists() and VID_SUM_JSON.exists()
    st.checkbox("centroids+ids+summaries present", value=cent_ready, disabled=True)

if show_diag:
    colA,colB,colC = st.columns([2,3,3])
    with colA: st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    with colB: st.caption(f"chunks mtime: {_iso(_file_mtime(CHUNKS_PATH)) if CHUNKS_PATH.exists() else 'missing'}")
    with colC: st.caption(f"index mtime: {_iso(_file_mtime(INDEX_PATH)) if INDEX_PATH.exists() else 'missing'}")

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Admin diagnostics block
if _is_admin():
    st.subheader("Diagnostics (admin)")
    st.code(json.dumps(_path_exists_report(), indent=2))
    st.markdown(f"**Approx unique sessions:** {_unique_sessions_count()}")
    if SESSION_SNAP.exists():
        st.markdown(f"**Last turn snapshot:** {_iso(_file_mtime(SESSION_SNAP))}")

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

    with st.expander("Creator inventory (from chunks.jsonl)"):
        inv = sorted(((c, len(vs)) for c,vs in build_creator_indexes_from_chunks(load_video_meta())[1].items()),
                     key=lambda x: -x[1])
        st.dataframe([{"creator": c, "videos": n} for c,n in inv], use_container_width=True)

    with st.expander("Domain artifacts check"):
        st.code(json.dumps(verify_domain_artifacts(), indent=2))

# --------------- Prompt input ---------------
prompt = st.chat_input("Ask about ApoB/LDL drugs, sleep, protein timing, fasting, supplements, protocols…")
if prompt is None:
    if st.session_state["turns"]:
        st.subheader("Previous replies and their sources")
        for i, t in enumerate(st.session_state["turns"], 1):
            with st.expander(f"Turn {i}: {t.get('prompt','')[:80]}", expanded=False):
                st.markdown(t.get("answer",""))
                vids = t.get("videos",[])
                web  = t.get("web",[])
                tabs = st.tabs(["Video quotes","Trusted websites","Web fetch trace"])
                with tabs[0]:
                    if vids:
                        cur_vid=None
                        vm_prev = load_video_meta()
                        for v in vids:
                            if v.get("video_id")!=cur_vid:
                                meta = vm_prev.get(v.get("video_id",""),{})
                                link = meta.get("url","") or f"https://www.youtube.com/watch?v={v.get('video_id','')}"
                                hdr = f"- **{v.get('title','')}** — _{v.get('creator','')}_"
                                st.markdown(f"{hdr}" + (f"  •  [{link}]({link})" if link else ""))
                                cur_vid = v.get("video_id")
                            url = v.get("url") or ""
                            if url:
                                st.markdown(f"  • **{v.get('ts','')}** — “{_normalize_text(v.get('text',''))[:180]}”  ·  [{url}]({url})")
                            else:
                                st.markdown(f"  • **{v.get('ts','')}** — “{_normalize_text(v.get('text',''))[:180]}”")
                    else:
                        st.caption("No video quotes saved for this turn.")
                with tabs[1]:
                    if web:
                        for j, s in enumerate(web, 1):
                            title = s.get("title") or s["domain"]
                            st.markdown(f"- W{j}: [{title}]({s['url']}) · {s['domain']}")
                    else:
                        st.caption("No trusted web snippets saved for this turn.")
                with tabs[2]:
                    trace = t.get("web_trace","")
                    if trace:
                        st.code(trace)
                    else:
                        st.caption("No fetch trace recorded.")
    cols=st.columns([1]*12)
    with cols[-1]:
        st.button("Clear chat", key="clear_idle", on_click=_clear_chat)
    st.stop()

# store user message
st.session_state["messages"].append({"role":"user","content":prompt})
with st.chat_message("user"): st.markdown(prompt)

# Guardrails
missing=[p for p in REQUIRED if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error("Missing required files:\n" + "\n".join(f"- {p}" for p in missing))
    st.stop()

# Load FAISS + encoder + summaries + domain router
try:
    index, _, payload = load_metas_and_model()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load index or encoder."); st.exception(e); st.stop()
embedder: SentenceTransformer = payload["embedder"]
try: embedder.encode(["warmup"], normalize_embeddings=True)
except Exception: pass

vm = load_video_meta()
C, vid_list = load_video_centroids()
summaries = load_video_summaries() or {}
domain_model, per_video_domain_probs, domain_set = load_domain_model()

# Selected experts
vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
universe = set(vid_list or list(vm.keys()) or list(vid_to_creator.keys()))
chosen = st.session_state.get("selected_creators", set(ALLOWED_CREATORS))
allowed_vids_all = {vid for vid in universe if vid_to_creator.get(vid) in chosen}

# Domain routing
qv = embedder.encode([prompt], normalize_embeddings=True).astype("float32")[0]
st.session_state["_last_query"] = prompt
st.session_state["_last_qv"] = qv

domain_top = classify_query_domains(domain_model, prompt, top_k=3, min_keep=1)

def _domain_score(vid: str) -> float:
    if not domain_top or not per_video_domain_probs.get(vid):
        return 0.0
    dv = per_video_domain_probs[vid]
    return float(sum(dv.get(d, 0.0) for d in domain_top) / max(1, len(domain_top)))

topK_route = 12
recency_weight_auto = 0.20
half_life_auto = 270

allowed_for_route = allowed_vids_all
if domain_top and per_video_domain_probs:
    MIN_DOMAIN_SCORE = 0.15
    candidates = [(vid, _domain_score(vid)) for vid in allowed_vids_all]
    allowed_for_route = {vid for vid, s in candidates if s >= MIN_DOMAIN_SCORE} or allowed_vids_all

routed_vids = route_videos_by_summary(
    prompt, qv, summaries, vm, C, list(vid_list) if vid_list else list(universe), allowed_for_route,
    topK=topK_route, recency_weight=recency_weight_auto, half_life_days=half_life_auto, min_kw_overlap=1
)
if not routed_vids and domain_top and per_video_domain_probs and allowed_for_route:
    alt = sorted(list(allowed_for_route), key=lambda v: -_domain_score(v))[:topK_route]
    routed_vids = alt

candidate_vids = set(routed_vids) if routed_vids else allowed_vids_all

# Recall knobs
K_scan = st.session_state.get("adv_scanK", 768)
K_use  = st.session_state.get("adv_useK", 48)
min_distinct_videos = 3
max_videos = max(int(st.session_state.get("adv_maxvid", 8)), min_distinct_videos)
per_video_cap = max(int(st.session_state.get("adv_cap", 4)), 2)
use_mmr = st.session_state.get("adv_mmr", True)
mmr_lambda = st.session_state.get("adv_lam", 0.45)
recency_weight = st.session_state.get("adv_rec", recency_weight_auto)
half_life = st.session_state.get("adv_hl", half_life_auto)

# Stage B
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

# Trusted web
web_snips=[]
if allow_web and requests and BeautifulSoup and int(max_web_auto)>0:
    with st.spinner("Fetching trusted websites…"):
        domains = selected_domains[:] if selected_domains else TRUSTED_DOMAINS
        web_snips = fetch_trusted_snippets(prompt, domains, max_snippets=int(max_web_auto), per_domain=1)
        if len(web_snips) < 2:
            more = fetch_trusted_snippets(prompt, TRUSTED_DOMAINS, max_snippets=2, per_domain=1)
            seen = {(s["domain"], s["url"]) for s in web_snips}
            for s in more:
                key=(s["domain"], s["url"])
                if key not in seen:
                    web_snips.append(s); seen.add(key)
                if len(web_snips)>=2: break

# Build evidence and answer
grouped_block, export_struct = build_grouped_evidence_for_prompt(
    hits, vm, summaries, routed_vids=routed_vids, max_quotes=3, qv=qv, embedder=embedder
)

with st.chat_message("assistant"):
    if not export_struct["videos"] and not web_snips:
        st.warning("No usable expert quotes. Showing web-only if available.")
        st.session_state["messages"].append({"role":"assistant","content":"No relevant video evidence found."})
        cols=st.columns([1]*12)
        with cols[-1]:
            st.button("Clear chat", key="clear_nohits", on_click=_clear_chat)
        st.stop()

    with st.spinner("Writing your answer…"):
        gen = stream_openai_answer(model_choice, prompt, st.session_state["messages"],
                                   grouped_block, web_snips, no_video=(len(export_struct["videos"])==0))
        ans = st.write_stream(gen)

    st.session_state["messages"].append({"role":"assistant","content":ans})

    this_turn = {
        "prompt": prompt,
        "answer": ans,
        "videos": export_struct["videos"],
        "web": web_snips,
        "web_trace": st.session_state.get("web_trace","")
    }
    st.session_state["turns"].append(this_turn)
    _save_turn_snapshot(this_turn)

    st.markdown("---")
    with st.expander("Sources for this reply", expanded=False):
        vids = this_turn["videos"]; web = this_turn["web"]
        tabs = st.tabs(["Video quotes","Trusted websites","Web fetch trace"])
        with tabs[0]:
            if vids:
                cur_vid=None
                for v in vids:
                    if v.get("video_id")!=cur_vid:
                        meta = vm.get(v.get("video_id",""),{})
                        link = meta.get("url","") or f"https://www.youtube.com/watch?v={v.get('video_id','')}"
                        hdr = f"- **{v.get('title','')}** — _{v.get('creator','')}_"
                        st.markdown(f"{hdr}" + (f"  •  [{link}]({link})" if link else ""))
                        cur_vid = v.get("video_id")
                    url = v.get("url") or ""
                    if url:
                        st.markdown(f"  • **{v.get('ts','')}** — “{_normalize_text(v.get('text',''))[:180]}”  ·  [{url}]({url})")
                    else:
                        st.markdown(f"  • **{v.get('ts','')}** — “{_normalize_text(v.get('text',''))[:180]}”")
            else:
                st.caption("No video quotes used.")
        with tabs[1]:
            if web:
                for j, s in enumerate(web, 1):
                    title = s.get("title") or s["domain"]
                    st.markdown(f"- W{j}: [{title}]({s['url']}) · {s['domain']}")
            else:
                st.caption("No trusted web snippets.")
        with tabs[2]:
            trace = this_turn.get("web_trace","")
            if trace:
                st.code(trace)
            else:
                st.caption("No fetch trace recorded.")

cols=st.columns([1]*12)
with cols[-1]:
    st.button("Clear chat", key="clear_done", on_click=_clear_chat)
