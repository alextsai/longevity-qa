# app/app.py
# -*- coding: utf-8 -*-
"""
Longevity / Nutrition Q&A — experts-first RAG with trusted web support.

Key design
- Route then scan: rank videos by per-video bullets + centroid + recency, pick top 5, then search only those videos.
- Experts and trusted sites are vertical checklists. All selected by default. Per-expert video counts come from chunks.jsonl
  (falls back to video_meta.json) so mislabeled metadata does not hide videos.
- Admin panel: rebuild precompute, repair centroid norms, build summaries fallback, freshness checks, keyword scan.
- Sources export as JSON from the expander.

This file is self-contained and safe to drop in.
"""

from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS","1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")

from pathlib import Path
import sys, json, pickle, time, re
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
    st.rerun()

# ---------------- Admin gate ----------------
def _is_admin()->bool:
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

# ---------------- LLM answerer ----------------
def openai_answer(model_name: str, question: str, history, grouped_video_block: str,
                  web_snips: list[dict], no_video: bool) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ OPENAI_API_KEY is not set."
    recent = [m for m in history[-6:] if m.get("role") in ("user","assistant")]
    convo = [("User: " if m["role"]=="user" else "Assistant: ")+m.get("content","") for m in recent]

    web_lines = [
        f"(W{j}) {s.get('domain','web')} — {s.get('url','')}\n“{' '.join((s.get('text','')).split())[:300]}”"
        for j,s in enumerate(web_snips,1)
    ]
    web_block = "\n".join(web_lines) if web_lines else "None"

    fallback_line = ("If no suitable video evidence exists, you MAY answer from trusted web snippets alone, "
                     "but begin with: 'Web-only evidence'.\n") if no_video else \
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

# ---------------- Loaders ----------------
@st.cache_data(show_spinner=False)
def load_video_meta()->Dict[str,Dict[str,Any]]:
    if VIDEO_META_JSON.exists():
        try:return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except:return {}
    return {}

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

def _recency_score(published_ts:float, now:float, half_life_days:float)->float:
    if published_ts<=0: return 0.0
    days=max(0.0,(now-published_ts)/86400.0)
    return 0.5 ** (days / max(1e-6, half_life_days))

def precompute_status(embedder_model:str)->Dict[str,Any]:
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
                status["msg"].append("chunks.jsonl is newer than centroids → run precompute.")
        else:
            status["msg"].append("Centroids/IDs missing → run precompute.")
    except Exception as e:
        status["msg"].append(f"precompute check error: {e}")
    return status

# ---------------- JSONL offsets ----------------
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
    if not (VID_CENT_NPY.exists() and VID_IDS_TXT.exists()): return None, None
    C = np.load(VID_CENT_NPY).astype("float32")
    vids = VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
    if C.shape[0]!=len(vids): return None, None
    return C, vids

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

def _dedupe_passages(items:List[Dict[str,Any]], time_window_sec:float=8.0, min_chars:int=40):
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
    import math, collections
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

def route_videos_by_summary(
    query: str, qv: np.ndarray,
    summaries: dict, vm: dict,
    C: np.ndarray | None, vids: list[str] | None,
    allowed_vids: set[str],
    topK: int, recency_weight: float, half_life_days: float
) -> list[str]:
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
        kw = _kw_score(pseudo, query, idf)
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

# ---------------- Grouped evidence ----------------
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

        bullets = summaries.get(vid,{}).get("bullets", [])
        for b in bullets[:max_quotes]:
            ts=_format_ts(b.get("ts",0))
            q=_normalize_text(b.get("text","")); q=q[:260]+"…" if len(q)>260 else q
            lines.append(f"  • {ts}: “{q}”")

        need = max(0, max_quotes - len(bullets))
        if need>0:
            clean=_dedupe_passages(items, time_window_sec=8.0, min_chars=40)
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
    out=[]
    for domain in allowed_domains:
        links=_ddg_domain_search(domain, query, headers, timeout)
        if not links: links=[f"https://{domain}"]
        links=links[:per_domain]
        for url in links:
            try:
                r=requests.get(url, headers=headers, timeout=timeout)
                if r.status_code!=200: continue
                soup=BeautifulSoup(r.text,"html.parser")
                paras=[p.get_text(" ",strip=True) for p in soup.find_all("p")]
                text=_normalize_text(" ".join(paras))[:2000]
                if len(text)<200: continue
                out.append({"domain":domain,"url":url,"text":text})
            except: continue
        if len(out)>=max_snippets: break
    return out[:max_snippets]

# ---------------- Precompute runners + repairs ----------------
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

        import math, collections
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

# ---------------- Chunks-derived creator index ----------------
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
    st.header("Answer settings")
    initial_k = st.number_input(
        "How many passages to scan first", 32, 5000, 768, 32,
        help="First-pass recall. The app embeds your question, pulls this many candidate passages from ALL indexed chunks inside the routed videos, then refines. Higher = better coverage but slower."
    )
    final_k = st.number_input(
        "How many passages to use", 8, 80, 36, 2,
        help="Second-pass selection. The top K passages after diversification are fed to the writer. Higher = more evidence but risk of rambling."
    )
    max_videos = st.number_input(
        "Maximum videos to use", 1, 12, 5, 1,
        help="Upper bound on distinct videos that can contribute evidence. Keeps answers focused and readable."
    )
    per_video_cap = st.number_input(
        "Passages per video", 1, 10, 4, 1,
        help="Limit per source so one video cannot dominate the answer. Typical sweet spot: 3–4."
    )

    st.subheader("Balance variety and accuracy")
    use_mmr = st.checkbox(
        "Encourage variety (recommended)", value=True,
        help="Applies Maximal Marginal Relevance. Removes near-duplicate quotes so different angles are covered."
    )
    mmr_lambda = st.slider(
        "Balance: accuracy vs variety", 0.1, 0.9, 0.45, 0.05,
        help="0.1 = more diverse but looser matches. 0.9 = very tight matches but less breadth. 0.4–0.5 works well."
    )

    st.subheader("Prefer newer videos")
    recency_weight = st.slider(
        "Recency influence", 0.0, 1.0, 0.20, 0.05,
        help="How much newer videos are favored when ranking sources. 0 = ignore publish date. 1 = strongly prefer recent."
    )
    half_life = st.slider(
        "Recency half-life (days)", 7, 720, 270, 7,
        help="Time-decay for recency. Each half-life reduces recency weight by 50%. Smaller = prefer very recent uploads."
    )

    # Experts with counts derived from chunks + vm
    vm = load_video_meta()
    vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
    counts={canon: len(creator_to_vids.get(canon,set())) for canon in ALLOWED_CREATORS}

    st.subheader("Experts")
    st.caption("All selected by default. Uncheck to exclude. Counts show how many videos per expert are in your index.")
    selected_creators_list=[]
    for i, canon in enumerate(ALLOWED_CREATORS):
        label=f"{canon} ({counts.get(canon,0)})"
        if st.checkbox(label, value=True, key=f"exp_{i}"):
            selected_creators_list.append(canon)
    selected_creators:set[str]=set(selected_creators_list)
    # persist selection for follow-ups
    st.session_state["selected_creators"]=selected_creators

    # Trusted sites
    st.subheader("Trusted sites")
    st.caption("Short, vetted excerpts to support claims from expert videos. They act as secondary evidence, not the primary source.")
    allow_web = st.checkbox(
        "Add supporting excerpts from trusted sites",
        value=True,
        help=("When enabled, the app fetches a few short paragraphs from the checked medical sites and blends them with the video evidence. "
              "If no relevant pages are found, the answer still works using videos alone."),
        disabled=(requests is None or BeautifulSoup is None)
    )
    selected_domains=[]
    for i,dom in enumerate(TRUSTED_DOMAINS):
        if st.checkbox(dom, value=True, key=f"site_{i}"):
            selected_domains.append(dom)
    max_web = st.slider(
        "Max supporting excerpts", 0, 8, 3, 1,
        help="Ceiling for how many website snippets may be included. 2–3 keeps answers tight while adding references."
    )

    model_choice = st.selectbox(
        "OpenAI model", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0,
        help="Model used to write the final answer from the selected evidence. Mini = cheaper/faster; 4o/4.1-mini = stronger reasoning."
    )

    st.divider()
    show_diag = st.toggle(
        "Show data diagnostics", value=False,
        help="Reveal file locations, modification times, and index health so you can verify precompute freshness without a shell."
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

# Chat history
if "messages" not in st.session_state: st.session_state.messages=[]
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
    st.caption("Use these actions if freshness checks fail or summaries are missing.")
    rep = _path_exists_report()
    st.code(json.dumps(rep, indent=2))

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

    # Creator inventory from chunks.jsonl
    with st.expander("Creator inventory (from chunks.jsonl)"):
        inv = sorted(((c, len(vs)) for c,vs in build_creator_indexes_from_chunks(load_video_meta())[1].items()),
                     key=lambda x: -x[1])
        st.dataframe([{"creator": c, "videos": n} for c,n in inv], use_container_width=True)

    st.markdown("---")
    st.caption("Keyword coverage scan (verification only). Checks whether specific terms appear in your transcripts and which experts mention them. Does not affect answers.")
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

# Prompt
prompt = st.chat_input("Ask about ApoB/LDL drugs, sleep, protein timing, fasting, supplements, protocols…")
if prompt is None:
    cols=st.columns([1]*12)
    with cols[-1]:
        st.button("Clear chat", key="clear_idle", help="Start a new conversation.", on_click=_clear_chat)
    st.stop()

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

# Use chunk-derived mapping for allow-list
vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
universe = set(vid_list or list(vm.keys()) or list(vid_to_creator.keys()))
chosen = st.session_state.get("selected_creators", set(ALLOWED_CREATORS))
allowed_vids = {vid for vid in universe if vid_to_creator.get(vid) in chosen}

# Stage A: route to top 5 videos
qv = embedder.encode([prompt], normalize_embeddings=True).astype("float32")[0]
topK = 5
routed_vids = route_videos_by_summary(
    prompt, qv, summaries, vm, C, list(universe), allowed_vids,
    topK=topK, recency_weight=float(recency_weight), half_life_days=float(half_life)
)
candidate_vids = set(routed_vids) if routed_vids else allowed_vids

# Stage B: search chunks only in routed videos
with st.spinner("Scanning selected videos…"):
    try:
        hits = stageB_search_chunks(
            prompt, index, embedder, candidate_vids,
            initial_k=min(int(initial_k), index.ntotal if index is not None else int(initial_k)),
            final_k=int(final_k), max_videos=int(max_videos), per_video_cap=int(per_video_cap),
            apply_mmr=bool(use_mmr), mmr_lambda=float(mmr_lambda),
            recency_weight=float(recency_weight), half_life_days=float(half_life), vm=vm
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Search failed."); st.exception(e); st.stop()

# Optional web support
web_snips=[]
if allow_web and selected_domains and requests and BeautifulSoup and int(max_web)>0:
    with st.spinner("Fetching trusted websites…"):
        web_snips = fetch_trusted_snippets(prompt, selected_domains, max_snippets=int(max_web))

# Build grouped evidence and answer
grouped_block = build_grouped_evidence_for_prompt(hits, vm, summaries, max_quotes=3)

with st.chat_message("assistant"):
    if not hits and not web_snips:
        st.warning("No relevant evidence found.")
        st.session_state.messages.append({"role":"assistant","content":"I couldn’t find enough evidence to answer that."})
        cols=st.columns([1]*12)
        with cols[-1]:
            st.button("Clear chat", key="clear_nohits", on_click=_clear_chat)
        st.stop()

    with st.spinner("Writing your answer…"):
        ans=openai_answer(model_choice, prompt, st.session_state.messages, grouped_block, web_snips, no_video=(len(hits)==0))

    st.markdown(ans)
    st.session_state.messages.append({"role":"assistant","content":ans})

    # Sources UI + JSON export
    with st.expander("Sources & timestamps", expanded=False):
        st.caption("Each bullet shows the timestamped quote pulled from a video. Web items are listed separately as supporting references.")
        groups = group_hits_by_video(hits)
        ordered = sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)

        export = {"videos": [], "web": []}

        for vid, items in ordered:
            info = vm.get(vid, {})
            title = info.get("title") or summaries.get(vid, {}).get("title") or vid
            creator_raw = _raw_creator_of_vid(vid, vm)
            creator = canonicalize_creator(creator_raw) or creator_raw
            url = info.get("url") or ""
            header = f"**{title}**" + (f" — _{creator}_" if creator else "")
            st.markdown(f"- [{header}]({url})" if url else f"- {header}")

            clean = _dedupe_passages(items, time_window_sec=8.0, min_chars=40)

            v = {"video_id": vid, "title": title, "creator": creator, "url": url, "quotes": []}
            for h in clean:
                ts = _format_ts((h.get("meta") or {}).get("start", 0))
                q = _normalize_text(h.get("text", ""))
                if len(q) > 160: q = q[:160] + "…"
                st.markdown(f"  • **{ts}** — “{q}”")
                v["quotes"].append({"ts": ts, "text": q})
            export["videos"].append(v)

        if web_snips:
            st.markdown("**Trusted websites**")
            for j, s in enumerate(web_snips, 1):
                st.markdown(f"W{j}. [{s['domain']}]({s['url']})")
                export["web"].append({"id": f"W{j}", "domain": s["domain"], "url": s["url"]})

        st.download_button(
            "Download sources as JSON",
            data=json.dumps(export, ensure_ascii=False, indent=2),
            file_name="sources.json",
            mime="application/json",
        )

# Footer + Clear chat
st.caption("Routing uses extractive bullets + centroids to pick the best videos, then searches their chunks.")
cols=st.columns([1]*12)
with cols[-1]:
    st.button("Clear chat", key="clear_done", on_click=_clear_chat)
