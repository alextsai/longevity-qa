# app/app.py
# -*- coding: utf-8 -*-
"""
Longevity / Nutrition Q&A — Experts-first RAG with trusted web support

This version:
- Exhaustive search by default: skip routing and search ALL chunks for ALL selected experts.
- initial_k = index.ntotal (full-index scan) for maximum recall.
- Five-expert allow-list in UI; composite bad names removed; synonym normalization.
- Admin via st.query_params ?admin=1 (+ optional ADMIN_KEY in secrets).
- Inline Precompute button (admin) to build centroids/ids/summaries.
- Diagnostics: shapes/norms/dim checks; mtimes; keyword scan; JSON source export.
- Trusted-sites fetch with DDG + simple fallback crawl; clear “no web excerpts” notice.
- Strong system prompt: mechanisms, effect sizes, therapy options, explicit citations.
"""

from __future__ import annotations

# --- Quiet env defaults to avoid noisy logs ---
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS","1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")

# --- Imports ---
from pathlib import Path
import sys, json, pickle, time, re, math
from typing import List, Dict, Any, Set
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
    requests=None; BeautifulSoup=None

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(os.getenv("DATA_DIR","/var/data")).resolve()

CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"
OFFSETS_NPY     = DATA_ROOT / "data/chunks/chunks.offsets.npy"
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"

# Precompute outputs (still used for summaries UI and diagnostics)
VID_CENT_NPY    = DATA_ROOT / "data/index/video_centroids.npy"
VID_IDS_TXT     = DATA_ROOT / "data/index/video_ids.txt"
VID_SUM_JSON    = DATA_ROOT / "data/catalog/video_summaries.json"

REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]

# --- Flags ---
WEB_FALLBACK = os.getenv("WEB_FALLBACK","true").strip().lower() in {"1","true","yes","on"}

# --- Trusted domains (default set; user can uncheck) ---
TRUSTED_DOMAINS = [
    "nih.gov","medlineplus.gov","cdc.gov","mayoclinic.org","health.harvard.edu",
    "familydoctor.org","healthfinder.gov","ama-assn.org","medicalxpress.com",
    "sciencedaily.com","nejm.org","med.stanford.edu","icahn.mssm.edu"
]

# --- Experts configuration: allow-list + exact excludes + synonyms ---
ALLOWED_CREATORS = [
    "Dr. Pradip Jamnadas, MD",
    "Andrew Huberman",
    "Healthy Immune Doc",
    "Peter Attia MD",
    "The Diary of A CEO",
]
EXCLUDED_CREATORS_EXACT = {
    # composite bad entries to drop entirely (exact, case-insensitive)
    "they diary of a ceo and louse tomlinson",
    "dr. pradip jamnadas, md and the primal podcast",
}
CREATOR_SYNONYMS = {
    "heathy immune doc": "Healthy Immune Doc",
    "healthy immune doc": "Healthy Immune Doc",
    "the diary of a ceo": "The Diary of A CEO",
}

def canonicalize_creator(name: str) -> str | None:
    """Return canonical allowed creator, or None if excluded/not allowed."""
    n = (name or "").strip()
    if not n: return None
    low = n.lower()
    if low in EXCLUDED_CREATORS_EXACT:  # drop composite names
        return None
    low = CREATOR_SYNONYMS.get(low, low)
    for canon in ALLOWED_CREATORS:
        if low == canon.lower():
            return canon
    return None

# --- Small utils ---
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

def _clear_chat():
    st.session_state["messages"]=[]
    st.rerun()

# --- Admin gate via query params (?admin=1&key=...) ---
def _is_admin()->bool:
    try:
        qp = st.query_params
    except Exception:
        return False
    if qp.get("admin","0")!="1":
        return False
    try:
        expected = st.secrets["ADMIN_KEY"]
    except Exception:
        expected = None
    if expected is None:
        return True
    return qp.get("key","")==str(expected)

# --- Diagnostics + metadata loaders ---
@st.cache_data(show_spinner=False, hash_funcs={Path:_file_mtime})
def load_video_meta(vm_path:Path=VIDEO_META_JSON)->Dict[str,Dict[str,Any]]:
    if vm_path.exists():
        try:return json.loads(vm_path.read_text(encoding="utf-8"))
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
            try:
                emb_dim = _load_embedder(embedder_model).get_sentence_embedding_dimension()
                status["ok_dim"] = (C.shape[1]==emb_dim)
            except Exception:
                status["ok_dim"] = False
            if status["chunks_mtime"] > max(status["cent_mtime"], status["ids_mtime"]):
                status["msg"].append("chunks.jsonl is newer than centroids → run precompute.")
        else:
            status["msg"].append("Centroids/IDs missing → run precompute.")
    except Exception as e:
        status["msg"].append(f"precompute check error: {e}")
    return status

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
            info=vm.get(vid,{})
            creator=info.get("podcaster") or info.get("channel") or "Unknown"
            canon = canonicalize_creator(creator)
            label = canon if canon else creator
            per_creator[label]=per_creator.get(label,0)+1
            total+=1
            if len(examples)<int(limit_examples):
                sn=_normalize_text(t); sn=sn[:260]+"…" if len(sn)>260 else sn
                examples.append({"video_id":vid,"creator":label,"ts":_format_ts(st_sec),"snippet":sn})
    per_creator=dict(sorted(per_creator.items(), key=lambda kv:-kv[1]))
    return {"total_matches":total,"per_creator":per_creator,"examples":examples}

# --- JSONL offsets for fast random access ---
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

# --- Model + FAISS ---
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
        raise RuntimeError(f"Embedding dim mismatch: FAISS={index.d} vs Encoder={embedder.get_sentence_embedding_dimension()}. Rebuild with model '{model_name}'.")
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

# --- Diversity (MMR) ---
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

# --- Stage B (single stage now; Stage-A routing is bypassed) ---
def stageB_search_chunks(query:str,
    index:faiss.Index, embedder:SentenceTransformer,
    candidate_vids:Set[str],
    initial_k:int, final_k:int, max_videos:int, per_video_cap:int,
    apply_mmr:bool, mmr_lambda:float,
    recency_weight:float, half_life_days:float, vm:dict)->List[Dict[str,Any]]:

    if index is None: return []
    qv=embedder.encode([query], normalize_embeddings=True).astype("float32")[0]

    # Full-index scan for maximum recall
    K = index.ntotal if index.ntotal>0 else int(initial_k)
    D,I=index.search(qv.reshape(1,-1), K)
    idxs=[int(x) for x in I[0] if x>=0]
    scores0=[float(s) for s in D[0][:len(idxs)]]

    rows=list(iter_jsonl_rows(idxs))
    texts=[]; metas=[]; keep=[]
    for _,j in rows:
        t=_normalize_text(j.get("text",""))
        if not t: keep.append(False); continue
        m=(j.get("meta") or {}).copy()
        vid=(m.get("video_id") or m.get("vid") or m.get("ytid") or
             j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
        if vid: m["video_id"]=vid
        if "start" not in m and "start_sec" in m: m["start"]=m.get("start_sec")
        m["start"]=_parse_ts(m.get("start",0))
        texts.append(t); metas.append(m)
        keep.append((not candidate_vids) or (vid in candidate_vids))

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

# --- Grouping for prompt ---
def group_hits_by_video(hits:List[Dict[str,Any]])->Dict[str,List[Dict[str,Any]]]:
    g={}
    for h in hits:
        vid=(h.get("meta") or {}).get("video_id") or "Unknown"
        g.setdefault(vid,[]).append(h)
    return g

def build_grouped_evidence_for_prompt(hits:List[Dict[str,Any]], vm:dict, summaries:dict, max_quotes:int=3)->str:
    groups=group_hits_by_video(hits)
    ordered=sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)
    lines=[]
    for v_idx,(vid,items) in enumerate(ordered,1):
        info=vm.get(vid,{})
        title=info.get("title") or summaries.get(vid,{}).get("title") or vid
        creator_raw=info.get("podcaster") or info.get("channel") or "Unknown"
        creator=canonicalize_creator(creator_raw) or creator_raw
        date=info.get("published_at") or info.get("publishedAt") or info.get("date") or summaries.get(vid,{}).get("published_at") or ""
        lines.append(f"[Video {v_idx}] {title} — {creator}" + (f" — {date}" if date else ""))
        summ=summaries.get(vid,{}).get("summary","")
        if summ: lines.append(f"  • summary: {summ[:300]}{'…' if len(summ)>300 else ''}")
        clean=_dedupe_passages(items, time_window_sec=8.0, min_chars=40)
        for h in clean[:max_quotes]:
            ts=_format_ts((h.get("meta") or {}).get("start",0))
            q=_normalize_text(h.get("text","")); q=q[:260]+"…" if len(q)>260 else q
            lines.append(f"  • {ts}: “{q}”")
        lines.append("")
    return "\n".join(lines).strip()

# --- Web fetch (DDG + simple fallback) ---
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
        if not links: links=[f"https://{domain}"]  # simple homepage crawl fallback
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

# --- LLM call with strong system prompt ---
def openai_answer(model_name:str, question:str, history:List[Dict[str,str]], grouped_video_block:str, web_snips:List[Dict[str,str]], no_video:bool)->str:
    if not os.getenv("OPENAI_API_KEY"): return "⚠️ OPENAI_API_KEY is not set."

    recent=history[-6:]; convo=[]
    for m in recent:
        role=m.get("role"); content=m.get("content","")
        if role in ("user","assistant") and content:
            label="User" if role=="user" else "Assistant"
            convo.append(f"{label}: {content}")

    web_lines=[f"(W{j}) {s.get('domain','web')} — {s.get('url','')}\n“{' '.join((s.get('text','')).split())[:300]}”" for j,s in enumerate(web_snips,1)]
    web_block="\n".join(web_lines) if web_lines else "None"

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

    user_payload=((("Recent conversation:\n"+"\n".join(convo)+"\n\n") if convo else "")
        + f"Question: {question}\n\n"
        + "Grouped Video Evidence:\n" + (grouped_video_block or "None") + "\n\n"
        + "Trusted Web Snippets:\n" + web_block + "\n\n"
        + "Write a concise, well-grounded answer.")

    try:
        client=OpenAI(timeout=60)
        r=client.chat.completions.create(
            model=model_name, temperature=0.2,
            messages=[{"role":"system","content":system},{"role":"user","content":user_payload}]
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ Generation error: {e}"

# --- Inline precompute (admin) keeps summaries fresh; not used for routing now ---
def _run_precompute_inline()->str:
    if not CHUNKS_PATH.exists(): return "missing chunks.jsonl"
    try:
        _, _, payload = load_metas_and_model()
    except Exception as e:
        return f"encoder/index load error: {e}"
    enc: SentenceTransformer = payload["embedder"]
    vm = load_video_meta()

    from collections import defaultdict, Counter
    texts_by_vid = defaultdict(list)
    metas_by_vid = defaultdict(list)

    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for ln in f:
            j=json.loads(ln)
            t=_normalize_text(j.get("text",""))
            m=(j.get("meta") or {})
            vid = m.get("video_id") or m.get("vid") or m.get("ytid") or j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id")
            if not vid or not t: continue
            st_sec=_parse_ts(m.get("start", m.get("start_sec",0)))
            texts_by_vid[vid].append(t)
            metas_by_vid[vid].append({"start":st_sec})

    vids=sorted(texts_by_vid.keys())
    if not vids: return "no videos in chunks"

    # Centroids still built for diagnostics (not used for routing in this exhaustive build)
    centroids=[]
    for vid in vids:
        X=enc.encode(texts_by_vid[vid], normalize_embeddings=True, batch_size=128).astype("float32")
        c=X.mean(axis=0); c = c / (np.linalg.norm(c)+1e-12)
        centroids.append(c)
    C=np.stack(centroids).astype("float32")
    VID_CENT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(VID_CENT_NPY, C)
    VID_IDS_TXT.write_text("\n".join(vids), encoding="utf-8")

    # Summaries/claims for better prompts + sources UI
    DF = Counter()
    for vid in vids:
        seen=set()
        for t in texts_by_vid[vid]:
            toks=set(w.lower() for w in t.split())
            for w in toks:
                if w in seen: continue
                DF[w]+=1; seen.add(w)
    N=len(vids)
    def score_text(t):
        from collections import Counter as Ctr
        words=[w.lower() for w in t.split()]
        tf=Ctr(words); val=0.0
        for w,cnt in tf.items():
            df=DF.get(w,1); idf=math.log((N+1)/(df+0.5))
            val += cnt*idf
        return val/(len(words)+1e-6)

    summaries={}
    for vid in vids:
        info=vm.get(vid,{})
        lines=texts_by_vid[vid]
        sc=[(i,score_text(t)) for i,t in enumerate(lines)]
        top=[i for i,_ in sorted(sc, key=lambda x:-x[1])[:12]]
        top_sorted=sorted(top[:10])
        claims=[]
        for i in top_sorted[:6]:
            txt=lines[i]
            claims.append({"ts": float(metas_by_vid[vid][i]["start"]),
                           "text": (txt[:280]+"…") if len(txt)>280 else txt})
        summary=" ".join(lines[i] for i in top_sorted[:6])
        summary = summary[:1200]+"…" if len(summary)>1200 else summary
        summaries[vid]={
            "title": info.get("title",""),
            "channel": info.get("channel") or info.get("podcaster",""),
            "published_at": info.get("published_at") or info.get("publishedAt") or info.get("date",""),
            "summary": summary,
            "claims": claims,
        }
    VID_SUM_JSON.parent.mkdir(parents=True, exist_ok=True)
    VID_SUM_JSON.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"ok: {len(vids)} videos"

# === App UI ===
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

with st.sidebar:
    st.header("Answer settings")
    st.caption("Exhaustive mode active: scans every chunk from the experts you keep checked.")
    initial_k = st.number_input("How many passages to scan first", 32, 5000, 1024, 32,
        help="Ignored here. The app scans all chunks for maximum coverage.")
    final_k = st.number_input("How many passages to use", 8, 80, 36, 2,
        help="How many top passages inform the answer.")
    max_videos = st.number_input("Maximum videos to use", 1, 12, 6, 1,
        help="Breadth without rambling.")
    per_video_cap = st.number_input("Passages per video", 1, 10, 4, 1,
        help="Prevents one video from dominating.")

    st.subheader("Balance variety and accuracy")
    use_mmr = st.checkbox("Encourage variety (recommended)", value=True,
        help="Reduces near-duplicate quotes for broader coverage.")
    mmr_lambda = st.slider("Balance: accuracy vs variety", 0.1, 0.9, 0.45, 0.05,
        help="Right = closer match. Left = more diverse.")

    st.subheader("Prefer newer videos")
    recency_weight = st.slider("Recency influence", 0.0, 1.0, 0.20, 0.05,
        help="Small preference for newer material.")
    half_life = st.slider("Recency half-life (days)", 7, 720, 270, 7,
        help="After this many days, recency value halves.")

    # Experts vertical list (strict allow-list)
    vm = load_video_meta()
    def _raw_creator_of(vid:str)->str:
        info=vm.get(vid,{})
        return info.get("podcaster") or info.get("channel") or "Unknown"

    counts={canon:0 for canon in ALLOWED_CREATORS}
    for vid in vm:
        canon = canonicalize_creator(_raw_creator_of(vid))
        if canon is None: continue
        counts[canon]=counts.get(canon,0)+1

    st.subheader("Experts")
    st.caption("All selected by default. Uncheck to exclude from the answer.")
    selected_creators_list=[]
    for i, canon in enumerate(ALLOWED_CREATORS):
        label=f"{canon} ({counts.get(canon,0)})"
        if st.checkbox(label, value=True, key=f"exp_{i}"):
            selected_creators_list.append(canon)
    selected_creators:set[str]=set(selected_creators_list)

    # Trusted sites
    st.subheader("Trusted sites")
    st.caption("Short, reliable excerpts that support expert videos.")
    allow_web = st.checkbox(
        "Add supporting excerpts from trusted sites",
        value=True,
        help=("Turns on short quotes from trusted sites."
              if (requests and BeautifulSoup) else
              "Install 'requests' and 'beautifulsoup4' to enable."),
        disabled=(requests is None or BeautifulSoup is None)
    )
    selected_domains=[]
    for i,dom in enumerate(TRUSTED_DOMAINS):
        if st.checkbox(dom, value=True, key=f"site_{i}"):
            selected_domains.append(dom)
    max_web = st.slider("Max supporting excerpts", 0, 8, 3, 1,
        help="Upper limit on short quotes pulled from trusted sites.")

    model_choice = st.selectbox("OpenAI model", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0,
        help="Model used to compose the final answer.")

    # Diagnostics toggle + file presence
    st.divider()
    show_diag = st.toggle("Show data diagnostics", value=False, help="Show file locations and last updated times.")
    st.subheader("Library status")
    st.checkbox("chunks.jsonl present", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("offsets built", value=OFFSETS_NPY.exists(), disabled=True)
    st.checkbox("faiss.index present", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("metas.pkl present", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("video_meta.json present", value=VIDEO_META_JSON.exists(), disabled=True)
    cent_ready = VID_CENT_NPY.exists() and VID_IDS_TXT.exists()
    st.caption("Video centroids: ready" if cent_ready else "Video centroids: not found (precompute optional).")

if show_diag:
    colA,colB,colC = st.columns([2,3,3])
    with colA: st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    with colB: st.caption(f"chunks.jsonl mtime: {datetime.fromtimestamp(_file_mtime(CHUNKS_PATH)).isoformat() if CHUNKS_PATH.exists() else 'missing'}")
    with colC: st.caption(f"index mtime: {datetime.fromtimestamp(_file_mtime(INDEX_PATH)).isoformat() if INDEX_PATH.exists() else 'missing'}")

# Show chat history
if "messages" not in st.session_state: st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Admin diagnostics panel
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
        st.caption(f"chunks mtime: {datetime.fromtimestamp(status.get('chunks_mtime',0)).isoformat() if status.get('chunks_mtime') else 'n/a'}")
        st.caption(f"centroids mtime: {datetime.fromtimestamp(status.get('cent_mtime',0)).isoformat() if status.get('cent_mtime') else 'n/a'}")
        st.caption(f"ids mtime: {datetime.fromtimestamp(status.get('ids_mtime',0)).isoformat() if status.get('ids_mtime') else 'n/a'}")
    for msg in status.get("msg",[]): st.warning(msg)

    if st.button("Rebuild precompute (admin)"):
        with st.spinner("Building centroids and summaries…"):
            msg=_run_precompute_inline()
        st.success(str(msg))
        st.cache_resource.clear(); st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.caption("Keyword coverage scan (verification only).")
    default_terms = "apob, apo-b, ldl, statin, ezetimibe, pcsk9, bempedoic, inclisiran, niacin"
    term_input = st.text_input("Terms (comma-separated)", default_terms)
    if st.button("Run scan"):
        terms=[t.strip() for t in term_input.split(",") if t.strip()]
        with st.spinner("Scanning transcripts…"):
            scan=scan_chunks_for_terms(terms, load_video_meta(), limit_examples=300)
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

# Load FAISS + encoder
try:
    index, _, payload = load_metas_and_model()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load index or encoder."); st.exception(e)
    st.stop()
embedder: SentenceTransformer = payload["embedder"]
vm = load_video_meta()
C, vid_list = load_video_centroids()  # not used for routing in this build; kept for diagnostics
summaries = load_video_summaries()

# Exhaustive candidate set: ALL videos from selected experts
def _creator_of_vid(vid:str)->str|None:
    raw = vm.get(vid,{}).get("podcaster") or vm.get(vid,{}).get("channel") or "Unknown"
    return canonicalize_creator(raw)

allowed_vids = {
    vid for vid in (vid_list or list(vm.keys()))
    if (_creator_of_vid(vid) in selected_creators)
}
candidate_vids = allowed_vids  # exhaustive
routed_vids = list(allowed_vids)

# Search all chunks
with st.spinner("Scanning every chunk from selected experts…"):
    try:
        hits=stageB_search_chunks(
            prompt, index, embedder, candidate_vids,
            initial_k=index.ntotal if index is not None else 1024,  # full-index scan
            final_k=int(final_k), max_videos=int(max_videos), per_video_cap=int(per_video_cap),
            apply_mmr=bool(use_mmr), mmr_lambda=float(mmr_lambda),
            recency_weight=float(recency_weight), half_life_days=float(half_life), vm=vm
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Search failed."); st.exception(e); st.stop()

# Optional web
web_snips=[]
if allow_web and selected_domains and requests and BeautifulSoup and int(max_web)>0:
    with st.spinner("Fetching trusted websites…"):
        web_snips = fetch_trusted_snippets(prompt, selected_domains, max_snippets=int(max_web))
    if not web_snips:
        st.info("Trusted sites enabled but no web excerpts were found or reachable.", icon="ℹ️")

# Build prompt evidence and answer
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

    # Sources block
    with st.expander("Sources & timestamps", expanded=False):
        groups = group_hits_by_video(hits)
        ordered = sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)

        export = {"videos": [], "web": []}

        for vid, items in ordered:
            info = vm.get(vid, {})
            title = info.get("title") or summaries.get(vid, {}).get("title") or vid
            creator_raw = info.get("podcaster") or info.get("channel") or ""
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
st.caption("Exhaustive mode scans every chunk. If you add new videos, you can rebuild summaries in admin (?admin=1).")
cols=st.columns([1]*12)
with cols[-1]:
    st.button("Clear chat", key="clear_done", on_click=_clear_chat)

# Soft check: centroid dim vs encoder
try:
    if VID_CENT_NPY.exists():
        Ctmp=np.load(VID_CENT_NPY)
        emb_dim=_load_embedder(load_metas_and_model()[2]["model_name"]).get_sentence_embedding_dimension()
        if Ctmp.ndim==2 and Ctmp.shape[1]!=emb_dim:
            st.warning("Video centroids dim != encoder dim. Re-run precompute with the same model.", icon="⚠️")
except Exception:
    pass
