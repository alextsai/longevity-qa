# app/app.py
# -*- coding: utf-8 -*-
"""
Longevity / Nutrition Q&A — experts-first RAG + trusted-web augmentation.

Fixes in this version
1) Recover missing video_id from metas.pkl, so routed quotes appear.
2) Resolve DuckDuckGo "/l/?uddg=" redirects and scheme-less URLs; fetch real pages.
3) Auto-renormalize video centroids to ||v||≈1.0 on load; warn if dim mismatch.
4) Quote relevance filter: sentence boundary + 40+ chars + query keyword overlap.
5) Per-turn sources are stored and rendered; follow-ups keep prior turns.
6) Self-check block shows how many quotes/web snippets were used and routing trace.
7) Robust creator canonicalization (no brittle synonym lists).
8) Minimal, safe timeouts and hard guards so empty sources are visible, not silent.

Environment:
- DATA_DIR must contain: data/chunks/chunks.jsonl, data/index/faiss.index, data/index/metas.pkl,
  data/catalog/video_meta.json, and optional data/index/video_centroids.npy + video_ids.txt + data/catalog/video_summaries.json
- OPENAI_API_KEY must be set.
"""

from __future__ import annotations
import os, sys, json, pickle, time, re, math, collections
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote, urljoin

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Optional web libs
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests = None
    BeautifulSoup = None

# ---------------------------- Config / Paths ----------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT       = Path(os.getenv("DATA_DIR", "/var/data")).resolve()
CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"
OFFSETS_NPY     = DATA_ROOT / "data/chunks/chunks.offsets.npy"
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"

VID_CENT_NPY    = DATA_ROOT / "data/index/video_centroids.npy"
VID_IDS_TXT     = DATA_ROOT / "data/index/video_ids.txt"
VID_SUM_JSON    = DATA_ROOT / "data/catalog/video_summaries.json"

REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]
WEB_FALLBACK = os.getenv("WEB_FALLBACK", "true").strip().lower() in {"1", "true", "yes", "on"}

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

# ---------------------------- Utilities ----------------------------
def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _parse_ts(v) -> float:
    if isinstance(v, (int, float)):
        try: return float(v)
        except: return 0.0
    try:
        sec = 0.0
        for p in str(v).split(":"):
            sec = sec*60 + float(p)
        return sec
    except:
        return 0.0

def _iso_to_epoch(iso: str) -> float:
    if not iso: return 0.0
    try:
        if "T" in iso: return datetime.fromisoformat(iso.replace("Z","+00:00")).timestamp()
        return datetime.fromisoformat(iso).timestamp()
    except: return 0.0

def _format_ts(sec: float) -> str:
    sec = int(max(0, float(sec))); h,r = divmod(sec, 3600); m,s = divmod(r,60)
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

# ---------------------------- Admin gate ----------------------------
def _is_admin() -> bool:
    try: qp = st.query_params
    except Exception: return False
    if qp.get("admin","0") != "1": return False
    try: expected = st.secrets["ADMIN_KEY"]
    except Exception: expected = None
    if expected is None: return True
    return qp.get("key","") == str(expected)

# ---------------------------- Creator canonicalization ----------------------------
def _raw_creator_of_vid(vid: str, vm: dict) -> str:
    info = vm.get(vid, {}) or {}
    for k in ("podcaster","channel","author","uploader","owner","creator"):
        if info.get(k): return str(info[k])
    for k,v in ((kk.lower(), vv) for kk,vv in info.items()):
        if k in {"podcaster","channel","author","uploader","owner","creator"} and v:
            return str(v)
    return "Unknown"

def _canonicalize_creator(name: str) -> str | None:
    n = _normalize_text(name).lower().replace("™","").replace("®","")
    if not n: return None
    if n in EXCLUDED_CREATORS_EXACT: return None
    toks = set(re.findall(r"[a-z0-9]+", n))
    if "healthy" in toks and "immune" in toks: return "Healthy Immune Doc"
    if "diary" in toks and "ceo" in toks:    return "The Diary of A CEO"
    if "huberman" in toks:                   return "Andrew Huberman"
    if "attia" in toks:                      return "Peter Attia MD"
    if "jamnadas" in toks:                   return "Dr. Pradip Jamnadas, MD"
    for canon in ALLOWED_CREATORS:
        if n == canon.lower(): return canon
        if re.sub(r"[^\w\s]","", n) == re.sub(r"[^\w\s]","", canon.lower()): return canon
    return None

# ---------------------------- Offsets for random access ----------------------------
def _ensure_offsets() -> np.ndarray:
    if OFFSETS_NPY.exists():
        try:
            arr = np.load(OFFSETS_NPY)
            saved = len(arr); cur = sum(1 for _ in CHUNKS_PATH.open("rb"))
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

# ---------------------------- Loaders ----------------------------
@st.cache_data(show_spinner=False)
def load_video_meta() -> Dict[str, Dict[str,Any]]:
    if VIDEO_META_JSON.exists():
        try: return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

@st.cache_resource(show_spinner=False)
def _load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")

@st.cache_resource(show_spinner=False)
def load_metas_and_model(index_path: Path=INDEX_PATH, metas_path: Path=METAS_PKL):
    if not index_path.exists() or not metas_path.exists(): return None, None, None
    index = faiss.read_index(str(index_path))
    with metas_path.open("rb") as f:
        payload = pickle.load(f)
    metas_from_pkl = payload.get("metas", [])  # list aligned to FAISS ids
    model_name = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")
    local_dir = DATA_ROOT/"models"/"all-MiniLM-L6-v2"
    try_name = str(local_dir) if (local_dir/"config.json").exists() else model_name
    embedder = _load_embedder(try_name)
    if index.d != embedder.get_sentence_embedding_dimension():
        raise RuntimeError(f"Embedding dim mismatch: FAISS={index.d} vs Encoder={embedder.get_sentence_embedding_dimension()}. Rebuild.")
    return index, metas_from_pkl, {"model_name": try_name, "embedder": embedder}

@st.cache_resource(show_spinner=False)
def load_video_centroids():
    if not (VID_CENT_NPY.exists() and VID_IDS_TXT.exists()): return None, None
    C = np.load(VID_CENT_NPY).astype("float32")
    # auto-renorm to ||v||≈1
    if C.ndim == 2 and C.size:
        n = np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        C = C / n
        try: np.save(VID_CENT_NPY, C)
        except: pass
    vids = VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
    if C.shape[0] != len(vids): return None, None
    return C, vids

@st.cache_data(show_spinner=False)
def load_video_summaries():
    if VID_SUM_JSON.exists():
        try: return json.loads(VID_SUM_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

# ---------------------------- Relevance helpers ----------------------------
def _quote_is_valid(text: str, query: str) -> bool:
    t = _normalize_text(text)
    if len(t) < 40: return False
    if not re.search(r"[\.!?:;]\s", t): return False
    qtok = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
    ttok = set(re.findall(r"[a-z0-9]+", t.lower()))
    overlap = len(qtok & ttok)
    return overlap >= 1  # at least one query keyword present

def _build_idf_over_bullets(summaries: dict) -> dict:
    DF = collections.Counter()
    for v in list(summaries.keys()):
        for b in summaries.get(v, {}).get("bullets", []):
            for w in set(re.findall(r"[a-z0-9]+", (b.get("text","") or "").lower())):
                DF[w] += 1
    N = max(1, len(summaries))
    return {w: math.log((N+1)/(df+0.5)) for w,df in DF.items()}

def _kw_score(text: str, query: str, idf: dict) -> Tuple[float,int]:
    if not text: return 0.0, 0
    qtok = [w for w in re.findall(r"[a-z0-9]+", (query or "").lower()) if w]
    t = re.findall(r"[a-z0-9]+", (text or "").lower())
    tf = {w: t.count(w) for w in set(t)}
    overlap = len(set(qtok) & set(t))
    score = sum(tf.get(w,0) * idf.get(w,0.0) for w in set(qtok)) / (len(t)+1e-6)
    return score, overlap

def _vid_epoch(vm: dict, vid: str) -> float:
    info = (vm or {}).get(vid, {})
    return _iso_to_epoch(info.get("published_at") or info.get("publishedAt") or info.get("date") or "")

def _recency_score(published_ts: float, now: float, half_life_days: float) -> float:
    if published_ts <= 0: return 0.0
    days = max(0.0, (now - published_ts) / 86400.0)
    return 0.5 ** (days / max(1e-6, half_life_days))

def mmr(qv: np.ndarray, doc_vecs: np.ndarray, k: int, lambda_diversity: float=0.45) -> List[int]:
    if doc_vecs.size == 0: return []
    sim = (doc_vecs @ qv.reshape(-1,1)).ravel()
    sel=[]; cand=set(range(doc_vecs.shape[0]))
    while cand and len(sel)<k:
        if not sel:
            cl=list(cand); pick = cl[int(np.argmax(sim[cl]))]
            sel.append(pick); cand.remove(pick); continue
        sv=doc_vecs[sel]; cl=list(cand)
        max_div=(sv @ doc_vecs[cl].T).max(axis=0)
        scores=lambda_diversity*sim[cl] - (1-lambda_diversity)*max_div
        pick=cl[int(np.argmax(scores))]
        sel.append(pick); cand.remove(pick)
    return sel

# ---------------------------- Routing ----------------------------
def route_videos_by_summary(
    query: str, qv: np.ndarray,
    summaries: dict, vm: dict,
    C: np.ndarray | None, vids: list[str] | None,
    allowed_vids: set[str],
    topK: int, recency_weight: float, half_life_days: float,
    min_kw_overlap: int = 2
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
        kw, overlap = _kw_score(pseudo, query, idf)
        if len(bullets)>0 and overlap < min_kw_overlap:
            continue
        cs = cent.get(v, 0.0)
        rec = _recency_score(_vid_epoch(vm, v), now, half_life_days)
        base = 0.6*cs + 0.3*kw
        score = (1.0 - recency_weight)*base + recency_weight*(0.1*rec + 0.9*base)
        scored.append((v, score))
    scored.sort(key=lambda x:-x[1])
    return [v for v,_ in scored[:int(topK)]]

# ---------------------------- Stage-B search with metas recovery ----------------------------
def stageB_search_chunks(
    query: str,
    index: faiss.Index, embedder: SentenceTransformer,
    candidate_vids: Set[str] | None,
    metas_aligned: List[dict],
    initial_k: int, final_k: int, max_videos: int, per_video_cap: int,
    apply_mmr: bool, mmr_lambda: float,
    recency_weight: float, half_life_days: float, vm: dict
) -> List[Dict[str,Any]]:
    if index is None: return []
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
    K  = min(int(initial_k), index.ntotal if index.ntotal>0 else int(initial_k))
    D,I = index.search(qv.reshape(1,-1), K)
    idxs    = [int(x) for x in I[0] if x>=0]
    scores0 = [float(s) for s in D[0][:len(idxs)]]

    rows = list(iter_jsonl_rows(idxs))
    texts=[]; metas=[]; keep=[]
    for i_global, j in rows:
        t = _normalize_text(j.get("text",""))
        m = (j.get("meta") or {}).copy()
        vid = (m.get("video_id") or m.get("vid") or m.get("ytid") or
               j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
        # recover from metas.pkl if missing
        if not vid and metas_aligned and 0 <= i_global < len(metas_aligned):
            vid = metas_aligned[i_global].get("video_id") or metas_aligned[i_global].get("vid")
        if vid: m["video_id"] = vid
        if "start" not in m and "start_sec" in m: m["start"] = m.get("start_sec")
        m["start"] = _parse_ts(m.get("start", 0))
        if t:
            texts.append(t); metas.append(m)
            keep.append((candidate_vids is None) or (vid in candidate_vids))

    if any(keep):
        texts=[t for t,k in zip(texts,keep) if k]
        metas=[m for m,k in zip(metas,keep) if k]
        idxs=[i for i,k in zip(idxs,keep) if k]
        scores0=[s for s,k in zip(scores0,keep) if k]
    if not texts: return []

    doc_vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")
    order = list(range(len(texts)))
    if apply_mmr:
        order = mmr(qv, doc_vecs, k=min(len(texts), max(8, final_k*2)), lambda_diversity=float(mmr_lambda))

    now=time.time(); blended=[]
    for li in order:
        if li >= len(texts): continue
        base = scores0[li] if li<len(scores0) else 0.0
        m    = metas[li]; t = texts[li]; vid = m.get("video_id")
        if not vid: continue
        if not _quote_is_valid(t, query): continue
        rec=_recency_score(_vid_epoch(vm, vid), now, half_life_days)
        score=(1.0-recency_weight)*float(base)+recency_weight*float(rec)
        blended.append({"i": idxs[li], "score": float(score), "text": t, "meta": m})

    # per-video caps
    picked=[]; seen_per={}; seen_vids=[]
    for h in sorted(blended, key=lambda x:-x["score"]):
        vid=h["meta"]["video_id"]
        if vid not in seen_vids and len(seen_vids)>=int(max_videos): continue
        if seen_per.get(vid,0)>=int(per_video_cap): continue
        if vid not in seen_vids: seen_vids.append(vid)
        seen_per[vid]=seen_per.get(vid,0)+1
        picked.append(h)
        if len(picked)>=int(final_k): break
    return picked

# ---------------------------- Group and export evidence ----------------------------
def group_hits_by_video(hits: List[Dict[str,Any]]) -> Dict[str, List[Dict[str,Any]]]:
    g={}
    for h in hits:
        vid=(h.get("meta") or {}).get("video_id") or "Unknown"
        g.setdefault(vid, []).append(h)
    return g

def build_grouped_evidence_for_prompt(hits: List[Dict[str,Any]], vm: dict, summaries: dict, query: str, max_quotes: int=3) -> Tuple[str, Dict[str,Any]]:
    groups = group_hits_by_video(hits)
    ordered = sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)
    lines=[]; export=[]
    for v_idx,(vid,items) in enumerate(ordered,1):
        info=vm.get(vid,{})
        title=info.get("title") or summaries.get(vid,{}).get("title") or vid
        creator_raw=_raw_creator_of_vid(vid, vm)
        creator=_canonicalize_creator(creator_raw) or creator_raw
        date=info.get("published_at") or info.get("publishedAt") or info.get("date") or summaries.get(vid,{}).get("published_at") or ""
        lines.append(f"[Video {v_idx}] {title} — {creator}" + (f" — {date}" if date else ""))

        # prefer summary bullets if they fit query
        added=0
        for b in summaries.get(vid,{}).get("bullets", []):
            q=_normalize_text(b.get("text",""))
            if _quote_is_valid(q, query):
                ts=_format_ts(b.get("ts",0))
                lines.append(f"  • {ts}: “{q[:260]}”")
                export.append({"video_id":vid,"title":title,"creator":creator,"ts":ts,"text":q})
                added += 1
                if added>=max_quotes: break

        # fill from items to ensure at least one quote
        if added < 1:
            clean=[h for h in items if _quote_is_valid(h.get("text",""), query)]
            clean.sort(key=lambda x:-x["score"])
            for h in clean[:max(1, max_quotes - added)]:
                ts=_format_ts((h.get("meta") or {}).get("start",0))
                q=_normalize_text(h.get("text",""))
                lines.append(f"  • {ts}: “{q[:260]}”")
                export.append({"video_id":vid,"title":title,"creator":creator,"ts":ts,"text":q})
                added += 1
        lines.append("")
    return "\n".join(lines).strip(), {"videos": export}

# ---------------------------- Web search + snippet fetch ----------------------------
def _resolve_ddg_href(href: str) -> str:
    if not href: return ""
    if href.startswith("http://") or href.startswith("https://"): return href
    try:
        if href.startswith("/l/?"):
            qs = parse_qs(urlparse(href).query)
            if "uddg" in qs and qs["uddg"]:
                return unquote(qs["uddg"][0])
        if href.startswith("//"): return "https:" + href
        return urljoin("https://duckduckgo.com", href)
    except Exception:
        return href

def _ddg_html(domain: str, query: str, headers: dict, timeout: float) -> List[str]:
    try:
        r=requests.get("https://duckduckgo.com/html/", params={"q":f"site:{domain} {query}"}, headers=headers, timeout=timeout)
        if r.status_code!=200: return []
        soup=BeautifulSoup(r.text,"html.parser")
        raw=[a.get("href") for a in soup.select("a.result__a") if a.get("href")]
        return [_resolve_ddg_href(h) for h in raw]
    except Exception:
        return []

def _ddg_lite(domain: str, query: str, headers: dict, timeout: float) -> List[str]:
    try:
        r=requests.get("https://duckduckgo.com/lite/", params={"q":f"site:{domain} {query}"}, headers=headers, timeout=timeout)
        if r.status_code!=200: return []
        soup=BeautifulSoup(r.text,"html.parser")
        raw=[a.get("href","") for a in soup.find_all("a")]
        links=[_resolve_ddg_href(h) for h in raw]
        return [u for u in links if u.startswith("http")]
    except Exception:
        return []

def fetch_trusted_snippets(query: str, allowed_domains: List[str], max_snippets: int=3, per_domain: int=1, timeout: float=7.0):
    trace=[]; out=[]
    if not requests or not BeautifulSoup or max_snippets<=0:
        st.session_state["web_trace"]="requests/bs4 missing."
        return []
    headers={"User-Agent":"Mozilla/5.0"}
    for domain in allowed_domains:
        links=_ddg_html(domain, query, headers, timeout)
        if not links:
            links=_ddg_lite(domain, query, headers, timeout); trace.append(f"{domain}: lite links={len(links)}")
        else:
            trace.append(f"{domain}: html links={len(links)}")
        if not links:
            links=[f"https://{domain}"]; trace.append(f"{domain}: fallback homepage")
        links=links[:per_domain]
        for url in links:
            try:
                r=requests.get(url, headers=headers, timeout=timeout)
                if r.status_code!=200: trace.append(f"{domain}: {url} status {r.status_code}"); continue
                soup=BeautifulSoup(r.text,"html.parser")
                title = _normalize_text((soup.title.string if soup.title else "") or "")[:140]
                paras=[p.get_text(" ",strip=True) for p in soup.find_all("p")]
                text=_normalize_text(" ".join(paras))[:1800]
                if len(text)<200: trace.append(f"{domain}: short {url}"); continue
                out.append({"domain":domain,"url":url,"title":title,"text":text})
            except Exception as e:
                trace.append(f"{domain}: fetch error {e}")
        if len(out)>=max_snippets: break
    st.session_state["web_trace"]="; ".join(trace) if trace else "no trace"
    return out[:max_snippets]

# ---------------------------- LLM answerer ----------------------------
def openai_answer(model_name: str, question: str, history, grouped_video_block: str,
                  web_snips: list[dict], no_video: bool) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return "⚠️ OPENAI_API_KEY is not set."
    recent = [m for m in history[-6:] if m.get("role") in ("user","assistant")]
    convo = [("User: " if m["role"]=="user" else "Assistant: ")+m.get("content","") for m in recent]
    web_lines = [f"(W{j}) {s.get('domain','web')} — {s.get('title','')[:80]} — {s.get('url','')}\n“{_normalize_text(s.get('text',''))[:300]}”" for j,s in enumerate(web_snips,1)]
    web_block = "\n".join(web_lines) if web_lines else "None"
    fallback = ("If no suitable video evidence exists, you MAY answer from trusted web snippets alone, begin with 'Web-only evidence'.\n"
               ) if (WEB_FALLBACK and no_video) else "Trusted web snippets are supporting evidence.\n"
    system = (
        "Answer ONLY from provided evidence (video quotes) plus trusted web snippets.\n"
        + fallback +
        "Cite each claim: use (Video k) for videos, (DOMAIN Wj) for web.\n"
        "Prefer human clinical data; flag mechanistic/animal data. Normalize units and include effect sizes if present.\n"
        "Output sections: Key summary • Practical protocol • Safety notes. Be concise and source-grounded."
    )
    payload = (("Recent conversation:\n"+"\n".join(convo)+"\n\n") if convo else "") + \
              f"Question: {question}\n\nGrouped Video Evidence:\n{grouped_video_block or 'None'}\n\nTrusted Web Snippets:\n{web_block}\n\nWrite a concise, well-grounded answer."
    client = OpenAI(timeout=60)
    r = client.chat.completions.create(
        model=model_name, temperature=0.2,
        messages=[{"role":"system","content":system},{"role":"user","content":payload}]
    )
    return (r.choices[0].message.content or "").strip()

# ---------------------------- Creator inventory (chunks-first) ----------------------------
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

# ---------------------------- Admin scan helper ----------------------------
def scan_chunks_for_terms(terms: List[str], vm: Dict[str,Dict[str,Any]], limit_examples: int=200):
    if not CHUNKS_PATH.exists(): return {"total_matches":0,"per_creator":{},"examples":[]}
    pat = re.compile(r"("+"|".join([re.escape(t) for t in terms]) + r")", re.IGNORECASE)
    total=0; per_creator={}; examples=[]
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for ln in f:
            try:j=json.loads(ln)
            except:continue
            t=j.get("text","") or ""
            if not pat.search(t): continue
            m=(j.get("meta") or {})
            vid=(m.get("video_id") or m.get("vid") or m.get("ytid") or j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id") or "Unknown")
            st_sec=_parse_ts(m.get("start", m.get("start_sec",0)))
            creator=_canonicalize_creator(_raw_creator_of_vid(vid, vm)) or _raw_creator_of_vid(vid, vm)
            per_creator[creator]=per_creator.get(creator,0)+1
            total+=1
            if len(examples)<int(limit_examples):
                sn=_normalize_text(t); sn=sn[:260]+"…" if len(sn)>260 else sn
                examples.append({"video_id":vid,"creator":creator,"ts":_format_ts(st_sec),"snippet":sn})
    per_creator=dict(sorted(per_creator.items(), key=lambda kv:-kv[1]))
    return {"total_matches":total,"per_creator":per_creator,"examples":examples}

# ---------------------------- Streamlit UI ----------------------------
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

if "turns" not in st.session_state: st.session_state["turns"]=[]
if "messages" not in st.session_state: st.session_state["messages"]=[]

with st.sidebar:
    st.markdown("**Auto Mode** · accuracy + diversity")
    vm = load_video_meta()
    vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
    counts={canon: len(creator_to_vids.get(canon,set())) for canon in ALLOWED_CREATORS}

    st.subheader("Experts")
    st.caption("All selected. Uncheck to exclude.")
    selected_creators=[]
    for i,canon in enumerate(ALLOWED_CREATORS):
        if st.checkbox(f"{canon} ({counts.get(canon,0)})", value=True, key=f"exp_{i}"):
            selected_creators.append(canon)
    selected_creators=set(selected_creators)
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
        st.caption("Defaults are tuned; change only if needed.")
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
    st.checkbox("offsets built", value=OFFSETS_NPY.exists(), disabled=True)
    st.checkbox("faiss.index present", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("metas.pkl present", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("video_meta.json present", value=VIDEO_META_JSON.exists(), disabled=True)
    cent_ready = VID_CENT_NPY.exists() and VID_IDS_TXT.exists() and VID_SUM_JSON.exists()
    st.caption("Video centroids/summaries: ready" if cent_ready else "Precompute not found. Use admin tools.")

# Diagnostics
if show_diag:
    colA,colB,colC = st.columns([2,3,3])
    with colA: st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    with colB: st.caption(f"chunks mtime: {_iso(_file_mtime(CHUNKS_PATH)) if CHUNKS_PATH.exists() else 'missing'}")
    with colC: st.caption(f"index mtime: {_iso(_file_mtime(INDEX_PATH)) if INDEX_PATH.exists() else 'missing'}")

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# Admin panel
def _path_exists_report()->Dict[str, Any]:
    return {
        "DATA_DIR": str(DATA_ROOT),
        "chunks": str(CHUNKS_PATH), "chunks_exists": CHUNKS_PATH.exists(),
        "centroids": str(VID_CENT_NPY), "centroids_exists": VID_CENT_NPY.exists(),
        "ids": str(VID_IDS_TXT), "ids_exists": VID_IDS_TXT.exists(),
        "summaries": str(VID_SUM_JSON), "summaries_exists": VID_SUM_JSON.exists(),
    }

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
            if spec is None or spec.loader is None: return f"precompute error: cannot load {p}"
            mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            mod.main(); return "ok: rebuilt via file loader"
    except Exception as e:
        return f"precompute error: {e}"

def _repair_centroids_in_place()->str:
    try:
        if not VID_CENT_NPY.exists(): return "centroids file missing"
        C = np.load(VID_CENT_NPY).astype("float32")
        if C.ndim != 2 or C.size == 0: return "centroids shape invalid"
        n = np.linalg.norm(C, axis=1, keepdims=True) + 1e-12
        C = C / n; np.save(VID_CENT_NPY, C); return "ok: renormalized"
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
                try: j=json.loads(ln)
                except: continue
                t=_normalize_text(j.get("text","")); m=(j.get("meta") or {})
                vid=(m.get("video_id") or m.get("vid") or m.get("ytid") or
                     j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
                if not vid or not t: continue
                ts=_parse_ts(m.get("start", m.get("start_sec", 0)))
                texts_by_vid.setdefault(vid, []).append(t)
                ts_by_vid.setdefault(vid, []).append(float(ts))
        vids=list(texts_by_vid.keys())
        if not vids: return "no videos detected in chunks.jsonl"
        DF=collections.Counter()
        for vid in vids:
            seen=set()
            for t in texts_by_vid[vid]:
                for w in set(re.findall(r"[a-z0-9]+",t.lower())):
                    if w not in seen: DF[w]+=1; seen.add(w)
        N=max(1,len(vids))
        def score_line(t:str)->float:
            words=re.findall(r"[a-z0-9]+",t.lower()); tf=collections.Counter(words)
            return sum(tf[w]*math.log((N+1)/(DF.get(w,1)+0.5)) for w in tf)/(len(words)+1e-6)
        summaries={}
        for vid in vids:
            lines=texts_by_vid[vid][:max_lines_per_video]; times=ts_by_vid[vid][:max_lines_per_video]
            idx_scores=[(i,score_line(t)) for i,t in enumerate(lines)]
            top=sorted([i for i,_ in sorted(idx_scores,key=lambda x:-x[1])[:12]])[:10]
            bullets=[{"ts":float(times[i]),"text":(lines[i][:280]+"…" if len(lines[i])>280 else lines[i])} for i in top[:6]]
            summary=" ".join(lines[i] for i in top[:6]); summary=(summary[:1200]+"…") if len(summary)>1200 else summary
            info=vm.get(vid,{})
            summaries[vid]={"title":info.get("title",""),"channel":info.get("channel",""),
                            "published_at":info.get("published_at") or info.get("publishedAt") or info.get("date") or "",
                            "bullets":bullets,"summary":summary}
        VID_SUM_JSON.parent.mkdir(parents=True, exist_ok=True)
        VID_SUM_JSON.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"ok: wrote {VID_SUM_JSON}"
    except Exception as e:
        return f"summaries fallback error: {e}"

if _is_admin():
    st.subheader("Diagnostics (admin)")
    try:
        _,_,payload = load_metas_and_model(); emb_name = payload["model_name"]
        status = {"msg":[]}
    except Exception as e:
        emb_name="unknown"; status={"msg":[f"load error: {e}"]}

    col1,col2,col3 = st.columns(3)
    with col1:
        st.caption(f"Embedder: {emb_name}")
        st.checkbox("centroids.npy present", value=VID_CENT_NPY.exists(), disabled=True)
        st.checkbox("video_ids.txt present", value=VID_IDS_TXT.exists(), disabled=True)
        st.checkbox("video_summaries.json present", value=VID_SUM_JSON.exists(), disabled=True)
    with col2:
        st.caption(f"chunks mtime:  {_iso(_file_mtime(CHUNKS_PATH))}")
        st.caption(f"centroids mtime: {_iso(_file_mtime(VID_CENT_NPY))}")
        st.caption(f"ids mtime:       {_iso(_file_mtime(VID_IDS_TXT))}")
    with col3:
        st.code(json.dumps(_path_exists_report(), indent=2))

    cols_dbg = st.columns(3)
    with cols_dbg[0]:
        if st.button("Rebuild precompute (admin)"):
            with st.spinner("Building centroids and summaries…"): msg=_run_precompute_inline()
            st.success(str(msg)); st.cache_resource.clear(); st.cache_data.clear(); st.rerun()
    with cols_dbg[1]:
        if st.button("Repair centroid norms"):
            with st.spinner("Renormalizing…"): msg=_repair_centroids_in_place()
            st.success(msg); st.cache_resource.clear(); st.cache_data.clear(); st.rerun()
    with cols_dbg[2]:
        if st.button("Build summaries now (fallback)"):
            with st.spinner("Generating summaries from chunks.jsonl…"): msg=_build_summaries_fallback()
            st.success(msg); st.cache_resource.clear(); st.cache_data.clear(); st.rerun()

    with st.expander("Creator inventory (from chunks.jsonl)"):
        inv = sorted(((c, len(vs)) for c,vs in build_creator_indexes_from_chunks(load_video_meta())[1].items()),
                     key=lambda x: -x[1])
        st.dataframe([{"creator": c, "videos": n} for c,n in inv], use_container_width=True)

# ---------------------------- Prompt loop ----------------------------
prompt = st.chat_input("Ask about ApoB/LDL drugs, sleep, fasting, supplements, protocols…")
if prompt is None:
    if st.session_state["turns"]:
        st.subheader("Previous replies and sources")
        for i,t in enumerate(st.session_state["turns"],1):
            with st.expander(f"Turn {i}: {t.get('prompt','')[:80]}"):
                st.markdown(t.get("answer",""))
                if t.get("videos"):
                    st.markdown("**Video quotes**")
                    for v in t["videos"]:
                        url = (load_video_meta().get(v.get("video_id",""),{}) or {}).get("url","")
                        header = f"- **{v.get('title','')}** — _{v.get('creator','')}_ • {v.get('ts','')}"
                        st.markdown(f"{header}")
                        st.markdown(f"  • “{_normalize_text(v.get('text',''))[:160]}”")
                if t.get("web"):
                    st.markdown("**Trusted websites**")
                    for j,s in enumerate(t["web"],1):
                        st.markdown(f"W{j}. [{s.get('title') or s['domain']}]({s['url']})")
                if t.get("web_trace"):
                    st.caption(f"web trace: {t['web_trace']}")
    cols=st.columns([1]*12)
    with cols[-1]:
        st.button("Clear chat", key="clear_idle", on_click=_clear_chat)
    st.stop()

st.session_state.messages.append({"role":"user","content":prompt})
with st.chat_message("user"): st.markdown(prompt)

# Guard files
missing=[p for p in REQUIRED if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error("Missing required files:\n" + "\n".join(f"- {p}" for p in missing))
    st.stop()

# Load index/model/meta/summaries
try:
    index, metas_aligned, payload = load_metas_and_model()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load index or encoder."); st.exception(e); st.stop()
embedder: SentenceTransformer = payload["embedder"]
vm = load_video_meta()
C, vid_list = load_video_centroids()
summaries = load_video_summaries()

# Allowed videos by expert selection
vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
universe = set(vid_list or list(vm.keys()) or list(vid_to_creator.keys()))
allowed_vids = {vid for vid in universe if vid_to_creator.get(vid) in st.session_state.get("selected_creators", set(ALLOWED_CREATORS))}
if not allowed_vids: allowed_vids = universe

# Routing with light follow-up context
routing_query = prompt
prev_users = [m["content"] for m in st.session_state.messages if m["role"]=="user"]
if len(prev_users) >= 2:
    routing_query = prev_users[-2] + " ; " + prompt
qv = embedder.encode([routing_query], normalize_embeddings=True).astype("float32")[0]
routed_vids = route_videos_by_summary(
    routing_query, qv, summaries, vm, C, list(universe), allowed_vids,
    topK=5, recency_weight=0.20, half_life_days=270, min_kw_overlap=2
)
candidate_vids = set(routed_vids) if routed_vids else allowed_vids

# Recall knobs (overridable)
K_scan       = st.session_state.get("adv_scanK", 768)
K_use        = st.session_state.get("adv_useK", 36)
max_videos   = st.session_state.get("adv_maxvid", 5)
per_video_cap= st.session_state.get("adv_cap", 4)
use_mmr      = st.session_state.get("adv_mmr", True)
mmr_lambda   = st.session_state.get("adv_lam", 0.45)
rec_w        = st.session_state.get("adv_rec", 0.20)
half_life    = st.session_state.get("adv_hl", 270)

# Stage-B search
with st.spinner("Scanning selected videos…"):
    try:
        hits = stageB_search_chunks(
            prompt, index, embedder, candidate_vids, metas_aligned,
            initial_k=min(int(K_scan), index.ntotal if index is not None else int(K_scan)),
            final_k=int(K_use), max_videos=int(max_videos), per_video_cap=int(per_video_cap),
            apply_mmr=bool(use_mmr), mmr_lambda=float(mmr_lambda),
            recency_weight=float(rec_w), half_life_days=float(half_life), vm=vm
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Search failed."); st.exception(e); st.stop()

# Web support
web_snips=[]
if allow_web and selected_domains and requests and BeautifulSoup and int(max_web_auto)>0:
    with st.spinner("Fetching trusted websites…"):
        web_snips = fetch_trusted_snippets(prompt, selected_domains, max_snippets=int(max_web_auto), per_domain=1)

# Build evidence + answer
grouped_block, export_struct = build_grouped_evidence_for_prompt(hits, vm, summaries, prompt, max_quotes=3)

with st.chat_message("assistant"):
    if not export_struct["videos"] and not web_snips:
        st.warning("Web-only evidence: expert video quotes were not available after filtering.")
    with st.spinner("Writing your answer…"):
        ans = openai_answer(model_choice, prompt, st.session_state.messages, grouped_block, web_snips, no_video=(len(export_struct["videos"])==0))
    st.markdown(ans)
    st.session_state.messages.append({"role":"assistant","content":ans})

    # Persist sources for THIS reply
    st.session_state["turns"].append({
        "prompt": prompt,
        "answer": ans,
        "videos": export_struct["videos"],
        "web": web_snips,
        "web_trace": st.session_state.get("web_trace","")
    })

    # Sources UI
    with st.expander("Sources for this reply", expanded=True):
        vids_by = group_hits_by_video([{"meta":{"video_id":v["video_id"]},"score":0,"text":v["text"]} for v in export_struct["videos"]])
        ordered_vids = sorted(vids_by.keys(), key=lambda vid: -len(vids_by[vid]))
        if export_struct["videos"]:
            for vid in ordered_vids:
                info = vm.get(vid, {})
                title = info.get("title") or "Video"
                creator = _canonicalize_creator(_raw_creator_of_vid(vid, vm)) or _raw_creator_of_vid(vid, vm)
                url = info.get("url") or ""
                st.markdown(f"- **{title}** — _{creator}_" + (f" — [{url}]({url})" if url else ""))
                for v in [x for x in export_struct["videos"] if x["video_id"]==vid]:
                    st.markdown(f"  • **{v['ts']}** — “{_normalize_text(v['text'])[:160]}”")
        else:
            st.markdown("_No video quotes used in this turn._")

        if web_snips:
            st.markdown("**Trusted websites**")
            for j,s in enumerate(web_snips,1):
                label = s.get("title") or s["domain"]
                st.markdown(f"W{j}. [{label}]({s['url']})")

        st.caption(
            f"Self-check: videos routed={len(candidate_vids)} • videos quoted={len(ordered_vids)} • "
            f"quotes used={len(export_struct['videos'])} • experts covered={len(set(v.get('creator') for v in export_struct['videos']))} • "
            f"web selected={len(web_snips)} • web trace: {st.session_state.get('web_trace','')}"
        )

# Footer
cols=st.columns([1]*12)
with cols[-1]:
    st.button("Clear chat", key="clear_done", on_click=_clear_chat)
