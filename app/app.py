# app/app.py
# -*- coding: utf-8 -*-
"""
Longevity / Nutrition Q&A
- Two-stage retrieval over curated YouTube chunks (FAISS + SBERT)
- Experts-first: user-selectable channels/podcasters drive answers
- Web evidence is supporting-only (never standalone)
- MMR re-ranking + recency blending with half-life
- Streamlit chat UI with grouped sources, timestamps, and JSON export
- Caching tied to file mtimes so precompute runs once and refreshes only when inputs change
- Defensive checks and clear error surfaces
"""

from __future__ import annotations

# ------------------ Minimal, safe env ------------------
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS","1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")  # CPU only

# ------------------ Imports -------------------
from pathlib import Path
import sys, json, pickle, time, re
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Optional web deps. If missing, the web option is disabled in UI.
try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests=None; BeautifulSoup=None

# ------------------ Paths -------------------
# Project root = repo root (../ from this file)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# DATA_DIR can be overridden at runtime (e.g., DATA_DIR=/var/data)
DATA_ROOT = Path(os.getenv("DATA_DIR","/var/data")).resolve()

# Core artifacts produced by ingestion (index + metas + chunks + catalog)
CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"              # chunked transcripts
OFFSETS_NPY     = DATA_ROOT / "data/chunks/chunks.offsets.npy"        # random-access offsets
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"                # FAISS ANN index
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"                  # {'metas': [...], 'model': '...'}
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"          # {video_id: {title, url, channel, published_at}}

# One-time precompute outputs (run scripts/precompute_video_summaries.py)
VID_CENT_NPY    = DATA_ROOT / "data/index/video_centroids.npy"        # [V,d] normalized centroids
VID_IDS_TXT     = DATA_ROOT / "data/index/video_ids.txt"              # one video_id per line
VID_SUM_JSON    = DATA_ROOT / "data/catalog/video_summaries.json"     # {vid:{title,channel,published_at,summary,claims}}

REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]

# ------------------ Trusted domains (supporting evidence only) -------------------
TRUSTED_DOMAINS = [
    "nih.gov","medlineplus.gov","cdc.gov","mayoclinic.org","health.harvard.edu",
    "familydoctor.org","healthfinder.gov","ama-assn.org","medicalxpress.com",
    "sciencedaily.com","nejm.org","med.stanford.edu","icahn.mssm.edu"
]

# ------------------ Small utils -------------------
def _normalize_text(s:str)->str:
    """Collapse whitespace and trim."""
    return re.sub(r"\s+"," ",(s or "").strip())

def _parse_ts(v)->float:
    """Accept seconds or 'H:MM:SS' strings. Return seconds as float."""
    if isinstance(v,(int,float)):
        try: return float(v)
        except: return 0.0
    try:
        sec=0.0
        for p in str(v).split(":"): sec=sec*60+float(p)
        return sec
    except: return 0.0

def _iso_to_epoch(iso:str)->float:
    """Parse ISO8601 into epoch seconds. Return 0.0 if invalid/missing."""
    if not iso: return 0.0
    try:
        if "T" in iso: return datetime.fromisoformat(iso.replace("Z","+00:00")).timestamp()
        return datetime.fromisoformat(iso).timestamp()
    except: return 0.0

def _format_ts(sec:float)->str:
    """Render seconds as M:SS or H:MM:SS for display."""
    sec = int(max(0,float(sec))); h,r=divmod(sec,3600); m,s=divmod(r,60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

def _file_mtime(p:Path)->float:
    """Return file mtime or 0.0 if missing. Used to bust caches on updates."""
    try: return p.stat().st_mtime
    except: return 0.0

# ------------------ Video metadata -------------------
@st.cache_data(show_spinner=False, hash_funcs={Path:_file_mtime})
def load_video_meta(vm_path:Path=VIDEO_META_JSON)->Dict[str,Dict[str,Any]]:
    """
    Load light per-video metadata. Cache invalidates automatically when file changes.
    Expected fields: title, url, channel or podcaster, published_at (ISO).
    """
    if vm_path.exists():
        try: return json.loads(vm_path.read_text(encoding="utf-8"))
        except: return {}
    return {}

def _vid_epoch(vm:dict, vid:str)->float:
    """Resolve publish date fields to epoch seconds for recency scoring."""
    info = (vm or {}).get(vid,{})
    return _iso_to_epoch(info.get("published_at") or info.get("publishedAt") or info.get("date") or "")

def _recency_score(published_ts:float, now:float, half_life_days:float)->float:
    """Exponential half-life decay. Value halves every N days since publish."""
    if published_ts<=0: return 0.0
    days = max(0.0,(now - published_ts)/86400.0)
    return 0.5 ** (days / max(1e-6, half_life_days))

# ------------------ Offsets over JSONL (O(1) random access) -------------------
def _ensure_offsets()->np.ndarray:
    """
    Build line offsets for chunks.jsonl once for fast random access.
    If chunks.jsonl grows, offsets are rebuilt. Safe for large files.
    """
    if OFFSETS_NPY.exists():
        try:
            arr=np.load(OFFSETS_NPY)
            saved=len(arr)
            cur=sum(1 for _ in CHUNKS_PATH.open("rb"))
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
    """Yield rows by integer index using offsets without loading the entire file."""
    if not CHUNKS_PATH.exists(): return
    offsets=_ensure_offsets()
    want=[i for i in indices if 0<=i<len(offsets)]
    if limit is not None: want=want[:limit]
    with CHUNKS_PATH.open("rb") as f:
        for i in want:
            f.seek(int(offsets[i]))
            raw=f.readline()
            try: yield i, json.loads(raw)
            except: continue

# ------------------ Model + FAISS -------------------
@st.cache_resource(show_spinner=False)
def _load_embedder(model_name:str)->SentenceTransformer:
    """Load SBERT encoder on CPU. Split so we can validate dims vs FAISS."""
    return SentenceTransformer(model_name, device="cpu")

@st.cache_resource(show_spinner=False)
def load_metas_and_model(index_path:Path=INDEX_PATH, metas_path:Path=METAS_PKL):
    """
    Load FAISS index, metas, and encoder. Validate embedding dimensionality.
    Returns (index, metas_from_pkl, {'model_name', 'embedder'})
    """
    if not index_path.exists() or not metas_path.exists():
        return None, None, None

    index = faiss.read_index(str(index_path))
    with metas_path.open("rb") as f:
        payload=pickle.load(f)

    metas_from_pkl = payload.get("metas",[])
    model_name = payload.get("model","sentence-transformers/all-MiniLM-L6-v2")

    # Prefer local cached model if present
    local_dir = DATA_ROOT/"models"/"all-MiniLM-L6-v2"
    try_name = str(local_dir) if (local_dir/"config.json").exists() else model_name
    embedder = _load_embedder(try_name)

    # Dimensionality check prevents silent mismatches
    idx_dim = index.d
    emb_dim = embedder.get_sentence_embedding_dimension()
    if idx_dim != emb_dim:
        raise RuntimeError(
            f"Embedding dim mismatch: FAISS={idx_dim} vs Encoder={emb_dim}. "
            f"Rebuild index or use encoder '{model_name}' used at index time."
        )

    return index, metas_from_pkl, {"model_name":try_name, "embedder":embedder}

@st.cache_resource(show_spinner=False)
def load_video_centroids():
    """Load per-video centroids + IDs for Stage A routing."""
    if not (VID_CENT_NPY.exists() and VID_IDS_TXT.exists()):
        return None, None
    C = np.load(VID_CENT_NPY).astype("float32")  # [V,d] normalized
    vids = VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
    if C.shape[0] != len(vids):  # defensive
        return None, None
    return C, vids

@st.cache_data(show_spinner=False)
def load_video_summaries():
    """Load per-video compact summaries and claims produced by precompute."""
    if VID_SUM_JSON.exists():
        try: return json.loads(VID_SUM_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

# ------------------ MMR (diversity) -------------------
def mmr(qv:np.ndarray, doc_vecs:np.ndarray, k:int, lambda_diversity:float=0.4)->List[int]:
    """
    Maximal Marginal Relevance: balances relevance and novelty.
    Returns indices into doc_vecs of selected items.
    """
    if doc_vecs.size==0: return []
    sim = (doc_vecs @ qv.reshape(-1,1)).ravel()
    sel=[]; cand=set(range(doc_vecs.shape[0]))
    while cand and len(sel)<k:
        if not sel:
            cl=list(cand); pick=cl[int(np.argmax(sim[cl]))]
            sel.append(pick); cand.remove(pick); continue
        sv=doc_vecs[sel]; cl=list(cand)
        max_div=(sv @ doc_vecs[cl].T).max(axis=0)
        scores = lambda_diversity*sim[cl] - (1-lambda_diversity)*max_div
        pick=cl[int(np.argmax(scores))]
        sel.append(pick); cand.remove(pick)
    return sel

# ------------------ Two-stage retrieval -------------------
def stageA_route_videos(
    qv:np.ndarray, C:np.ndarray, vids:List[str], topN:int,
    allowed_vids:Set[str] | None, vm:dict,
    recency_weight:float, half_life_days:float,
    pin_boost:float=0.0, pinned:Set[str] | None=None
)->List[str]:
    """
    Stage A: route query to top-N videos by centroid cosine blended with recency.
    Optionally give small score boost to pinned videos.
    """
    sims = (C @ qv.reshape(-1,1)).ravel()
    now=time.time(); pinned = pinned or set()
    blend=[]
    for i,vid in enumerate(vids):
        if allowed_vids and vid not in allowed_vids:
            continue
        rec=_recency_score(_vid_epoch(vm, vid), now, half_life_days)
        score = (1.0-recency_weight)*float(sims[i]) + recency_weight*float(rec)
        if vid in pinned:
            score += float(pin_boost)
        blend.append((vid, score))
    blend.sort(key=lambda x:-x[1])
    return [v for v,_ in blend[:topN]]

def stageB_search_chunks(
    query:str,
    index:faiss.Index, embedder:SentenceTransformer,
    candidate_vids:Set[str],
    initial_k:int, final_k:int, max_videos:int, per_video_cap:int,
    apply_mmr:bool, mmr_lambda:float,
    recency_weight:float, half_life_days:float, vm:dict
)->List[Dict[str,Any]]:
    """
    Stage B: search within routed/allowed videos.
    - Runs ANN over chunks
    - Filters by allowed videos
    - Optional MMR for diversity
    - Blends ANN score with recency
    - Enforces per-video caps and total caps
    """
    if index is None:
        return []

    # Encode query
    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]

    # ANN search
    K = min(int(initial_k), index.ntotal if index.ntotal>0 else int(initial_k))
    if K<=0:
        return []
    D,I = index.search(qv.reshape(1,-1), K)
    idxs=[int(x) for x in I[0] if x>=0]
    scores0=[float(s) for s in D[0][:len(idxs)]]

    # Gather texts/metas and filter by candidate videos
    rows=list(iter_jsonl_rows(idxs))
    texts=[]; metas=[]; keep_mask=[]
    for _,j in rows:
        t=_normalize_text(j.get("text",""))
        if not t:
            keep_mask.append(False); continue
        m=(j.get("meta") or {}).copy()
        vid = (m.get("video_id") or m.get("vid") or m.get("ytid")
               or j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
        if vid: m["video_id"]=vid
        if "start" not in m and "start_sec" in m: m["start"]=m.get("start_sec")
        m["start"]=_parse_ts(m.get("start",0))
        texts.append(t); metas.append(m)
        keep_mask.append((not candidate_vids) or (vid in candidate_vids))

    # Apply mask
    if any(keep_mask):
        texts = [t for t,k in zip(texts,keep_mask) if k]
        metas = [m for m,k in zip(metas,keep_mask) if k]
        idxs  = [i for i,k in zip(idxs, keep_mask) if k]
        scores0=[s for s,k in zip(scores0,keep_mask) if k]
    if not texts: return []

    # Encode docs and apply MMR for intra-list diversity
    doc_vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")
    order=list(range(len(texts)))
    if apply_mmr:
        sel=mmr(qv, doc_vecs, k=min(len(texts), max(8, final_k*2)), lambda_diversity=float(mmr_lambda))
        order=sel

    # Blend base ANN score with recency
    now=time.time()
    blended=[]
    for li in order:
        i_global=idxs[li] if li<len(idxs) else None
        base=scores0[li] if li<len(scores0) else 0.0
        m=metas[li]; t=texts[li]; vid=m.get("video_id")
        rec=_recency_score(_vid_epoch(vm, vid), now, half_life_days)
        score=(1.0-recency_weight)*float(base)+recency_weight*float(rec)
        blended.append((i_global, score, t, m))
    blended.sort(key=lambda x:-x[1])

    # Enforce per-video quotas and total caps
    picked=[]; seen_per_video={}; distinct=[]
    for ig,sc,tx,me in blended:
        vid=me.get("video_id","Unknown")
        if vid not in distinct and len(distinct)>=int(max_videos): continue
        c=seen_per_video.get(vid,0)
        if c>=int(per_video_cap): continue
        if vid not in distinct: distinct.append(vid)
        seen_per_video[vid]=c+1
        picked.append({"i":ig,"score":float(sc),"text":tx,"meta":me})
        if len(picked)>=int(final_k): break
    return picked

# ------------------ Grouping & prompt building -------------------
def group_hits_by_video(hits:List[Dict[str,Any]])->Dict[str,List[Dict[str,Any]]]:
    """Group selected hits by video_id for display and prompting."""
    g={}
    for h in hits:
        vid=(h.get("meta") or {}).get("video_id") or "Unknown"
        g.setdefault(vid,[]).append(h)
    return g

def build_grouped_evidence_for_prompt(hits:List[Dict[str,Any]], vm:dict, summaries:dict, max_quotes:int=3)->str:
    """
    Produce a compact, readable block that the LLM can cite as (Video k).
    Includes title, channel/podcaster, date, optional summary line, and a few timed quotes.
    """
    groups=group_hits_by_video(hits)
    ordered=sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)
    lines=[]
    for v_idx,(vid,items) in enumerate(ordered,1):
        info = vm.get(vid,{})
        title = info.get("title") or summaries.get(vid,{}).get("title") or vid
        # Prefer 'podcaster' if present; fall back to 'channel'
        creator = info.get("podcaster") or info.get("channel") or summaries.get(vid,{}).get("channel") or "Unknown"
        date = info.get("published_at") or info.get("publishedAt") or info.get("date") or summaries.get(vid,{}).get("published_at") or ""
        head=f"[Video {v_idx}] {title} — {creator}" + (f" — {date}" if date else "")
        lines.append(head)
        # summary line if available
        summ = summaries.get(vid,{}).get("summary","")
        if summ:
            lines.append(f"  • summary: {summ[:300]}{'…' if len(summ)>300 else ''}")
        # quoted spans in time order
        for h in sorted(items, key=lambda r: float(r['meta'].get('start',0)))[:max_quotes]:
            ts=_format_ts(h["meta"].get("start",0))
            q=(h["text"] or "").strip().replace("\n"," ")
            if len(q)>260: q=q[:260]+"…"
            lines.append(f"  • {ts}: “{q}”")
        lines.append("")
    return "\n".join(lines).strip()

# ------------------ Web fetch (supporting-only) -------------------
def fetch_trusted_snippets(query:str, allowed_domains:List[str], max_snippets:int=3, per_domain:int=1, timeout:float=6.0):
    """
    Very light domain-scoped scraping for supporting evidence.
    Disabled if requests/bs4 are not available. Avoids JS-heavy pages.
    """
    if not requests or not BeautifulSoup or max_snippets<=0: return []
    headers={"User-Agent":"Mozilla/5.0"}
    out=[]
    for domain in allowed_domains:
        try:
            resp=requests.get("https://duckduckgo.com/html/", params={"q":f"site:{domain} {query}"}, headers=headers, timeout=timeout)
            if resp.status_code!=200: continue
            soup=BeautifulSoup(resp.text,"html.parser")
            links=[a.get("href") for a in soup.select("a.result__a") if a.get("href") and domain in a.get("href")]
            links=links[:per_domain]
            for url in links:
                try:
                    r2=requests.get(url,headers=headers,timeout=timeout)
                    if r2.status_code!=200: continue
                    s2=BeautifulSoup(r2.text,"html.parser")
                    paras=[p.get_text(" ",strip=True) for p in s2.find_all("p")]
                    text=_normalize_text(" ".join(paras))[:2000]
                    if len(text)<200: continue
                    out.append({"domain":domain,"url":url,"text":text})
                except: continue
            if len(out)>=max_snippets: break
        except: continue
    return out[:max_snippets]

# ------------------ LLM call -------------------
def openai_answer(model_name:str, question:str, history:List[Dict[str,str]], grouped_video_block:str, web_snips:List[Dict[str,str]])->str:
    """
    Grounded synthesis. Priority = expert videos; web = supporting-only.
    Requires OPENAI_API_KEY in env. Deterministic temperature.
    """
    api_key=os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "⚠️ OPENAI_API_KEY is not set."

    # Trim recent chat for limited context
    recent=history[-6:]
    convo=[]
    for m in recent:
        role=m.get("role"); content=m.get("content","")
        if role in ("user","assistant") and content:
            label="User" if role=="user" else "Assistant"
            convo.append(f"{label}: {content}")

    # Web snippets block (hardened trimming, robust fields)
    web_lines = []
    for j, s in enumerate(web_snips, 1):
        txt = " ".join((s.get("text","")).split())[:300]
        dom = s.get("domain","web")
        url = s.get("url","")
        web_lines.append(f"(W{j}) {dom} — {url}\n“{txt}”")
    web_block = "\n".join(web_lines) if web_lines else "None"

    # Strict, experts-first system prompt
    system = (
        "Answer ONLY from the provided evidence. Priority: (1) grouped VIDEO evidence from selected experts, "
        "(2) trusted WEB snippets as SUPPORTING evidence only.\n"
        "If video evidence is insufficient or conflicting, say so and list what is missing. Do NOT guess.\n"
        "Rules:\n"
        "• Cite each claim/step: (Video k) for videos, (DOMAIN Wj) for web.\n"
        "• Prefer human clinical data; label animal/in-vitro/mechanistic explicitly.\n"
        "• Normalize units; give concrete ranges only when sources provide them.\n"
        "• No diagnosis or individualized advice. You may describe therapies/drugs ONLY as discussed in the sources, with source-specific qualifiers.\n"
        "• Resolve conflicts by stating both findings and which has higher-quality evidence.\n"
        "Structure:\n"
        "• Key takeaways — specific, source-grounded, detailed\n"
        "• Practical protocol — numbered, stepwise, actionable; include doses/timing if present\n"
        "• Safety notes — contraindications, interactions, and when to consult a clinician\n"
        "Output must be concise, uncertainty labeled, and free of speculation."
    )

    # Compose user payload with grouped evidence and optional supporting web
    user_payload=((("Recent conversation:\n"+"\n".join(convo)+"\n\n") if convo else "")
        + f"Question: {question}\n\n"
        + "Grouped Video Evidence:\n" + (grouped_video_block or "None") + "\n\n"
        + "Trusted Web Snippets (supporting only):\n" + web_block + "\n\n"
        + "Write a concise, well-grounded answer.")

    try:
        client=OpenAI(timeout=60)
        r=client.chat.completions.create(
            model=model_name, temperature=0.2,  # keep stable
            messages=[{"role":"system","content":system},{"role":"user","content":user_payload}]
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ Generation error: {e}"

# ------------------ Streamlit UI -------------------
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

# Top status bar with quick status + clear chat
colA, colB, colC, colD = st.columns([2,2,2,2])
with colA:
    st.caption(f"DATA_DIR = `{DATA_ROOT}`")
with colB:
    st.caption(f"chunks.jsonl mtime: {datetime.fromtimestamp(_file_mtime(CHUNKS_PATH)).isoformat() if CHUNKS_PATH.exists() else 'missing'}")
with colC:
    st.caption(f"index mtime: {datetime.fromtimestamp(_file_mtime(INDEX_PATH)).isoformat() if INDEX_PATH.exists() else 'missing'}")
with colD:
    if st.button("Clear chat"):
        st.session_state.pop("messages", None)
        st.rerun()

with st.sidebar:
    st.header("How the answer is built")

    # First/second pass sizes
    initial_k = st.number_input("How many passages to scan first", 32, 5000, 256, 32,
        help="First pass ANN over FAISS.")
    final_k = st.number_input("How many passages to use", 8, 80, 24, 2,
        help="Second pass sampling of top passages.")

    st.subheader("Keep it focused")
    max_videos = st.number_input("Maximum videos to use", 1, 12, 4, 1,
        help="At most this many distinct videos contribute.")
    per_video_cap = st.number_input("Passages per video", 1, 10, 3, 1,
        help="Prevents one video from dominating.")

    st.subheader("Balance variety and accuracy")
    use_mmr = st.checkbox("Encourage variety (recommended)", value=True,
        help="MMR reduces redundancy.")
    mmr_lambda = st.slider("Balance: accuracy vs variety", 0.1, 0.9, 0.4, 0.05,
        help="Higher = favor relevance. Lower = favor diversity.")

    st.subheader("Prefer newer videos")
    recency_weight = st.slider("Recency influence", 0.0, 1.0, 0.30, 0.05,
        help="0 ignores date; 1 heavily favors recent.")
    half_life = st.slider("Recency half-life (days)", 7, 720, 180, 7,
        help="Value halves every N days.")

    st.subheader("Route to best videos first")
    topN_videos = st.number_input("Videos to consider before chunk search", 1, 50, 10, 1,
        help="Stage A: pick likely videos, then search inside them.")

    # ----- Expert source controls: include/exclude creators, pin videos -----
    vm = load_video_meta()
    creators = sorted({(info.get("podcaster") or info.get("channel") or "Unknown") for info in vm.values()})

    include_creators = st.multiselect(
        "Include channels / podcasters", options=creators, default=creators,
        help="Videos from these experts are eligible."
    )
    exclude_creators = st.multiselect(
        "Exclude channels / podcasters", options=creators, default=[],
        help="Videos from these experts are excluded."
    )

    # Candidate pool respecting include − exclude
    def creator_of(vid:str)->str:
        info = vm.get(vid,{})
        return info.get("podcaster") or info.get("channel") or "Unknown"

    vids_pool = [
        vid for vid in vm.keys()
        if (creator_of(vid) in include_creators) and (creator_of(vid) not in set(exclude_creators))
    ]
    vid_labels = [f"{vm.get(vid,{}).get('title','(no title)')}  [{vid}]" for vid in vids_pool]
    chosen_vid_labels = st.multiselect("Pin specific videos (optional)", options=vid_labels, default=[])
    lookup = {f"{vm.get(vid,{}).get('title','(no title)')}  [{vid}]": vid for vid in vids_pool}
    chosen_vids: Set[str] = {lookup[x] for x in chosen_vid_labels} if chosen_vid_labels else set()

    # ----- Trusted sites as supporting-only evidence -----
    allow_web = st.checkbox("Add supporting excerpts from trusted sites", value=True,
                            disabled=(requests is None or BeautifulSoup is None))
    allowed_domains = st.multiselect("Trusted sites", options=TRUSTED_DOMAINS, default=TRUSTED_DOMAINS)
    max_web = st.slider("Max supporting excerpts", 0, 8, 3, 1)

    # ----- Model choice -----
    model_choice = st.selectbox("OpenAI model", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)

    st.divider()
    st.subheader("Library status")
    st.checkbox("chunks.jsonl present", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("offsets built", value=OFFSETS_NPY.exists(), disabled=True)
    st.checkbox("faiss.index present", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("metas.pkl present", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("video_meta.json present", value=VIDEO_META_JSON.exists(), disabled=True)

    # Precompute freshness detector (centroids must be newer than chunks)
    cent_ready = VID_CENT_NPY.exists() and VID_IDS_TXT.exists()
    st.caption("Video centroids: ready" if cent_ready else "Video centroids: not found (run scripts/precompute_video_summaries.py)")
    if cent_ready:
        newer_chunks = _file_mtime(CHUNKS_PATH) > max(_file_mtime(VID_CENT_NPY), _file_mtime(VID_IDS_TXT))
        if newer_chunks:
            st.warning("chunks.jsonl changed after centroids were built. Re-run precompute to refresh routing.", icon="⚠️")

# ------------------ Chat history -------------------
if "messages" not in st.session_state: st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

# ------------------ Input -------------------
prompt = st.chat_input("Ask about sleep, protein timing, fasting, supplements, protocols…")
if prompt is None:
    st.stop()
st.session_state.messages.append({"role":"user","content":prompt})
with st.chat_message("user"): st.markdown(prompt)

# ------------------ Guardrails: required files -------------------
missing=[p for p in REQUIRED if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error("Missing required files:\n" + "\n".join(f"- {p}" for p in missing))
    st.stop()

# ------------------ Load index + encoder -------------------
try:
    index, metas_from_pkl, payload = load_metas_and_model()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load index or encoder. See details below.")
        st.exception(e)
    st.stop()

if index is None or payload is None:
    with st.chat_message("assistant"): st.error("Index or model not available.")
    st.stop()

embedder: SentenceTransformer = payload["embedder"]
vm = load_video_meta()
C, vid_list = load_video_centroids()
summaries = load_video_summaries()

# ------------------ Stage A: route to videos -------------------
routed_vids=[]
candidate_vids:set[str]=set()
with st.spinner("Routing to likely videos…"):
    qv = embedder.encode([prompt], normalize_embeddings=True).astype("float32")[0]

    # Allowed set = explicit pins OR all videos from included creators minus excluded creators
    def creator_of(vid:str)->str:
        info = vm.get(vid,{})
        return info.get("podcaster") or info.get("channel") or "Unknown"

    allowed_vids_all = [
        vid for vid in (vid_list or list(vm.keys()))
        if creator_of(vid) in include_creators and creator_of(vid) not in set(exclude_creators)
    ]
    allowed_vids = set(chosen_vids) if chosen_vids else set(allowed_vids_all)

    # Stage A uses centroids when available; otherwise fallback to allowed filtering only
    PIN_BOOST = 0.05  # 5% bump to ensure pinned items get a small preference
    if C is not None and vid_list is not None:
        routed_vids = stageA_route_videos(
            qv, C, vid_list, int(topN_videos),
            allowed_vids, vm, float(recency_weight), float(half_life),
            pin_boost=PIN_BOOST, pinned=chosen_vids
        )
        candidate_vids = set(routed_vids)
    else:
        candidate_vids = allowed_vids  # fallback: restrict search to allowed creators/videos

# ------------------ Stage B: search inside routed/allowed videos -------------------
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

# ------------------ Optional supporting web snippets -------------------
web_snips=[]
if allow_web and allowed_domains:
    with st.spinner("Fetching trusted websites…"):
        web_snips = fetch_trusted_snippets(prompt, allowed_domains, max_snippets=int(max_web))

# ------------------ Enforce 'web is supporting-only' policy -------------------
video_has_hits = bool(hits)
if not video_has_hits and web_snips:
    with st.chat_message("assistant"):
        st.warning("No relevant expert video evidence found. Trusted sites are supporting-only. Adjust creators or refine your query.")
    st.stop()

# ------------------ Build grouped evidence and answer -------------------
grouped_block = build_grouped_evidence_for_prompt(hits, vm, summaries, max_quotes=3)

with st.chat_message("assistant"):
    if not hits and not web_snips:
        st.warning("No relevant evidence found.")
        st.session_state.messages.append({"role":"assistant","content":"I couldn’t find enough evidence to answer that."})
        st.stop()

    with st.spinner("Writing your answer…"):
        ans = openai_answer(model_choice, prompt, st.session_state.messages, grouped_block, web_snips)

    st.markdown(ans)
    st.session_state.messages.append({"role":"assistant","content":ans})

    # --------- Sources UI + export ---------
    with st.expander("Sources & timestamps", expanded=False):
        groups = group_hits_by_video(hits)
        ordered = sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)
        export_payload = {"videos":[], "web":[]}

        for vid, items in ordered:
            info = vm.get(vid,{})
            title = info.get("title") or summaries.get(vid,{}).get("title") or vid
            creator = info.get("podcaster") or info.get("channel") or summaries.get(vid,{}).get("channel") or ""
            url = info.get("url") or ""
            header = f"**{title}**" + (f" — _{creator}_" if creator else "")
            st.markdown(f"- [{header}]({url})" if url else f"- {header}")

            v_entry={"video_id":vid,"title":title,"creator":creator,"url":url,"quotes":[]}
            for h in sorted(items, key=lambda r: float(r["meta"].get("start",0))):
                ts=_format_ts(h["meta"].get("start",0))
                quote=(h["text"] or "").strip().replace("\n"," ")
                if len(quote)>160: quote=quote[:160]+"…"
                st.markdown(f"  • **{ts}** — “{quote}”")
                v_entry["quotes"].append({"ts":ts,"text":quote})
            export_payload["videos"].append(v_entry)

        if web_snips:
            st.markdown("**Trusted websites (supporting evidence)**")
            for j,s in enumerate(web_snips,1):
                st.markdown(f"W{j}. [{s['domain']}]({s['url']})")
                export_payload["web"].append({"id":f"W{j}","domain":s["domain"],"url":s["url"]})

        st.download_button("Download sources as JSON", data=json.dumps(export_payload, ensure_ascii=False, indent=2),
                           file_name="sources.json", mime="application/json")

# ------------------ Footer: precompute hint -------------------
st.caption(
    "If you add new videos or change chunks.jsonl, run: "
    "`DATA_DIR=/var/data python scripts/precompute_video_summaries.py` "
    "to refresh video centroids and summaries."
)

# ------------------ Self-check (quick runtime invariants) -------------------
# These checks surface as subtle UI changes or warnings rather than crashing.
try:
    # Encoder vs index dimension already validated. Here we ensure centroids, if present,
    # have matching width with encoder embeddings to avoid routing errors.
    if VID_CENT_NPY.exists():
        C_tmp = np.load(VID_CENT_NPY)
        emb_dim = _load_embedder(payload["model_name"]).get_sentence_embedding_dimension()
        if C_tmp.shape[1] != emb_dim:
            st.warning("Video centroids dim != encoder dim. Re-run precompute with the same model used for the index.", icon="⚠️")
except Exception:
    pass
