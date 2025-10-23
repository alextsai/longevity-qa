# app/app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS","1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")

from pathlib import Path
import sys, json, pickle, time, re
from typing import List, Dict, Any, Tuple
from datetime import datetime

import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

try:
    import requests
    from bs4 import BeautifulSoup
except Exception:
    requests=None; BeautifulSoup=None

# ------------ Paths ------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA_ROOT = Path(os.getenv("DATA_DIR","/var/data")).resolve()

CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"
OFFSETS_NPY     = DATA_ROOT / "data/chunks/chunks.offsets.npy"
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"

# NEW artifacts from precompute
VID_CENT_NPY    = DATA_ROOT / "data/index/video_centroids.npy"
VID_IDS_TXT     = DATA_ROOT / "data/index/video_ids.txt"
VID_SUM_JSON    = DATA_ROOT / "data/catalog/video_summaries.json"

REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]

# ------------ Labels ------------
try:
    from utils.labels import label_and_url
except Exception:
    def label_and_url(meta: dict) -> Tuple[str,str]:
        vid = meta.get("video_id") or "Unknown"
        ts = int(meta.get("start",0))
        return (f"{vid} @ {ts}s","")

# ------------ Trusted web domains ------------
TRUSTED_DOMAINS = [
    "nih.gov","medlineplus.gov","cdc.gov","mayoclinic.org","health.harvard.edu",
    "familydoctor.org","healthfinder.gov","ama-assn.org","medicalxpress.com",
    "sciencedaily.com","nejm.org","med.stanford.edu","icahn.mssm.edu"
]

# ------------ Utils ------------
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

@st.cache_resource(show_spinner=False)
def load_video_meta()->Dict[str,Dict[str,Any]]:
    if VIDEO_META_JSON.exists():
        try: return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

def _vid_epoch(vm:dict, vid:str)->float:
    info = (vm or {}).get(vid,{})
    return _iso_to_epoch(info.get("published_at") or info.get("publishedAt") or info.get("date") or "")

def _recency_score(published_ts:float, now:float, half_life_days:float)->float:
    if published_ts<=0: return 0.0
    days = max(0.0,(now - published_ts)/86400.0)
    return 0.5 ** (days / max(1e-6, half_life_days))

def _format_ts(sec:float)->str:
    sec = int(max(0,float(sec))); h,r=divmod(sec,3600); m,s=divmod(r,60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"

# ------------ Offsets ------------
def _ensure_offsets()->np.ndarray:
    if OFFSETS_NPY.exists():
        try: return np.load(OFFSETS_NPY)
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
    offsets=_ensure_offsets()
    want=[i for i in indices if 0<=i<len(offsets)]
    if limit is not None: want=want[:limit]
    with CHUNKS_PATH.open("rb") as f:
        for i in want:
            f.seek(int(offsets[i]))
            raw=f.readline()
            try: yield i, json.loads(raw)
            except: continue

# ------------ Model + FAISS ------------
@st.cache_resource(show_spinner=False)
def load_metas_and_model():
    if not INDEX_PATH.exists() or not METAS_PKL.exists():
        return None, None, None
    index = faiss.read_index(str(INDEX_PATH))
    with METAS_PKL.open("rb") as f:
        payload=pickle.load(f)
    metas_from_pkl = payload.get("metas",[])
    model_name = payload.get("model","sentence-transformers/all-MiniLM-L6-v2")
    local_dir = DATA_ROOT/"models"/"all-MiniLM-L6-v2"
    try_name = str(local_dir) if (local_dir/"config.json").exists() else model_name
    embedder = SentenceTransformer(try_name, device="cpu")
    return index, metas_from_pkl, {"model_name":try_name, "embedder":embedder}

@st.cache_resource(show_spinner=False)
def load_video_centroids():
    if not VID_CENT_NPY.exists() or not VID_IDS_TXT.exists():
        return None, None
    C = np.load(VID_CENT_NPY)  # [V,d] normalized
    vids = VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
    return C.astype("float32"), vids

@st.cache_resource(show_spinner=False)
def load_video_summaries():
    if VID_SUM_JSON.exists():
        try: return json.loads(VID_SUM_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

# ------------ MMR ------------
def mmr(qv:np.ndarray, doc_vecs:np.ndarray, k:int, lambda_diversity:float=0.4):
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

# ------------ Two-stage retrieval ------------
def stageA_route_videos(qv:np.ndarray, C:np.ndarray, vids:List[str], topN:int,
                        allowed_vids:set[str]|None, vm:dict,
                        recency_weight:float, half_life_days:float)->List[str]:
    # cosine is dot since normalized
    sims = (C @ qv.reshape(-1,1)).ravel()
    now=time.time()
    blend=[]
    for i,vid in enumerate(vids):
        if allowed_vids and vid not in allowed_vids: 
            continue
        rec=_recency_score(_vid_epoch(vm, vid), now, half_life_days)
        score = (1.0-recency_weight)*float(sims[i]) + recency_weight*float(rec)
        blend.append((vid, score))
    blend.sort(key=lambda x:-x[1])
    return [v for v,_ in blend[:topN]]

def stageB_search_chunks(query:str,
    index:faiss.Index, embedder:SentenceTransformer,
    candidate_vids:set[str],
    initial_k:int, final_k:int, max_videos:int, per_video_cap:int,
    apply_mmr:bool, mmr_lambda:float,
    recency_weight:float, half_life_days:float, vm:dict)->List[Dict[str,Any]]:

    qv = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
    K = min(int(initial_k), index.ntotal if index is not None else initial_k)
    D,I = index.search(qv.reshape(1,-1), K)
    idxs=[int(x) for x in I[0] if x>=0]
    scores0=[float(s) for s in D[0][:len(idxs)]]

    rows=list(iter_jsonl_rows(idxs))
    texts=[]; metas=[]; keep_mask=[]
    for _,j in rows:
        t=_normalize_text(j.get("text",""))
        m=(j.get("meta") or {}).copy()
        vid = m.get("video_id") or m.get("vid") or m.get("ytid") or j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id")
        if vid: m["video_id"]=vid
        if "start" not in m and "start_sec" in m: m["start"]=m.get("start_sec")
        m["start"]=_parse_ts(m.get("start",0))
        texts.append(t); metas.append(m)
        keep_mask.append((not candidate_vids) or (vid in candidate_vids))
    # filter to routed videos if provided
    if any(keep_mask):
        texts = [t for t,k in zip(texts,keep_mask) if k]
        metas = [m for m,k in zip(metas,keep_mask) if k]
        idxs  = [i for i,k in zip(idxs, keep_mask) if k]
        scores0=[s for s,k in zip(scores0,keep_mask) if k]
    if not texts: return []

    doc_vecs = embedder.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")
    order=list(range(len(texts)))
    if apply_mmr:
        sel=mmr(qv, doc_vecs, k=min(len(texts), max(8, final_k*2)), lambda_diversity=float(mmr_lambda))
        order=sel

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

# ------------ Grouping ------------
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
        info = vm.get(vid,{})
        title = info.get("title") or summaries.get(vid,{}).get("title") or vid
        channel = info.get("channel") or summaries.get(vid,{}).get("channel") or "Unknown"
        date = info.get("published_at") or info.get("publishedAt") or info.get("date") or summaries.get(vid,{}).get("published_at") or ""
        head=f"[Video {v_idx}] {title} — {channel}" + (f" — {date}" if date else "")
        lines.append(head)
        # add summary one-liner if we have it
        summ = summaries.get(vid,{}).get("summary","")
        if summ:
            lines.append(f"  • summary: {summ[:300]}{'…' if len(summ)>300 else ''}")
        # quoted spans
        for h in sorted(items, key=lambda r: float(r['meta'].get('start',0)))[:max_quotes]:
            ts=_format_ts(h["meta"].get("start",0))
            q=(h["text"] or "").strip().replace("\n"," ")
            if len(q)>260: q=q[:260]+"…"
            lines.append(f"  • {ts}: “{q}”")
        lines.append("")
    return "\n".join(lines).strip()

# ------------ Web fetch ------------
def fetch_trusted_snippets(query:str, allowed_domains:List[str], max_snippets:int=3, per_domain:int=1, timeout:float=6.0):
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

# ------------ LLM ------------
def openai_answer(model_name:str, question:str, history:List[Dict[str,str]], grouped_video_block:str, web_snips:List[Dict[str,str]])->str:
    if not os.getenv("OPENAI_API_KEY"): return "⚠️ OPENAI_API_KEY is not set."
    recent=history[-6:]
    convo=[]
    for m in recent:
        role=m.get("role"); content=m.get("content","")
        if role in ("user","assistant") and content:
            label="User" if role=="user" else "Assistant"
            convo.append(f"{label}: {content}")
    web_lines=[f"(W{j}) {s['domain']} — {s['url']}\n“{s['text'][:300]}”" for j,s in enumerate(web_snips,1)]
    web_block="\n".join(web_lines) if web_lines else "None"
    system=(
        "Answer ONLY from the provided evidence. Prioritize grouped video evidence, then trusted web snippets.\n"
        "Use at most ~4 videos and ~3 trusted web snippets. Merge findings; avoid listing every quote.\n"
        "Structure:\n"
        "• Key takeaways\n"
        "• Practical protocol (clear, stepwise)\n"
        "• Safety notes and when to consult a clinician\n"
        "Cite inline like (Video 2) or (CDC W1). If evidence is insufficient, say so."
    )
    user_payload=((("Recent conversation:\n"+"\n".join(convo)+"\n\n") if convo else "")
        + f"Question: {question}\n\n"
        + "Grouped Video Evidence:\n" + (grouped_video_block or "None") + "\n\n"
        + "Trusted Web Snippets:\n" + web_block + "\n\n"
        + "Write a concise, well-grounded answer.")
    try:
        client=OpenAI(timeout=45)
        r=client.chat.completions.create(
            model=model_name, temperature=0.2,
            messages=[{"role":"system","content":system},{"role":"user","content":user_payload}]
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ Generation error: {e}"

# ------------ UI ------------
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

with st.sidebar:
    st.header("How the answer is built")

    initial_k = st.number_input("How many passages to scan first", 32, 2000, 128, 32,
        help="First pass. Bigger can find more ideas, but is slower.")
    final_k = st.number_input("How many passages to use", 8, 60, 24, 2,
        help="Second pass. Too high can make answers ramble.")

    st.subheader("Keep it focused")
    max_videos = st.number_input("Maximum videos to use", 1, 12, 4, 1,
        help="At most this many different videos contribute to the answer.")
    per_video_cap = st.number_input("Passages per video", 1, 10, 3, 1,
        help="Stops any one video from dominating. 2–3 works well.")

    st.subheader("Balance variety and accuracy")
    use_mmr = st.checkbox("Encourage variety (recommended)", value=True,
        help="Avoids near-duplicate passages so evidence covers different angles.")
    mmr_lambda = st.slider("Balance: accuracy vs variety", 0.1, 0.9, 0.4, 0.05,
        help="Higher = closer match. Lower = more diverse viewpoints.")

    st.subheader("Prefer newer videos")
    recency_weight = st.slider("Recency influence", 0.0, 1.0, 0.30, 0.05,
        help="0 ignores date. 1 strongly prefers newer videos.")
    half_life = st.slider("How fast recency fades (days)", 7, 720, 180, 7,
        help="Every N days, recency value halves. Smaller favors very recent content.")

    st.subheader("Route to best videos first")
    topN_videos = st.number_input("Videos to consider before chunk search", 1, 30, 8, 1,
        help="Stage A. We pick up to this many likely videos first, then search inside them.")

    st.subheader("Pick sources to include")
    vm = load_video_meta()
    channels = sorted({(info.get("channel") or "Unknown") for info in vm.values()})
    chosen_channels = st.multiselect(
        "Channels to include", options=channels, default=channels,
        help="Only videos from these channels will be used."
    )
    # Build video choices filtered by chosen channels
    vids_by_channel = [vid for vid,info in vm.items() if (info.get("channel") or "Unknown") in chosen_channels]
    vid_labels = [f"{vm.get(vid,{}).get('title','(no title)')}  [{vid}]" for vid in vids_by_channel]
    chosen_vid_labels = st.multiselect(
        "Optional: pick specific videos (otherwise all from chosen channels)",
        options=vid_labels, default=[]
    )
    chosen_vids = set()
    if chosen_vid_labels:
        lookup = {f"{vm.get(vid,{}).get('title','(no title)')}  [{vid}]": vid for vid in vids_by_channel}
        chosen_vids = {lookup[x] for x in chosen_vid_labels}

    st.subheader("Trusted websites")
    allow_web = st.checkbox("Add short excerpts from trusted health sites", value=False)
    allowed_domains = st.multiselect(
        "Sites to include", options=TRUSTED_DOMAINS, default=TRUSTED_DOMAINS,
        help="Only these sites will be searched if enabled."
    )
    max_web = st.slider("Max website excerpts", 0, 8, 3, 1)

    st.subheader("Model")
    model_choice = st.selectbox("OpenAI model", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)

    st.divider()
    st.subheader("Library status")
    st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    st.checkbox("chunks.jsonl present", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("offsets built", value=OFFSETS_NPY.exists(), disabled=True)
    st.checkbox("faiss.index present", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("metas.pkl present", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("video_meta.json present", value=VIDEO_META_JSON.exists(), disabled=True)
    if VID_CENT_NPY.exists() and VID_IDS_TXT.exists():
        st.caption("Video centroids: ready")
    else:
        st.caption("Video centroids: not found (run scripts/precompute_video_summaries.py)")

# Chat history
if "messages" not in st.session_state: st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

prompt = st.chat_input("Ask about sleep, protein timing, fasting, supplements, protocols…")
if prompt is None: st.stop()
st.session_state.messages.append({"role":"user","content":prompt})
with st.chat_message("user"): st.markdown(prompt)

# Guardrails
missing=[p for p in REQUIRED if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error("Missing required files:\n" + "\n".join(f"- {p}" for p in missing))
    st.stop()

# Load models + artifacts
try:
    index, metas_from_pkl, payload = load_metas_and_model()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load index or encoder."); st.exception(e); st.stop()
if index is None or payload is None:
    with st.chat_message("assistant"): st.error("Index or model not available.")
    st.stop()
embedder: SentenceTransformer = payload["embedder"]
vm = load_video_meta()
C, vid_list = load_video_centroids()
summaries = load_video_summaries()

# Stage A: route to videos (optional)
routed_vids=[]
candidate_vids=set()
with st.spinner("Routing to likely videos…"):
    qv = embedder.encode([prompt], normalize_embeddings=True).astype("float32")[0]
    allowed_vids = set(chosen_vids) if chosen_vids else {vid for vid in (vid_list or vm.keys()) if (vm.get(vid,{}).get("channel") or "Unknown") in set(chosen_channels)}
    if C is not None and vid_list is not None:
        routed_vids = stageA_route_videos(qv, C, vid_list, int(topN_videos), allowed_vids, vm, float(recency_weight), float(half_life))
        candidate_vids = set(routed_vids)
    else:
        candidate_vids = allowed_vids  # fallback: only filter by source choices

# Stage B: search chunks inside routed/allowed videos
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
            st.error("Search failed."); st.exception(e); st.stop()

# Optional web
web_snips=[]
if allow_web and allowed_domains:
    with st.spinner("Fetching trusted websites…"):
        web_snips = fetch_trusted_snippets(prompt, allowed_domains, max_snippets=int(max_web))

# Build grouped evidence and answer
grouped_block = build_grouped_evidence_for_prompt(hits, vm, summaries, max_quotes=3)

with st.chat_message("assistant"):
    if not hits and not web_snips:
        st.warning("No relevant evidence found."); st.session_state.messages.append({"role":"assistant","content":"I couldn’t find enough evidence to answer that."}); st.stop()
    with st.spinner("Writing your answer…"):
        ans = openai_answer(model_choice, prompt, st.session_state.messages, grouped_block, web_snips)
    st.markdown(ans)
    st.session_state.messages.append({"role":"assistant","content":ans})

    # Grouped Sources UI
    with st.expander("Sources & timestamps", expanded=False):
        groups = group_hits_by_video(hits)
        ordered = sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)
        for vid, items in ordered:
            info = vm.get(vid,{})
            title = info.get("title") or summaries.get(vid,{}).get("title") or vid
            channel = info.get("channel") or summaries.get(vid,{}).get("channel") or ""
            url = info.get("url") or ""
            header = f"**{title}**" + (f" — _{channel}_" if channel else "")
            st.markdown(f"- [{header}]({url})" if url else f"- {header}")
            for h in sorted(items, key=lambda r: float(r["meta"].get("start",0))):
                ts=_format_ts(h["meta"].get("start",0))
                quote=(h["text"] or "").strip().replace("\n"," ")
                if len(quote)>140: quote=quote[:140]+"…"
                st.markdown(f"  • **{ts}** — “{quote}”")
        if web_snips:
            st.markdown("**Trusted websites**")
            for j,s in enumerate(web_snips,1):
                st.markdown(f"W{j}. [{s['domain']}]({s['url']})")
        st.caption("Videos are grouped with timestamps. Trusted sites appear separately.")
