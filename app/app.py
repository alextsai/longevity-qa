# app/app.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, pickle, time, re, math, collections
from pathlib import Path
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

# ------------ Paths ------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
DATA_ROOT       = Path(os.getenv("DATA_DIR","/var/data")).resolve()
CHUNKS_PATH     = DATA_ROOT/"data/chunks/chunks.jsonl"
OFFSETS_NPY     = DATA_ROOT/"data/chunks/chunks.offsets.npy"
INDEX_PATH      = DATA_ROOT/"data/index/faiss.index"
METAS_PKL       = DATA_ROOT/"data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT/"data/catalog/video_meta.json"
VID_CENT_NPY    = DATA_ROOT/"data/index/video_centroids.npy"
VID_IDS_TXT     = DATA_ROOT/"data/index/video_ids.txt"
VID_SUM_JSON    = DATA_ROOT/"data/catalog/video_summaries.json"
REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]

# ------------ Knobs (hard relevance) ------------
MIN_SIM  = float(os.getenv("MIN_SIM", "0.42"))         # cosine to query
MIN_KW   = float(os.getenv("MIN_KW", "0.03"))          # lexical overlap
MIN_CHARS= int(os.getenv("MIN_CHARS", "40"))
MIN_VID_QUOTES = int(os.getenv("MIN_VID_QUOTES","1"))  # per routed video
MIN_VIDEOS     = int(os.getenv("MIN_VIDEOS","2"))      # overall
WEB_FALLBACK   = os.getenv("WEB_FALLBACK","false").lower() in {"1","true","yes"}

TRUSTED_DOMAINS = [
    "nih.gov","medlineplus.gov","cdc.gov","mayoclinic.org","health.harvard.edu",
    "familydoctor.org","healthfinder.gov","ama-assn.org","medicalxpress.com",
    "sciencedaily.com","nejm.org","med.stanford.edu","icahn.mssm.edu"
]

ALLOWED_CREATORS = [
    "Dr. Pradip Jamnadas, MD","Andrew Huberman","Healthy Immune Doc",
    "Peter Attia MD","The Diary of A CEO",
]
EXCLUDED_CREATORS_EXACT = {
    "they diary of a ceo and louse tomlinson",
    "dr. pradip jamnadas, md and the primal podcast",
}
CREATOR_SYNONYMS = {
    "heathy immune doc":"Healthy Immune Doc","heathly immune doc":"Healthy Immune Doc",
    "healthy immune doc":"Healthy Immune Doc","healthy  immune  doc":"Healthy Immune Doc",
    "healthy immune doc youtube":"Healthy Immune Doc","healthy immune doc ":"Healthy Immune Doc",
    "the diary of a ceo":"The Diary of A CEO","diary of a ceo":"The Diary of A CEO",
    "the diary of a ceo podcast":"The Diary of A CEO"
}

# ------------ Utils ------------
def _normalize_text(s:str)->str: return re.sub(r"\s+"," ",(s or "").strip())
def _parse_ts(v)->float:
    try:
        if isinstance(v,(int,float)): return float(v)
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
    sec=int(max(0,float(sec))); h,r=divmod(sec,3600); m,s=divmod(r,60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"
def _file_mtime(p:Path)->float:
    try: return p.stat().st_mtime
    except: return 0.0
def _iso(ts: float) -> str:
    try: return datetime.fromtimestamp(ts).isoformat()
    except: return "n/a"

def _is_admin()->bool:
    try: qp=st.query_params
    except: return False
    if qp.get("admin","0")!="1": return False
    try: expected=st.secrets["ADMIN_KEY"]
    except Exception: expected=None
    return True if expected is None else qp.get("key","")==str(expected)

def canonicalize_creator(name:str)->str|None:
    n=(name or "").strip()
    if not n: return None
    low=re.sub(r"\s+"," ",n.lower()).replace("™","").replace("®","").strip()
    if low in EXCLUDED_CREATORS_EXACT: return None
    low=CREATOR_SYNONYMS.get(low,low)
    for canon in ALLOWED_CREATORS:
        if low==canon.lower(): return canon
    strip_punct=re.sub(r"[^\w\s]","",low)
    for canon in ALLOWED_CREATORS:
        if strip_punct==re.sub(r"[^\w\s]","",canon.lower()): return canon
    return None

# ------------ Load ------------
@st.cache_data(show_spinner=False)
def load_video_meta()->Dict[str,Dict[str,Any]]:
    if VIDEO_META_JSON.exists():
        try: return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

def _raw_creator_of_vid(vid:str, vm:dict)->str:
    info=vm.get(vid,{}) or {}
    for k in ("podcaster","channel","author","uploader","owner","creator"):
        if info.get(k): return str(info[k])
    for k,v in ((kk.lower(),vv) for kk,vv in info.items()):
        if k in {"podcaster","channel","author","uploader","owner","creator"} and v: return str(v)
    return "Unknown"

def _vid_epoch(vm:dict, vid:str)->float:
    info=(vm or {}).get(vid,{})
    return _iso_to_epoch(info.get("published_at") or info.get("publishedAt") or info.get("date") or "")

def _recency_score(published_ts:float, now:float, half_life_days:float)->float:
    if published_ts<=0: return 0.0
    days=max(0.0,(now-published_ts)/86400.0)
    return 0.5 ** (days/max(1e-6,half_life_days))

@st.cache_resource(show_spinner=False)
def _load_embedder(model_name:str)->SentenceTransformer:
    return SentenceTransformer(model_name, device="cpu")

@st.cache_resource(show_spinner=False)
def load_metas_and_model(index_path:Path=INDEX_PATH, metas_path:Path=METAS_PKL):
    if not index_path.exists() or not metas_path.exists(): return None,None,None
    index=faiss.read_index(str(index_path))
    with metas_path.open("rb") as f: payload=pickle.load(f)
    model_name=payload.get("model","sentence-transformers/all-MiniLM-L6-v2")
    local=DATA_ROOT/"models"/"all-MiniLM-L6-v2"
    try_name=str(local) if (local/"config.json").exists() else model_name
    embedder=_load_embedder(try_name)
    if index.d!=embedder.get_sentence_embedding_dimension():
        raise RuntimeError("Embedding dim mismatch. Rebuild.")
    return index, payload.get("metas",[]), {"model_name":try_name,"embedder":embedder}

@st.cache_resource(show_spinner=False)
def load_video_centroids():
    if not (VID_CENT_NPY.exists() and VID_IDS_TXT.exists()): return None,None
    C=np.load(VID_CENT_NPY).astype("float32")
    vids=VID_IDS_TXT.read_text(encoding="utf-8").splitlines()
    if C.shape[0]!=len(vids): return None,None
    return C, vids

@st.cache_data(show_spinner=False)
def load_video_summaries():
    if VID_SUM_JSON.exists():
        try: return json.loads(VID_SUM_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

# ------------ JSONL offsets ------------
def _ensure_offsets()->np.ndarray:
    if OFFSETS_NPY.exists():
        try:
            arr=np.load(OFFSETS_NPY); saved=len(arr); cur=sum(1 for _ in CHUNKS_PATH.open("rb"))
            if cur<=saved: return arr
        except: pass
    pos=0; offs=[]
    with CHUNKS_PATH.open("rb") as f:
        for ln in f: offs.append(pos); pos+=len(ln)
    arr=np.array(offs,dtype=np.int64); OFFSETS_NPY.parent.mkdir(parents=True,exist_ok=True)
    np.save(OFFSETS_NPY,arr); return arr

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

# ------------ Creator index from chunks ------------
def build_creator_indexes_from_chunks(vm:dict)->tuple[dict,dict]:
    vid_to_creator:Dict[str,str]={}; creator_to_vids:Dict[str,set]={}
    if CHUNKS_PATH.exists():
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            for ln in f:
                try: j=json.loads(ln)
                except: continue
                m=(j.get("meta") or {})
                vid = (m.get("video_id") or m.get("vid") or m.get("ytid") or
                       j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
                if not vid: continue
                raw=(m.get("channel") or m.get("author") or m.get("uploader") or _raw_creator_of_vid(vid,vm))
                canon=canonicalize_creator(raw)
                if canon is None: continue
                if vid not in vid_to_creator:
                    vid_to_creator[vid]=canon
                    creator_to_vids.setdefault(canon,set()).add(vid)
    for vid in vm.keys():
        if vid in vid_to_creator: continue
        canon=canonicalize_creator(_raw_creator_of_vid(vid,vm))
        if canon is None: continue
        vid_to_creator[vid]=canon
        creator_to_vids.setdefault(canon,set()).add(vid)
    return vid_to_creator, creator_to_vids

# ------------ Routing ------------
def _build_idf_over_bullets(summaries: dict) -> dict:
    DF=collections.Counter()
    vids=list(summaries.keys())
    for v in vids:
        for b in summaries.get(v,{}).get("bullets",[]):
            for w in set((b.get("text","") or "").lower().split()): DF[w]+=1
    N=max(1,len(vids))
    return {w: math.log((N+1)/(df+0.5)) for w,df in DF.items()}

def _kw_score(text:str, query:str, idf:dict)->float:
    q=[w for w in (query or "").lower().split() if w]
    t=(text or "").lower().split()
    tf={w:t.count(w) for w in set(t)}
    return sum(tf.get(w,0)*idf.get(w,0.0) for w in set(q))/(len(t)+1e-6)

def route_videos_by_summary(query:str, qv:np.ndarray, summaries:dict, vm:dict,
                            C:np.ndarray|None, vids:list[str]|None,
                            allowed_vids:set[str], topK:int,
                            recency_weight:float, half_life_days:float)->list[str]:
    universe=[v for v in (vids or list(vm.keys())) if (not allowed_vids or v in allowed_vids)]
    if not universe: return []
    cent={}
    if C is not None and vids is not None and len(vids)==C.shape[0]:
        sim=(C @ qv.reshape(-1,1)).ravel(); cent={vids[i]:float(sim[i]) for i in range(len(vids))}
    idf=_build_idf_over_bullets(summaries); now=time.time(); scored=[]
    for v in universe:
        bullets=summaries.get(v,{}).get("bullets",[])
        pseudo=" ".join(b.get("text","") for b in bullets)[:2000]
        kw=_kw_score(pseudo, query, idf); cs=cent.get(v,0.0)
        rec=_recency_score(_vid_epoch(vm,v), now, half_life_days)
        base=0.6*cs+0.3*kw; score=(1.0-recency_weight)*base+recency_weight*(0.1*rec+0.9*base)
        scored.append((v,score))
    scored.sort(key=lambda x:-x[1])
    return [v for v,_ in scored[:int(topK)]]

# ------------ Relevance gating ------------
def _cosine(a:np.ndarray,b:np.ndarray)->float:
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def _kw_overlap(a:str,b:str)->float:
    A=set(w for w in re.findall(r"[a-z0-9]+",a.lower()) if len(w)>2)
    B=set(w for w in re.findall(r"[a-z0-9]+",b.lower()) if len(w)>2)
    if not A or not B: return 0.0
    return len(A&B)/max(1,len(A|B))

def quote_is_valid(query:str, qv:np.ndarray, txt:str, dv:np.ndarray)->bool:
    if len(_normalize_text(txt))<MIN_CHARS: return False
    if _cosine(qv,dv)<MIN_SIM and _kw_overlap(query,txt)<MIN_KW: return False
    return True

def _dedupe_passages(items:List[Dict[str,Any]], time_window_sec:float=8.0):
    out=[]; seen=[]
    for h in sorted(items,key=lambda r: float((r.get("meta") or {}).get("start",0))):
        ts=float((h.get("meta") or {}).get("start",0)); txt=_normalize_text(h.get("text",""))
        if any(abs(ts-float((s.get("meta") or {}).get("start",0)))<=time_window_sec and
               _normalize_text(s.get("text",""))==txt for s in seen): continue
        seen.append(h); out.append(h)
    return out

# ------------ Stage B search with gates ------------
def stageB_search_chunks(query:str, index:faiss.Index, embedder:SentenceTransformer,
    candidate_vids:Set[str]|None, initial_k:int, final_k:int, max_videos:int, per_video_cap:int,
    apply_mmr:bool, mmr_lambda:float, recency_weight:float, half_life_days:float, vm:dict)->List[Dict[str,Any]]:

    if index is None: return []
    qv=embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
    K=min(int(initial_k), index.ntotal if index.ntotal>0 else int(initial_k))
    D,I=index.search(qv.reshape(1,-1), K)
    idxs=[int(x) for x in I[0] if x>=0]

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
        if t and ((candidate_vids is None) or (vid in candidate_vids)):
            texts.append(t); metas.append(m); keep.append(True)

    if not texts: return []
    doc_vecs=embedder.encode(texts, normalize_embeddings=True, batch_size=64).astype("float32")

    # Hard gate by semantic + lexical
    gated=[]
    for t,m,dv in zip(texts,metas,doc_vecs):
        if quote_is_valid(query,qv,t,dv): gated.append((t,m,dv))
    if not gated: return []

    # MMR order
    order=list(range(len(gated)))
    if apply_mmr:
        sim=np.array([_cosine(qv,dv) for _,_,dv in gated],dtype=np.float32)
        # pick k with MMR
        selected=[]; cand=set(order)
        while cand and len(selected)<min(len(gated), max(8, final_k*2)):
            if not selected:
                i=int(max(cand, key=lambda i: sim[i])); selected.append(i); cand.remove(i); continue
            maxdiv=np.array([max([np.dot(gated[i][2],gated[j][2]) for j in selected]) for i in cand])
            cidx=list(cand); scores=mmr_lambda*sim[cidx] - (1-mmr_lambda)*maxdiv
            pick=cidx[int(np.argmax(scores))]; selected.append(pick); cand.remove(pick)
        order=selected

    now=time.time(); blended=[]
    for li in order:
        t,m,dv=gated[li]; vid=m.get("video_id")
        rec=_recency_score(_vid_epoch(vm,vid), now, half_life_days)
        score=(1.0-recency_weight)*_cosine(qv,dv)+recency_weight*rec
        blended.append((score,t,m))
    blended.sort(key=lambda x:-x[0])

    picked=[]; per_vid={}; distinct=[]
    for sc,tx,me in blended:
        vid=me.get("video_id","Unknown")
        if vid not in distinct and len(distinct)>=int(max_videos): continue
        if per_vid.get(vid,0)>=int(per_video_cap): continue
        if vid not in distinct: distinct.append(vid)
        per_vid[vid]=per_vid.get(vid,0)+1
        picked.append({"score":float(sc),"text":tx,"meta":me})
        if len(picked)>=int(final_k): break

    # Enforce per-video minimum; drop routed videos that produced no valid quotes
    g=collections.defaultdict(list)
    for h in picked: g[h["meta"].get("video_id","Unknown")].append(h)
    kept=[]
    for vid,items in g.items():
        if len(items)>=MIN_VID_QUOTES: kept.extend(items)
    return kept

# ------------ Group + export ------------
def group_hits_by_video(hits:List[Dict[str,Any]])->Dict[str,List[Dict[str,Any]]]:
    g=collections.defaultdict(list)
    for h in hits:
        vid=(h.get("meta") or {}).get("video_id") or "Unknown"
        g[vid].append(h)
    return g

def build_grouped_block(hits:List[Dict[str,Any]], vm:dict, summaries:dict, max_quotes:int=3)->tuple[str,list]:
    groups=group_hits_by_video(hits)
    ordered=sorted(groups.items(), key=lambda kv: max(x["score"] for x in kv[1]), reverse=True)
    lines=[]; export=[]
    for v_idx,(vid,items) in enumerate(ordered,1):
        info=vm.get(vid,{})
        title=info.get("title") or summaries.get(vid,{}).get("title") or vid
        creator_raw=_raw_creator_of_vid(vid,vm); creator=canonicalize_creator(creator_raw) or creator_raw
        date=info.get("published_at") or info.get("publishedAt") or info.get("date") or summaries.get(vid,{}).get("published_at") or ""
        url=info.get("url") or ""
        lines.append(f"[Video {v_idx}] {title} — {creator}" + (f" — {date}" if date else ""))
        clean=_dedupe_passages(items, time_window_sec=8.0)
        v={"video_id":vid,"title":title,"creator":creator,"url":url,"quotes":[]}
        for h in clean[:max_quotes]:
            ts=_format_ts((h.get("meta") or {}).get("start",0))
            q=_normalize_text(h.get("text","")); q=q[:260]+"…" if len(q)>260 else q
            lines.append(f"  • {ts}: “{q}”"); v["quotes"].append({"ts":ts,"text":q})
        export.append(v); lines.append("")
    return ("\n".join(lines).strip(), export)

# ------------ Web support ------------
def _ddg_links(domain:str, query:str, timeout:float)->list[str]:
    if not requests: return []
    try:
        # try lite first
        r=requests.get("https://duckduckgo.com/lite/", params={"q":f"site:{domain} {query}"}, timeout=timeout)
        if r.status_code!=200: return []
        soup=BeautifulSoup(r.text,"html.parser")
        links=[a.get("href") for a in soup.select("a") if a.get("href")]
        links=[u for u in links if domain in u and not re.search(r"/(home|index)\.?\w*$", u)]
        return links[:2]
    except Exception:
        return []

def fetch_trusted_snippets(query:str, allowed_domains:List[str], max_snippets:int=3, per_domain:int=1, timeout:float=8.0):
    if not (requests and BeautifulSoup) or max_snippets<=0: return []
    headers={"User-Agent":"Mozilla/5.0"}
    out=[]
    q_terms=set(w for w in re.findall(r"[a-z0-9]+",query.lower()) if len(w)>2)
    for domain in allowed_domains:
        links=_ddg_links(domain, query, timeout)[:per_domain]
        for url in links:
            try:
                r=requests.get(url, headers=headers, timeout=timeout)
                if r.status_code!=200: continue
                soup=BeautifulSoup(r.text,"html.parser")
                paras=[p.get_text(" ",strip=True) for p in soup.find_all("p")]
                text=_normalize_text(" ".join(paras))[:2000]
                if len(text)<200: continue
                # require at least one query term to appear
                if not any(t in text.lower() for t in q_terms): continue
                out.append({"domain":domain,"url":url,"text":text})
            except: continue
        if len(out)>=max_snippets: break
    return out[:max_snippets]

# ------------ LLM ------------
def openai_answer(model_name:str, question:str, history:List[Dict[str,str]],
                  grouped_video_block:str, web_snips:List[Dict[str,str]], no_video:bool)->str:
    if not os.getenv("OPENAI_API_KEY"): return "⚠️ OPENAI_API_KEY is not set."
    recent=[m for m in history[-6:] if m.get("role") in ("user","assistant")]
    convo=[("User: " if m["role"]=="user" else "Assistant: ")+m.get("content","") for m in recent]
    web_lines=[f"(W{j}) {s.get('domain','web')} — {s.get('url','')}\n“{s.get('text','')[:300]}”" for j,s in enumerate(web_snips,1)]
    web_block="\n".join(web_lines) if web_lines else "None"
    fallback_line = ("If no gated video quotes exist, reply: 'Web-only evidence'.\n" if (WEB_FALLBACK and no_video)
                     else "Use web as supporting evidence only.\n")
    system=("Answer only from the provided evidence. Priority: (1) VIDEO quotes, (2) trusted WEB snippets.\n"+
            fallback_line+
            "Rules:\n"
            "• Do not cite a video without a quote.\n"
            "• If no video quotes meet relevance gates, say 'Insufficient expert video evidence' unless WEB-only mode is allowed.\n"
            "• Cite every claim with (Video k) or (DOMAIN Wj). Be concise.\n")
    user_payload=(("Recent conversation:\n"+"\n".join(convo)+"\n\n") if convo else "")+ \
                 f"Question: {question}\n\n"+"Grouped Video Evidence:\n"+(grouped_video_block or "None")+"\n\n"+ \
                 "Trusted Web Snippets:\n"+web_block+"\n\n"+"Write a concise, source-grounded answer."
    try:
        client=OpenAI(timeout=60)
        r=client.chat.completions.create(model=model_name, temperature=0.2,
               messages=[{"role":"system","content":system},{"role":"user","content":user_payload}])
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ Generation error: {e}"

# ------------ UI ------------
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

def _clear_chat():
    st.session_state["messages"]=[]; st.session_state["turn_sources"]=[]; st.rerun()

with st.sidebar:
    st.markdown("**Auto Mode**")
    vm=load_video_meta()
    vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
    counts={c: len(creator_to_vids.get(c,set())) for c in ALLOWED_CREATORS}
    st.subheader("Experts")
    st.caption("All selected. Uncheck to exclude.")
    selected=set()
    for i,c in enumerate(ALLOWED_CREATORS):
        if st.checkbox(f"{c} ({counts.get(c,0)})", value=True, key=f"exp_{i}"): selected.add(c)
    st.session_state["selected_creators"]=selected

    st.subheader("Trusted sites")
    allow_web=st.checkbox("Include supporting website excerpts", value=True)
    sel_domains=[]; 
    for i,dom in enumerate(TRUSTED_DOMAINS):
        if st.checkbox(dom, value=True, key=f"site_{i}"): sel_domains.append(dom)
    model_choice=st.selectbox("Answering model", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)

    with st.expander("Advanced", expanded=False):
        K_scan=st.number_input("Scan candidates first (K)", 256, 5000, 1024, 64)
        K_use =st.number_input("Use top passages", 8, 60, 36, 2)
        max_videos=st.number_input("Max videos", 1, 12, 5, 1)
        per_video_cap=st.number_input("Passages per video cap", 1, 10, 4, 1)
        use_mmr=st.checkbox("Diversify with MMR", value=True)
        mmr_lambda=st.slider("MMR balance", 0.1, 0.9, 0.45, 0.05)
        recency_weight=st.slider("Recency weight", 0.0, 1.0, 0.20, 0.05)
        half_life=st.slider("Recency half-life (days)", 7, 720, 270, 7)

    st.divider()
    if st.toggle("Show data diagnostics", value=False):
        st.caption(f"DATA_DIR = `{DATA_ROOT}`")
        st.caption(f"chunks mtime: {_iso(_file_mtime(CHUNKS_PATH))}")
        st.caption(f"centroids mtime: {_iso(_file_mtime(VID_CENT_NPY))}")
        st.caption(f"ids mtime: {_iso(_file_mtime(VID_IDS_TXT))}")
        st.caption(f"summaries mtime: {_iso(_file_mtime(VID_SUM_JSON))}")
    st.subheader("Library status")
    st.checkbox("chunks.jsonl present", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("faiss.index present", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("metas.pkl present", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("video_meta.json present", value=VIDEO_META_JSON.exists(), disabled=True)

# render prior turns
if "messages" not in st.session_state: st.session_state.messages=[]
if "turn_sources" not in st.session_state: st.session_state.turn_sources=[]
for i,(m) in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]): st.markdown(m["content"])
    if m["role"]=="assistant" and i//2 < len(st.session_state.turn_sources):
        src = st.session_state.turn_sources[i//2]
        with st.expander("Sources for this reply", expanded=False):
            if src.get("videos"):
                for v in src["videos"]:
                    head=f"**{v['title']}**" + (f" — _{v['creator']}_" if v.get("creator") else "")
                    if v.get("url"): st.markdown(f"- [{head}]({v['url']})")
                    else: st.markdown(f"- {head}")
                    for q in v["quotes"]:
                        st.markdown(f"  • **{q['ts']}** — “{q['text']}”")
            if src.get("web"):
                st.markdown("**Trusted websites**")
                for w in src["web"]:
                    st.markdown(f"{w['id']}. [{w['domain']}]({w['url']})")

# admin helpers
if _is_admin():
    with st.expander("Creator inventory (from chunks.jsonl)"):
        vm_admin=load_video_meta()
        inv=sorted(((c,len(vs)) for c,vs in build_creator_indexes_from_chunks(vm_admin)[1].items()), key=lambda x:-x[1])
        st.dataframe([{"creator":c,"videos":n} for c,n in inv], use_container_width=True)

prompt = st.chat_input("Ask a question…")
if prompt is None:
    cols=st.columns([1]*12)
    with cols[-1]: st.button("Clear chat", on_click=_clear_chat)
    st.stop()

# append user
st.session_state.messages.append({"role":"user","content":prompt})
with st.chat_message("user"): st.markdown(prompt)

# guard
missing=[p for p in REQUIRED if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error("Missing required files:\n"+"\n".join(f"- {p}" for p in missing))
    st.stop()

# load index/encoder/artifacts
try:
    index,_,payload=load_metas_and_model()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load index or encoder."); st.exception(e)
    st.stop()
embedder:SentenceTransformer=payload["embedder"]
vm=load_video_meta(); C, vid_list = load_video_centroids(); summaries=load_video_summaries()

# allowed universe
vid_to_creator, creator_to_vids = build_creator_indexes_from_chunks(vm)
universe=set(vid_list or list(vm.keys()) or list(vid_to_creator.keys()))
chosen=st.session_state.get("selected_creators", set(ALLOWED_CREATORS))
allowed_vids={vid for vid in universe if vid_to_creator.get(vid) in chosen}

# routing
routing_query=prompt
if len([m for m in st.session_state.messages if m["role"]=="user"])>=2:
    prev=[m["content"] for m in st.session_state.messages if m["role"]=="user"][-2:-1]
    if prev: routing_query = prev[0] + " ; " + prompt
qv=embedder.encode([routing_query], normalize_embeddings=True).astype("float32")[0]
routed_vids=route_videos_by_summary(routing_query, qv, summaries, vm, C, list(universe),
                                    allowed_vids, topK=5, recency_weight=float(recency_weight), half_life_days=float(half_life))
candidate_vids=set(routed_vids) if routed_vids else allowed_vids

# search with hard gates
with st.spinner("Scanning selected videos…"):
    hits=stageB_search_chunks(prompt, index, embedder, candidate_vids,
                              int(K_scan), int(K_use), int(max_videos), int(per_video_cap),
                              bool(use_mmr), float(mmr_lambda),
                              float(recency_weight), float(half_life), vm)

# require minimum coverage
videos_with_quotes=len(group_hits_by_video(hits))
if videos_with_quotes<MIN_VIDEOS and not WEB_FALLBACK:
    with st.chat_message("assistant"):
        st.warning("Insufficient expert video evidence after relevance filtering.")
    st.stop()

# web
web_snips=[]
if allow_web and sel_domains and requests and BeautifulSoup:
    with st.spinner("Fetching trusted websites…"):
        web_snips=fetch_trusted_snippets(prompt, sel_domains, max_snippets=3, per_domain=1)

# grouped block + export
grouped_block, export_videos = build_grouped_block(hits, vm, summaries, max_quotes=3)

with st.chat_message("assistant"):
    if not hits and not web_snips:
        st.warning("No relevant evidence found.")
        st.session_state.messages.append({"role":"assistant","content":"Insufficient evidence."})
        st.session_state.turn_sources.append({"videos":[],"web":[]})
        st.stop()
    with st.spinner("Writing your answer…"):
        ans=openai_answer(model_choice, prompt, st.session_state.messages, grouped_block, web_snips, no_video=(len(hits)==0))
    st.markdown(ans)
    st.session_state.messages.append({"role":"assistant","content":ans})

    # per-turn sources persist
    export={"videos":export_videos, "web":[{"id":f"W{j}","domain":s["domain"],"url":s["url"]} for j,s in enumerate(web_snips,1)]}
    st.session_state.turn_sources.append(export)

    with st.expander("Sources for this reply", expanded=False):
        if export_videos:
            for v in export_videos:
                head=f"**{v['title']}**" + (f" — _{v['creator']}_" if v.get("creator") else "")
                if v.get("url"): st.markdown(f"- [{head}]({v['url']})")
                else: st.markdown(f"- {head}")
                for q in v["quotes"]:
                    st.markdown(f"  • **{q['ts']}** — “{q['text']}”")
        if web_snips:
            st.markdown("**Trusted websites**")
            for j,s in enumerate(web_snips,1):
                st.markdown(f"W{j}. [{s['domain']}]({s['url']})")

# footer
cols=st.columns([1]*12)
with cols[-1]: st.button("Clear chat", on_click=lambda: (_clear_chat()))
