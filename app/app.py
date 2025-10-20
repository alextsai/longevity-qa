# app/app.py
from __future__ import annotations
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY","1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS","1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
DATA_ROOT_ENV=os.getenv("DATA_DIR","/var/data")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME",f"{DATA_ROOT_ENV}/models")
os.environ.setdefault("HF_HOME",f"{DATA_ROOT_ENV}/models")

from pathlib import Path
import sys, json, pickle
from typing import List, Dict, Any, Tuple
import streamlit as st
import numpy as np
import faiss
try:
    import torch; torch.set_num_threads(1)
except Exception:
    pass
from openai import OpenAI
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
DATA_ROOT = Path(DATA_ROOT_ENV).resolve()
from utils.labels import label_and_url

CHUNKS_PATH = DATA_ROOT/"data/chunks/chunks.jsonl"
INDEX_PATH  = DATA_ROOT/"data/index/faiss.index"
METAS_PKL   = DATA_ROOT/"data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT/"data/catalog/video_meta.json"
REQUIRED=[INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]

def _parse_ts(v)->float:
    if isinstance(v,(int,float)):
        try: return float(v)
        except: return 0.0
    if isinstance(v,str):
        try:
            sec=0.0
            for p in v.split(":"): sec=sec*60+float(p)
            return sec
        except: return 0.0
    return 0.0

@st.cache_resource(show_spinner=False)
def load_video_meta()->Dict[str,Dict[str,str]]:
    if VIDEO_META_JSON.exists():
        try: return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except: return {}
    return {}

@st.cache_resource(show_spinner=False)
def load_chunks_aligned()->Tuple[List[str],List[dict]]:
    texts, metas = [], []
    if not CHUNKS_PATH.exists(): return texts, metas
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            try: j=json.loads(line)
            except: continue
            t=(j.get("text") or "").strip()
            m=(j.get("meta") or {}).copy()
            vid = m.get("video_id") or m.get("vid") or m.get("ytid") or j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id")
            if vid: m["video_id"]=vid
            if "start" not in m and "start_sec" in m: m["start"]=m["start_sec"]
            m["start"]=_parse_ts(m.get("start",0))
            if t: texts.append(t); metas.append(m)
    return texts, metas

@st.cache_resource(show_spinner=False)
def load_index_and_model():
    if not INDEX_PATH.exists() or not METAS_PKL.exists():
        return None, None, None
    index = faiss.read_index(str(INDEX_PATH))
    with METAS_PKL.open("rb") as f: payload=pickle.load(f)
    model_name = payload.get("model","sentence-transformers/all-MiniLM-L6-v2")
    embedder = SentenceTransformer(model_name, device="cpu")
    return index, payload.get("metas", []), {"model_name":model_name,"embedder":embedder}

def mmr_rerank_lite(q_emb, cand_embs, hits, final_k, per_video_cap, lam=0.4):
    # Hard pool cap to protect memory/latency
    pool = min(len(hits), len(hits), 150)
    cand_embs = cand_embs[:pool]
    hits = hits[:pool]
    sim_q = cand_embs @ q_emb
    sims = cand_embs @ cand_embs.T
    selected=[]; counts={}
    if pool==0: return []
    seed=int(np.argmax(sim_q)); v=hits[seed]["meta"].get("video_id") or "unknown"
    selected.append(seed); counts[v]=1
    remain=set(range(pool)); remain.discard(seed)
    while len(selected)<final_k and remain:
        best=None; bestscore=-1e9
        for i in list(remain):
            vid = hits[i]["meta"].get("video_id") or "unknown"
            if counts.get(vid,0)>=per_video_cap: 
                remain.discard(i); continue
            max_sim = np.max(sims[i, selected]) if selected else 0.0
            score = lam*sim_q[i] - (1.0-lam)*max_sim
            if score>bestscore: bestscore=score; best=i
        if best is None: break
        selected.append(best); remain.discard(best)
        vid = hits[best]["meta"].get("video_id") or "unknown"
        counts[vid]=counts.get(vid,0)+1
    return [hits[i] for i in selected]

def search_chunks(query, index, embedder, metas, texts, initial_k, final_k, per_video_cap, use_mmr, lambda_div):
    if not query.strip(): return []
    with torch.no_grad() if "torch" in sys.modules else nullcontext():
        q = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]
    K = max(1, min(int(initial_k), int(index.ntotal)))
    D, I = index.search(q.reshape(1,-1), K)
    idxs, scores = I[0].tolist(), D[0].tolist()
    raw=[]
    for i,s in zip(idxs,scores):
        if 0<=i<len(texts):
            m = metas[i] if i<len(metas) else {}
            if not m.get("video_id"):
                vid=m.get("vid") or m.get("ytid")
                if vid: m["video_id"]=vid
            m["start"]=_parse_ts(m.get("start",0))
            raw.append({"i":i,"score":float(s),"text":texts[i],"meta":m})
    if not raw: return []
    if not use_mmr:
        # simple per-video cap then trim
        out=[]; c={}
        for h in raw:
            vid=h["meta"].get("video_id") or "unknown"
            c[vid]=c.get(vid,0)+1
            if c[vid]<=per_video_cap: out.append(h)
            if len(out)>=final_k: break
        return out
    # MMR on a small pool only
    pool = min(len(raw), max(final_k*3, 60), 150)
    cand_texts=[h["text"] for h in raw[:pool]]
    with torch.no_grad() if "torch" in sys.modules else nullcontext():
        embs = embedder.encode(cand_texts, normalize_embeddings=True, batch_size=64).astype("float32")
    return mmr_rerank_lite(q, embs, raw[:pool], int(final_k), int(per_video_cap), lambda_div)

def openai_answer(model_name, question, history, hits)->str:
    key=os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st,"secrets") else None)
    if not key: return "⚠️ OPENAI_API_KEY is not set."
    recent=[]
    for m in reversed(history[:-1]):
        if m.get("role") in ("user","assistant") and m.get("content"): recent.append(m)
        if len(recent)>=6: break
    convo=[]
    for m in reversed(recent):
        label="User" if m["role"]=="user" else "Assistant"
        convo.append(f"{label}: {m['content']}")
    ev=[]; 
    for i,h in enumerate(hits,1):
        lbl,_=label_and_url(h["meta"])
        ev.append(f"[{i}] {lbl}\n{h['text']}\n")
    system=(
        "Answer ONLY from the excerpts. Use short quoted phrases with [n] citations. "
        "If evidence is insufficient, say you don't know. Be concise."
    )
    user_payload=(("Recent conversation:\n"+"\n".join(convo)+"\n\n") if convo else "") + \
        f"Question: {question}\n\nExcerpts:\n" + "\n".join(ev) + \
        "\nWrite the best possible answer consistent with the excerpts. Quote fragments and cite like [n]."
    try:
        client=OpenAI()
        r=client.chat.completions.create(
            model=model_name, temperature=0.2, timeout=40,
            messages=[{"role":"system","content":system},{"role":"user","content":user_payload}],
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ OpenAI error: {e}"

def openai_ping(model_name)->str:
    key=os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st,"secrets") else None)
    if not key: return "NO_KEY"
    try:
        client=OpenAI()
        r=client.chat.completions.create(model=model_name, temperature=0, messages=[{"role":"user","content":"Return 'pong'."}])
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERR:{e}"

# ---- UI ----
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

with st.sidebar:
    st.header("Settings")
    initial_k = st.number_input(
        "Candidate chunks searched", min_value=20, max_value=2000, value=128, step=16,
        help="How many FAISS hits to pull before optional re-ranking. Higher = more recall, more RAM/CPU."
    )
    final_k = st.number_input(
        "Evidence chunks kept", min_value=5, max_value=120, value=50, step=5,
        help="Chunks sent to the model as evidence. Balance recall vs cost/latency."
    )
    per_video_cap = st.number_input(
        "Max chunks per video", min_value=1, max_value=30, value=6, step=1,
        help="Cap per source to avoid one video dominating the evidence."
    )
    use_mmr = st.toggle("Use MMR re-rank (safer pool cap 150)", value=True,
                        help="Diversifies evidence. Pool hard-capped at 150 to protect memory.")
    lambda_div = st.slider("Diversity (MMR λ)", 0.0, 1.0, 0.4, 0.05,
                           help="0=more novelty, 1=more query similarity. 0.3–0.5 works well.")
    st.subheader("Model")
    model_choice = st.selectbox("OpenAI model", ["gpt-4o-mini","gpt-4o"], index=0)

    st.divider()
    st.subheader("Index status (DATA_DIR)")
    st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    st.checkbox("chunks.jsonl", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("faiss.index", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("metas.pkl", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("video_meta.json", value=VIDEO_META_JSON.exists(), disabled=True)

    texts_cache, metas_cache = load_chunks_aligned()
    st.markdown(f"**Chunks indexed:** {len(texts_cache):,}")

    vm = load_video_meta()
    chan_to_vids: Dict[str,set] = {}
    for m in metas_cache:
        vid=m.get("video_id"); 
        if not vid: continue
        ch=(vm.get(vid,{}).get("channel") or m.get("channel") or m.get("uploader") or "").strip() or "Unknown"
        chan_to_vids.setdefault(ch,set()).add(vid)
    if chan_to_vids:
        st.subheader("Primary sources (videos per channel)")
        top=sorted(((c,len(v)) for c,v in chan_to_vids.items()), key=lambda x:x[1], reverse=True)[:6]
        for ch, n in top:
            q=ch.replace(" ","+"); url=f"https://www.youtube.com/results?search_query={q}"
            st.markdown(f"- [{ch}]({url}) — {n:,} videos")

    st.divider()
    with st.expander("Health checks", expanded=False):
        try:
            ix = faiss.read_index(str(INDEX_PATH)) if INDEX_PATH.exists() else None
            metas_len = len(pickle.load(open(METAS_PKL,"rb"))["metas"]) if METAS_PKL.exists() else 0
            lines = sum(1 for _ in open(CHUNKS_PATH,encoding="utf-8")) if CHUNKS_PATH.exists() else 0
            st.code(f"ntotal={getattr(ix,'ntotal',0)} metas={metas_len} chunks_lines={lines}", language="bash")
        except Exception as e:
            st.error(f"Alignment check failed: {e}")
        st.code(f"OpenAI ping: {openai_ping(model_choice)}", language="bash")

if "messages" not in st.session_state: st.session_state.messages=[]
for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

prompt = st.chat_input("Ask about sleep, protein timing, fasting, supplements, protocols…")
if prompt is None: st.stop()
st.session_state.messages.append({"role":"user","content":prompt})
with st.chat_message("user"): st.markdown(prompt)

missing=[p.relative_to(DATA_ROOT) for p in REQUIRED if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error("Required files missing:\n"+"\n".join(f"- {DATA_ROOT/p}" for p in missing))
    st.stop()

try:
    index, metas_from_pkl, payload = load_index_and_model()
    texts, metas = load_chunks_aligned()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load FAISS/chunks."); st.exception(e); st.stop()

if index is None or payload is None or not texts:
    with st.chat_message("assistant"): st.error("Index/metas/chunks not found."); 
    st.stop()

if int(index.ntotal)!=len(texts):
    with st.chat_message("assistant"):
        st.error(f"Index mismatch. faiss={int(index.ntotal):,} vs chunks={len(texts):,}. Rebuild index.")
    st.stop()
if metas_from_pkl is not None and len(metas_from_pkl)!=len(texts):
    with st.chat_message("assistant"):
        st.error("metas.pkl length mismatch. Rebuild all artifacts together.")
    st.stop()

embedder: SentenceTransformer = payload["embedder"]

from contextlib import nullcontext
with st.spinner("Searching sources…"):
    try:
        hits = search_chunks(prompt, index, embedder, metas, texts,
                             initial_k=int(initial_k), final_k=int(final_k),
                             per_video_cap=int(per_video_cap),
                             use_mmr=bool(use_mmr), lambda_div=float(lambda_div))
    except Exception as e:
        with st.chat_message("assistant"): st.error("Search failed."); st.exception(e); st.stop()

with st.chat_message("assistant"):
    if not hits:
        st.warning("No sources found. Try broadening the query.")
        st.session_state.messages.append({"role":"assistant","content":"No matching excerpts were found."})
        st.stop()
    with st.expander("Top matches (raw excerpts)", expanded=False):
        for i,h in enumerate(hits[:3],1):
            lbl,url=label_and_url(h["meta"])
            st.markdown(f"**{i}. {lbl}** — score {h['score']:.3f}")
            st.caption(h["text"])
    with st.spinner("Synthesizing answer…"):
        answer = openai_answer(st.session_state.get("model_choice", model_choice), prompt, st.session_state.messages, hits)
    st.markdown(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})
    with st.expander("Sources & timestamps", expanded=False):
        for i,h in enumerate(hits,1):
            lbl,url=label_and_url(h["meta"])
            st.markdown(f"{i}. [{lbl}]({url})" if url else f"{i}. {lbl}")
        st.caption("Quotes map to [n] citations.")
