# app/app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

# ---------- Low-memory + offline-friendly runtime knobs (set before imports) ----------
import os, warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")         # CPU only
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
DATA_ROOT_ENV = os.getenv("DATA_DIR", "/var/data")
# local cache dirs (works even if HF is rate-limited)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", f"{DATA_ROOT_ENV}/models")
os.environ.setdefault("HF_HOME", f"{DATA_ROOT_ENV}/models")

# ---------- Standard imports ----------
from pathlib import Path
import sys, json, pickle
from typing import List, Tuple, Dict, Any
from contextlib import nullcontext

import streamlit as st
import numpy as np
import faiss

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    torch = None

from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ---------- Paths / helpers ----------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_ROOT = Path(DATA_ROOT_ENV).resolve()

# local helper for nice labels/URLs
from utils.labels import label_and_url

CHUNKS_PATH     = DATA_ROOT / "data/chunks/chunks.jsonl"
INDEX_PATH      = DATA_ROOT / "data/index/faiss.index"
METAS_PKL       = DATA_ROOT / "data/index/metas.pkl"
VIDEO_META_JSON = DATA_ROOT / "data/catalog/video_meta.json"
REQUIRED = [INDEX_PATH, METAS_PKL, CHUNKS_PATH, VIDEO_META_JSON]

def _parse_ts(v) -> float:
    if isinstance(v, (int, float)):
        try: return float(v)
        except: return 0.0
    if isinstance(v, str):
        try:
            sec = 0.0
            for p in v.split(":"):
                sec = sec * 60 + float(p)
            return sec
        except: return 0.0
    return 0.0

# ---------- Cached loaders ----------
@st.cache_resource(show_spinner=False)
def load_video_meta() -> Dict[str, Dict[str, str]]:
    if not VIDEO_META_JSON.exists(): return {}
    try: return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
    except Exception: return {}

@st.cache_resource(show_spinner=False)
def load_chunks_aligned() -> Tuple[List[str], List[dict]]:
    texts, metas = [], []
    if not CHUNKS_PATH.exists(): return texts, metas
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            try: j = json.loads(line)
            except Exception: continue
            t = (j.get("text") or "").strip()
            m = (j.get("meta") or {}).copy()
            vid = (
                m.get("video_id") or m.get("vid") or m.get("ytid") or
                j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id")
            )
            if vid: m["video_id"] = vid
            if "start" not in m and "start_sec" in m: m["start"] = m["start_sec"]
            m["start"] = _parse_ts(m.get("start", 0))
            if t: texts.append(t); metas.append(m)
    return texts, metas

@st.cache_resource(show_spinner=False)
def load_index_and_model():
    """
    Loads FAISS, metas, and the embedder.
    Embedder first tries local path /var/data/models/all-MiniLM-L6-v2 to avoid HF 429s.
    """
    if not INDEX_PATH.exists() or not METAS_PKL.exists():
        return None, None, None

    index = faiss.read_index(str(INDEX_PATH))
    with METAS_PKL.open("rb") as f:
        payload = pickle.load(f)
    model_name = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")

    # Prefer local model cache; fallback to HF repo id
    local_dir = Path(DATA_ROOT) / "models" / "all-MiniLM-L6-v2"
    model_path = str(local_dir) if local_dir.exists() else model_name
    embedder = SentenceTransformer(model_path, device="cpu")

    return index, payload.get("metas", []), {"model_name": model_name, "embedder": embedder}

# ---------- MMR (hard-capped) ----------
def mmr_rerank_lite(q_emb: np.ndarray,
                    cand_embs: np.ndarray,
                    hits: List[Dict[str, Any]],
                    final_k: int,
                    per_video_cap: int,
                    lam: float = 0.4) -> List[Dict[str, Any]]:
    pool = min(len(hits), max(int(final_k) * 2, 40), 80)  # <= 80 to cap RAM
    cand_embs = cand_embs[:pool]
    hits = hits[:pool]
    if pool == 0: return []

    sim_q = cand_embs @ q_emb
    sims  = cand_embs @ cand_embs.T

    selected, counts = [], {}
    seed = int(np.argmax(sim_q))
    v = hits[seed]["meta"].get("video_id") or "unknown"
    selected.append(seed); counts[v] = 1
    remaining = set(range(pool)); remaining.discard(seed)

    while len(selected) < final_k and remaining:
        best, bestscore = None, -1e9
        for i in list(remaining):
            vid = hits[i]["meta"].get("video_id") or "unknown"
            if counts.get(vid, 0) >= per_video_cap:
                remaining.discard(i); continue
            max_sim = float(np.max(sims[i, selected])) if selected else 0.0
            score = lam * sim_q[i] - (1.0 - lam) * max_sim
            if score > bestscore: bestscore, best = score, i
        if best is None: break
        selected.append(best); remaining.discard(best)
        vid = hits[best]["meta"].get("video_id") or "unknown"
        counts[vid] = counts.get(vid, 0) + 1

    return [hits[i] for i in selected]

# ---------- Retrieval ----------
def search_chunks(query: str,
                  index: faiss.Index,
                  embedder: SentenceTransformer,
                  metas: List[dict],
                  texts: List[str],
                  initial_k: int,
                  final_k: int,
                  per_video_cap: int,
                  use_mmr: bool,
                  lambda_diversity: float) -> List[Dict[str, Any]]:
    if not query.strip(): return []

    ctx = torch.no_grad() if torch is not None else nullcontext()
    with ctx:
        q = embedder.encode([query], normalize_embeddings=True).astype("float32")[0]

    K = max(1, min(int(initial_k), int(index.ntotal)))
    D, I = index.search(q.reshape(1, -1), K)
    idxs, scores = I[0].tolist(), D[0].tolist()

    raw: List[Dict[str, Any]] = []
    for i, s in zip(idxs, scores):
        if 0 <= i < len(texts):
            m = metas[i] if i < len(metas) else {}
            if not m.get("video_id"):
                vid = m.get("vid") or m.get("ytid")
                if vid: m["video_id"] = vid
            m["start"] = _parse_ts(m.get("start", 0))
            raw.append({"i": i, "score": float(s), "text": texts[i], "meta": m})

    if not raw: return []

    if not use_mmr:
        out, counts = [], {}
        for h in raw:
            vid = h["meta"].get("video_id") or "unknown"
            counts[vid] = counts.get(vid, 0) + 1
            if counts[vid] <= per_video_cap:
                out.append(h)
            if len(out) >= final_k: break
        return out

    pool = min(len(raw), max(int(final_k) * 2, 40), 80)
    cand_texts = [h["text"] for h in raw[:pool]]
    with (torch.no_grad() if torch is not None else nullcontext()):
        embs = embedder.encode(cand_texts, normalize_embeddings=True, batch_size=16).astype("float32")
    return mmr_rerank_lite(q, embs, raw[:pool], int(final_k), int(per_video_cap), float(lambda_diversity))

# ---------- OpenAI (compact + retries) ----------
def openai_answer(model_name: str, question: str, history, hits) -> str:
    key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    if not key:
        return "⚠️ OPENAI_API_KEY is not set."

    # keep ≤3 prior turns
    ctx_msgs = []
    for m in reversed(history[:-1]):
        if m.get("role") in ("user", "assistant") and m.get("content"):
            ctx_msgs.append(m)
        if len(ctx_msgs) >= 3: break
    ctx_msgs = list(reversed(ctx_msgs))

    ev = []
    for i, h in enumerate(hits, 1):
        lbl, _ = label_and_url(h["meta"])
        txt = (h["text"] or "")
        if len(txt) > 280: txt = txt[:280]
        ev.append(f"[{i}] {lbl}\n{txt}\n")

    system = (
        "Answer ONLY from the excerpts. Use short quoted phrases followed by [n] citations. "
        "If evidence is insufficient, say you don't know. Be concise."
    )
    convo = ("\n".join(f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in ctx_msgs) + "\n\n") if ctx_msgs else ""
    user_payload = (
        f"{('Recent conversation:\n' + convo) if convo else ''}"
        f"Question: {question}\n\nExcerpts:\n" + "\n".join(ev) +
        "\nWrite the best possible answer consistent with the excerpts. Quote fragments and cite like [n]."
    )

    client = OpenAI(timeout=30, max_retries=3)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=model_name,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_payload},
                ],
                timeout=30,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            err = str(e)
            if attempt < 2 and ("502" in err or "timeout" in err.lower() or "Bad gateway" in err):
                import time; time.sleep(1.5 * (attempt + 1)); continue
            return f"⚠️ OpenAI error: {err}"

def openai_ping(model_name: str) -> str:
    key = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None)
    if not key: return "NO_KEY"
    try:
        client = OpenAI(timeout=15, max_retries=2)
        r = client.chat.completions.create(model=model_name, temperature=0,
                                           messages=[{"role":"user","content":"Return 'pong'."}], timeout=15)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"ERR:{e}"

# ---------- UI ----------
st.set_page_config(page_title="Longevity / Nutrition Q&A", page_icon="🍎", layout="wide")
st.title("Longevity / Nutrition Q&A")

with st.sidebar:
    st.header("Settings")
    st.markdown("Tune retrieval and synthesis. Defaults are safe for small CPU instances.")

    initial_k = st.number_input(
        "Candidate chunks searched",
        min_value=20, max_value=2000, value=64, step=16,
        help="How many FAISS hits to pull before re-ranking. Higher = more recall, more RAM/CPU."
    )
    final_k = st.number_input(
        "Evidence chunks kept",
        min_value=5, max_value=120, value=30, step=5,
        help="Chunks sent to the model as evidence. Balance recall vs cost/latency."
    )
    per_video_cap = st.number_input(
        "Max chunks per video",
        min_value=1, max_value=30, value=5, step=1,
        help="Cap per source so one video cannot dominate the evidence."
    )
    use_mmr = st.toggle(
        "Use MMR re-rank (pool ≤ 80)",
        value=False,
        help="Diversifies evidence with a hard pool/batch cap to protect memory."
    )
    lambda_div = st.slider(
        "Diversity (MMR λ)",
        min_value=0.0, max_value=1.0, value=0.4, step=0.05,
        help="0 = more novelty, 1 = more query similarity. 0.3–0.5 works well."
    )

    st.subheader("Model")
    model_choice = st.selectbox("OpenAI model", ["gpt-4o-mini","gpt-4o"], index=0)
    st.session_state["model_choice"] = model_choice

    st.divider()
    st.subheader("Index status (DATA_DIR)")
    st.caption(f"DATA_DIR = `{DATA_ROOT}`")
    st.checkbox("data/chunks/chunks.jsonl", value=CHUNKS_PATH.exists(), disabled=True)
    st.checkbox("data/index/faiss.index", value=INDEX_PATH.exists(), disabled=True)
    st.checkbox("data/index/metas.pkl", value=METAS_PKL.exists(), disabled=True)
    st.checkbox("data/catalog/video_meta.json", value=VIDEO_META_JSON.exists(), disabled=True)

    texts_cache, metas_cache = load_chunks_aligned()
    st.markdown(f"**Chunks indexed:** {len(texts_cache):,}")

    # Primary sources: by videos per channel
    vm = load_video_meta()
    chan_to_vids: Dict[str, set] = {}
    for m in metas_cache:
        vid = m.get("video_id")
        if not vid: continue
        ch = (vm.get(vid, {}).get("channel") or m.get("channel") or m.get("uploader") or "").strip() or "Unknown"
        chan_to_vids.setdefault(ch, set()).add(vid)
    if chan_to_vids:
        st.subheader("Primary sources (videos per channel)")
        top = sorted(((ch, len(vs)) for ch, vs in chan_to_vids.items()), key=lambda x: x[1], reverse=True)[:6]
        for ch, vcount in top:
            q = ch.replace(" ", "+"); url = f"https://www.youtube.com/results?search_query={q}"
            st.markdown(f"- [{ch}]({url}) — {vcount:,} videos")

    st.divider()
    with st.expander("Health checks", expanded=False):
        try:
            ix = faiss.read_index(str(INDEX_PATH)) if INDEX_PATH.exists() else None
            metas_len = len(pickle.load(open(METAS_PKL,"rb"))["metas"]) if METAS_PKL.exists() else 0
            lines = sum(1 for _ in open(CHUNKS_PATH,encoding="utf-8")) if CHUNKS_PATH.exists() else 0
            st.code(f"ntotal={getattr(ix,'ntotal',0)}, metas={metas_len}, chunks_lines={lines}", language="bash")
        except Exception as e:
            st.error(f"Alignment check failed: {e}")
        st.code(f"OpenAI ping: {openai_ping(model_choice)}", language="bash")
        st.caption("Expect 'pong'. If ERR/NO_KEY appears, fix OPENAI_API_KEY/network or model choice.")

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about sleep, protein timing, fasting, supplements, protocols…")
if prompt is None:
    st.stop()

st.session_state.messages.append({"role": "user", "content": prompt})
with st.chat_message("user"):
    st.markdown(prompt)

# ---------- Required files guard ----------
missing = [p.relative_to(DATA_ROOT) for p in REQUIRED if not p.exists()]
if missing:
    with st.chat_message("assistant"):
        st.error("Required files missing under DATA_DIR:\n" + "\n".join(f"- {DATA_ROOT / p}" for p in missing))
    st.stop()

# ---------- Load artifacts ----------
try:
    index, metas_from_pkl, payload = load_index_and_model()
    texts, metas = load_chunks_aligned()
except Exception as e:
    with st.chat_message("assistant"):
        st.error("Failed to load FAISS/chunks.")
        st.exception(e)
    st.stop()

if index is None or payload is None or not texts:
    with st.chat_message("assistant"):
        st.error("Index/metas/chunks not found or empty.")
    st.stop()

# Alignment hard checks
if int(index.ntotal) != len(texts):
    with st.chat_message("assistant"):
        st.error(f"Index mismatch. FAISS ntotal={int(index.ntotal):,} vs chunks={len(texts):,}. Rebuild index.")
    st.stop()
if metas_from_pkl is not None and len(metas_from_pkl) != len(texts):
    with st.chat_message("assistant"):
        st.error("metas.pkl length mismatch. Rebuild artifacts together.")
    st.stop()

embedder: SentenceTransformer = payload["embedder"]

# ---------- Retrieval ----------
with st.spinner("Searching sources…"):
    try:
        hits = search_chunks(
            prompt,
            index=index,
            embedder=embedder,
            metas=metas,
            texts=texts,
            initial_k=int(initial_k),
            final_k=int(final_k),
            per_video_cap=int(per_video_cap),
            use_mmr=bool(use_mmr),
            lambda_diversity=float(lambda_div),
        )
    except Exception as e:
        with st.chat_message("assistant"):
            st.error("Search failed.")
            st.exception(e)
        st.stop()

# ---------- Answer ----------
with st.chat_message("assistant"):
    if not hits:
        st.warning("No sources found. Try broadening the query.")
        st.session_state.messages.append({"role": "assistant", "content": "No matching excerpts were found."})
        st.stop()

    with st.expander("Top matches (raw excerpts)", expanded=False):
        for i, h in enumerate(hits[:3], 1):
            lbl, url = label_and_url(h["meta"])
            st.markdown(f"**{i}. {lbl}** — score {h['score']:.3f}")
            st.caption(h["text"])

    with st.spinner("Synthesizing answer…"):
        answer = openai_answer(st.session_state.get("model_choice", model_choice), prompt, st.session_state.messages, hits)

    st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.expander("Sources & timestamps", expanded=False):
        for i, h in enumerate(hits, 1):
            lbl, url = label_and_url(h["meta"])
            st.markdown(f"{i}. [{lbl}]({url})" if url else f"{i}. {lbl}")
        st.caption("Quoted spans in the answer cite these sources as [n].")
