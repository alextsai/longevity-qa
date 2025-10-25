# scripts/precompute_video_summaries.py
# Build high-quality, extractive per-video bullets + centroids.
#
# Outputs:
#   /var/data/data/index/video_centroids.npy      (float32 [V, d], unit-norm)
#   /var/data/data/index/video_ids.txt            (one video_id per line)
#   /var/data/data/catalog/video_summaries.json   ({
#       vid:{
#         title, channel, published_at,
#         bullets:[{ts, text, chunk_idx, span}],
#         key_terms:[...],
#         metrics:{coverage, redundancy, n_bullets}
#       }
#   })
#
# Run:
#   DATA_DIR=/var/data python scripts/precompute_video_summaries.py

import os, json, math, pickle, re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_ROOT = Path(os.getenv("DATA_DIR", "/var/data")).resolve()
CHUNKS = DATA_ROOT / "data/chunks/chunks.jsonl"
VIDEO_META = DATA_ROOT / "data/catalog/video_meta.json"
OUT_CENT = DATA_ROOT / "data/index/video_centroids.npy"
OUT_IDS  = DATA_ROOT / "data/index/video_ids.txt"
OUT_SUM  = DATA_ROOT / "data/catalog/video_summaries.json"
METAS_PKL= DATA_ROOT / "data/index/metas.pkl"

SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')  # simple sentence split

def _parse_ts(v):
    if isinstance(v,(int,float)): return float(v)
    try:
        sec=0.0
        for p in str(v).split(":"):
            sec=sec*60+float(p)
        return sec
    except: return 0.0

def normalize(s): 
    return " ".join((s or "").split())

def sent_tokenize(text: str):
    text = normalize(text)
    if not text: return []
    parts = SENT_SPLIT.split(text)
    # strip empties, keep only reasonable sentences
    return [p.strip() for p in parts if len(p.strip()) >= 20]

def _jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    return len(a & b) / max(1, len(a | b))

def _unit_norm(X: np.ndarray) -> np.ndarray:
    X = X.astype("float32")
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def main():
    assert CHUNKS.exists(), f"missing {CHUNKS}"
    vm = {}
    if VIDEO_META.exists():
        try: vm = json.loads(VIDEO_META.read_text(encoding="utf-8"))
        except: vm = {}

    # pick model from metas.pkl to stay aligned with index
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    if METAS_PKL.exists():
        try:
            with METAS_PKL.open("rb") as f:
                payload=pickle.load(f)
            model_name = payload.get("model", model_name)
        except: pass

    # Prefer local cached model if present
    local_dir = DATA_ROOT / "models" / "all-MiniLM-L6-v2"
    try_name = str(local_dir) if (local_dir/"config.json").exists() else model_name
    enc = SentenceTransformer(try_name, device="cpu")

    # 1) load chunks grouped by video; also build raw corpus for DF
    texts_by_vid = defaultdict(list)       # per-video list of chunk texts
    metas_by_vid = defaultdict(list)       # per-video list of {"start": float}
    with CHUNKS.open(encoding="utf-8") as f:
        for ln in f:
            j=json.loads(ln)
            t = normalize(j.get("text",""))
            m = (j.get("meta") or {})
            vid = (
                m.get("video_id") or m.get("vid") or m.get("ytid") or
                j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id")
            )
            if not vid or not t:
                continue
            st = _parse_ts(m.get("start", m.get("start_sec", 0)))
            texts_by_vid[vid].append(t)
            metas_by_vid[vid].append({"start": st})

    vids = sorted(texts_by_vid.keys())
    if not vids:
        raise SystemExit("No videos found in chunks.jsonl")

    # 2) compute chunk embeddings and video centroids (mean of unit-norm chunks)
    centroids = []
    chunk_embeds_by_vid = {}
    for vid in vids:
        chunks = texts_by_vid[vid]
        X = enc.encode(chunks, normalize_embeddings=True, batch_size=128).astype("float32")
        chunk_embeds_by_vid[vid] = X
        c = X.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids.append(c)
    C = np.stack(centroids).astype("float32")
    OUT_CENT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_CENT, C)
    OUT_IDS.write_text("\n".join(vids), encoding="utf-8")

    # 3) sentence-level extractive bullets per video
    # Build DF over words (per-video, sentence-aware)
    DF = Counter()
    for vid in vids:
        seen = set()
        for chunk in texts_by_vid[vid]:
            for sent in sent_tokenize(chunk):
                for w in set(sent.lower().split()):
                    if w in seen: continue
                    DF[w] += 1
                    seen.add(w)
    N_videos = max(1, len(vids))

    def tfidf_sentence(sent: str) -> float:
        words = [w for w in sent.lower().split() if w]
        tf = Counter(words)
        val=0.0
        for w,cnt in tf.items():
            df = DF.get(w,1)
            idf = math.log((N_videos+1)/(df+0.5))
            val += cnt * idf
        return val / (len(words)+1e-6)

    summaries = {}
    for v_idx, vid in enumerate(vids):
        info = vm.get(vid,{})
        title = info.get("title") or ""
        channel = info.get("channel") or info.get("podcaster") or ""
        pub = info.get("published_at") or info.get("publishedAt") or info.get("date") or ""

        # sentence candidates with timestamps (approx by source chunk start)
        sentences = []
        chunk_list = texts_by_vid[vid]
        for i_chunk, chunk in enumerate(chunk_list):
            start_ts = metas_by_vid[vid][i_chunk]["start"]
            for sent in sent_tokenize(chunk):
                sentences.append({
                    "text": sent,
                    "ts": float(start_ts),
                    "chunk_idx": i_chunk
                })

        if not sentences:
            summaries[vid] = {
                "title": title, "channel": channel, "published_at": pub,
                "bullets": [], "key_terms": [], "metrics": {"coverage":0.0, "redundancy":0.0, "n_bullets":0}
            }
            continue

        # embed sentences for cosine-to-centroid and novelty
        S_txt = [s["text"] for s in sentences]
        S_emb = enc.encode(S_txt, normalize_embeddings=True, batch_size=128)
        c = C[v_idx].reshape(1, -1)
        cos = (S_emb @ c.T).ravel()

        # score sentences: TF-IDF + cosine + position prior (earlier, repeated ideas up)
        # position prior: earlier ts → slightly higher; transform to [0..1] with exp decay
        ts_all = np.array([s["ts"] for s in sentences], dtype="float32")
        if len(ts_all)>0:
            tmax = max(1.0, float(ts_all.max()))
            pos_prior = np.exp(-ts_all / (0.25*tmax))  # 0.25 video-length half-life
        else:
            pos_prior = np.ones(len(sentences), dtype="float32")

        tfidf_vals = np.array([tfidf_sentence(s["text"]) for s in sentences], dtype="float32")
        raw_score = 0.45*tfidf_vals + 0.45*cos + 0.10*pos_prior

        # MMR-style selection with diversity and time de-duplication
        keep = []
        keep_emb = []
        keep_times = []
        time_window = 15.0     # seconds
        sim_thresh = 0.85      # cosine
        jac_thresh = 0.60      # token jaccard

        order = list(np.argsort(-raw_score))
        for idx in order:
            s = sentences[idx]
            e = S_emb[idx]
            t = s["ts"]
            words = set(s["text"].lower().split())

            # time near-duplicate?
            duplicate = False
            for j, (ke, kt, ks) in enumerate(zip(keep_emb, keep_times, keep)):
                if abs(t-kt) <= time_window:
                    cos_sim = float(np.dot(e, ke))
                    if cos_sim >= sim_thresh:
                        duplicate = True; break
                    if _jaccard(words, set(ks["text"].lower().split())) >= jac_thresh:
                        duplicate = True; break
            if duplicate: 
                continue

            keep.append(s)
            keep_emb.append(e)
            keep_times.append(t)
            if len(keep) >= 10:   # cap bullets
                break

        # sort bullets by time
        keep_sorted = sorted(keep, key=lambda x: x["ts"])

        # metrics: redundancy (mean max sim among bullets), coverage (fraction tfidf mass kept)
        redundancy = 0.0
        if len(keep_emb) > 1:
            E = np.vstack(keep_emb)
            sims = E @ E.T
            np.fill_diagonal(sims, 0.0)
            redundancy = float(np.mean(np.max(sims, axis=1)))

        kept_idx = [S_txt.index(k["text"]) for k in keep_sorted]
        coverage = float(tfidf_vals[kept_idx].sum() / (tfidf_vals.sum()+1e-9))

        # key terms (top IDF contributors across kept bullets)
        term_counter = Counter()
        for b in keep_sorted:
            for w in set(b["text"].lower().split()):
                term_counter[w] += 1
        key_terms = [w for w,_ in term_counter.most_common(12)]

        # attach spans: locate sentence inside its chunk text
        bullets=[]
        for b in keep_sorted:
            ci = b["chunk_idx"]
            chunk_text = chunk_list[ci]
            text = b["text"]
            # find span; if multiple matches, take first
            start = chunk_text.find(text)
            span = [start, start+len(text)] if start >= 0 else [-1, -1]
            bullets.append({
                "ts": float(b["ts"]),
                "text": text,
                "chunk_idx": int(ci),
                "span": span
            })

        summaries[vid] = {
            "title": title,
            "channel": channel,
            "published_at": pub,
            "bullets": bullets,
            "key_terms": key_terms,
            "metrics": {
                "coverage": round(coverage,4),
                "redundancy": round(redundancy,4),
                "n_bullets": len(bullets)
            }
        }

    OUT_SUM.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUM.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {OUT_CENT}, {OUT_IDS}, {OUT_SUM}  (videos={len(vids)})")

if __name__ == "__main__":
    main()
