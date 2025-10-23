# scripts/precompute_video_summaries.py
# Build per-video centroids + concise summaries from chunks.jsonl
# Outputs:
#   /var/data/data/index/video_centroids.npy      (float32 [V, d])
#   /var/data/data/index/video_ids.txt            (one video_id per line)
#   /var/data/data/catalog/video_summaries.json   ({vid:{title,channel,date,summary,claims:[{ts,text}]}})
# Run:
#   DATA_DIR=/var/data python scripts/precompute_video_summaries.py

import os, json, math, pickle
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

def main():
    assert CHUNKS.exists(), f"missing {CHUNKS}"
    vm = {}
    if VIDEO_META.exists():
        try: vm = json.loads(VIDEO_META.read_text(encoding="utf-8"))
        except: vm = {}

    # pick model name from metas.pkl to stay aligned
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

    # 1) load chunks grouped by video
    texts_by_vid = defaultdict(list)
    metas_by_vid = defaultdict(list)
    with CHUNKS.open(encoding="utf-8") as f:
        for ln in f:
            j=json.loads(ln)
            t = normalize(j.get("text",""))
            m = (j.get("meta") or {})
            vid = m.get("video_id") or m.get("vid") or m.get("ytid") or j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id")
            if not vid or not t: 
                continue
            st = _parse_ts(m.get("start", m.get("start_sec", 0)))
            texts_by_vid[vid].append(t)
            metas_by_vid[vid].append({"start": st})

    vids = sorted(texts_by_vid.keys())
    if not vids:
        raise SystemExit("No videos found in chunks.jsonl")

    # 2) centroid per video (mean of normalized chunk embeddings)
    centroids = []
    for vid in vids:
        X = enc.encode(texts_by_vid[vid], normalize_embeddings=True, batch_size=128).astype("float32")
        c = X.mean(axis=0)
        # re-normalize centroid for cosine
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids.append(c)
    C = np.stack(centroids).astype("float32")
    np.save(OUT_CENT, C)
    OUT_IDS.write_text("\n".join(vids), encoding="utf-8")

    # 3) simple per-video summary + top claims (light TF-IDF on words)
    # build corpus DF
    DF = Counter()
    for vid in vids:
        seen = set()
        for t in texts_by_vid[vid]:
            toks = set(w.lower() for w in t.split())
            for w in toks:
                DF[w] += 1
    N = len(vids)

    def score_text(t):
        words = [w.lower() for w in t.split()]
        tf = Counter(words)
        val=0.0
        for w,cnt in tf.items():
            df = DF.get(w,1)
            idf = math.log((N+1)/(df+0.5))
            val += cnt * idf
        return val / (len(words)+1e-6)

    summaries = {}
    for vid in vids:
        info = vm.get(vid,{})
        # pick 6 high-scoring lines across the video as pseudo-summary
        lines = texts_by_vid[vid]
        scores = [(i, score_text(t)) for i,t in enumerate(lines)]
        top = [i for i,_ in sorted(scores, key=lambda x: -x[1])[:12]]  # 12 raw
        # keep diversity by ordering by position
        top_sorted = sorted(top[:10])  # keep <=10
        # build claims as (ts,text) of first 6
        claims=[]
        for i in top_sorted[:6]:
            claims.append({
                "ts": float(metas_by_vid[vid][i]["start"]),
                "text": lines[i][:280]+"…" if len(lines[i])>280 else lines[i]
            })
        # make a compact summary from first 6 top lines
        summary = " ".join(lines[i] for i in top_sorted[:6])
        if len(summary) > 1200:
            summary = summary[:1200]+"…"
        summaries[vid] = {
            "title": info.get("title") or "",
            "channel": info.get("channel") or "",
            "published_at": info.get("published_at") or info.get("publishedAt") or info.get("date") or "",
            "summary": summary,
            "claims": claims
        }

    OUT_SUM.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote:", OUT_CENT, OUT_IDS, OUT_SUM)

if __name__ == "__main__":
    main()
