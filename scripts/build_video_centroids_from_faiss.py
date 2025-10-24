# scripts/build_video_centroids_from_faiss.py
import os, json, pickle, numpy as np
from pathlib import Path
import faiss
from collections import defaultdict

DATA = Path(os.getenv("DATA_DIR", "/var/data")) / "data"
INDEX = DATA / "index/faiss.index"
METAS = DATA / "index/metas.pkl"
CHUNKS = DATA / "chunks/chunks.jsonl"

def load_vids_from_metas(metas):
    vids = []
    for m in metas:
        vid = None
        if isinstance(m, dict):
            vid = m.get("video_id") or m.get("vid") or m.get("ytid") or m.get("id")
        vids.append(str(vid) if vid else "")
    return vids

def load_vids_from_chunks(n_rows):
    vids = []
    with open(CHUNKS, encoding="utf-8") as f:
        for i, ln in enumerate(f):
            if i >= n_rows: break
            try:
                j = json.loads(ln)
            except Exception:
                vids.append("")
                continue
            m = (j.get("meta") or {})
            vid = (m.get("video_id") or m.get("vid") or m.get("ytid") or
                   j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
            vids.append(str(vid) if vid else "")
    # pad if chunks shorter than index
    if len(vids) < n_rows:
        vids.extend([""] * (n_rows - len(vids)))
    return vids

def main():
    if not INDEX.exists():
        raise FileNotFoundError(f"missing {INDEX}")
    ix = faiss.read_index(str(INDEX))
    n, d = ix.ntotal, ix.d

    # try metas.pkl
    vids = []
    if METAS.exists():
        mp = pickle.load(open(METAS, "rb"))
        metas = mp.get("metas", [])
        if len(metas) == n:
            vids = load_vids_from_metas(metas)

    # fall back to chunks.jsonl if needed
    if (not vids) or all(v == "" for v in vids):
        if not CHUNKS.exists():
            raise FileNotFoundError("no video ids in metas and chunks.jsonl missing")
        vids = load_vids_from_chunks(n)

    # sanity: if *still* empty, bail with a clear message
    if all(v == "" for v in vids):
        raise RuntimeError("Could not derive any video_id. Check metas.pkl and chunks.jsonl formatting.")

    # get vectors
    X = None
    try:
        xb = ix.get_xb()
        if xb is not None:
            arr = faiss.vector_float_to_array(xb)
            X = arr.reshape(n, d)
    except Exception:
        pass
    if X is None:
        X = np.empty((n, d), dtype="float32")
        for i in range(n):
            X[i] = ix.reconstruct(i)

    # group by video and average
    sum_vec = defaultdict(lambda: np.zeros(d, "float32"))
    count   = defaultdict(int)
    for i, vid in enumerate(vids):
        if not vid:  # skip rows without id
            continue
        sum_vec[vid] += X[i]
        count[vid]   += 1

    if not count:
        raise RuntimeError("All rows lacked video_id. Nothing to aggregate.")

    video_ids = []
    centroids = []
    for vid, c in count.items():
        video_ids.append(vid)
        centroids.append(sum_vec[vid] / max(1, c))

    centroids = np.vstack(centroids).astype("float32")
    (DATA / "index").mkdir(parents=True, exist_ok=True)
    with open(DATA / "index/video_ids.txt", "w") as f:
        f.write("\n".join(video_ids) + "\n")
    np.save(DATA / "index/video_centroids.npy", centroids)

    print(f"OK videos: {len(video_ids)} dim: {d} saved in {DATA/'index'}")

if __name__ == "__main__":
    main()
