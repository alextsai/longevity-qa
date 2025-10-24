# scripts/build_video_centroids_from_faiss.py
import os, pickle, numpy as np
from pathlib import Path
import faiss
DATA = Path(os.getenv("DATA_DIR", "/var/data")) / "data"
ix = faiss.read_index(str(DATA/"index/faiss.index"))
mp = pickle.load(open(DATA/"index/metas.pkl","rb"))
metas = mp["metas"]
# map row -> video_id
vids=[]
for m in metas:
    vid=None
    if isinstance(m, dict):
        vid = m.get("video_id") or m.get("vid") or m.get("ytid") or m.get("id")
    vids.append(str(vid) if vid else "")
n = ix.ntotal
assert n == len(vids), f"rows mismatch faiss={n} metas={len(vids)}"
# try fast path
X=None
try:
    xb = ix.get_xb()
    if xb is not None:
        arr = faiss.vector_float_to_array(xb)
        X = arr.reshape(n, ix.d)
except Exception:
    pass
# fallback
if X is None:
    X = np.empty((n, ix.d), dtype="float32")
    for i in range(n):
        X[i] = ix.reconstruct(i)
from collections import defaultdict
sum_vec=defaultdict(lambda: np.zeros(ix.d, "float32"))
count  =defaultdict(int)
for i, vid in enumerate(vids):
    if not vid: continue
    sum_vec[vid]+=X[i]; count[vid]+=1
video_ids=[]; centroids=[]
for vid,c in count.items():
    video_ids.append(vid)
    centroids.append(sum_vec[vid]/c)
centroids = np.vstack(centroids).astype("float32")
(DATA/"index").mkdir(parents=True, exist_ok=True)
with open(DATA/"index/video_ids.txt","w") as f:
    f.write("\n".join(video_ids)+"\n")
np.save(DATA/"index/video_centroids.npy", centroids)
print("OK videos:", len(video_ids), "dim:", ix.d)
