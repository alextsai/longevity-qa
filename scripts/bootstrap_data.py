# scripts/bootstrap_data.py
import os, sys, tarfile, json
from pathlib import Path
import numpy as np

# optional imports inside functions to keep boot light
DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data")).resolve()
URL = os.getenv("DATA_DRIVE_URL", "https://drive.google.com/drive/folders/1J521WtbU_tnArrD3W_Yh7Q6l3XCCKcjH?usp=drive_link")

def ensure_dirs():
    (DATA_DIR / "data/chunks").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "data/index").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "data/catalog").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "models").mkdir(parents=True, exist_ok=True)

def need_paths():
    return [
        DATA_DIR/"data/chunks/chunks.jsonl",
        DATA_DIR/"data/index/faiss.index",
        DATA_DIR/"data/index/metas.pkl",
        DATA_DIR/"data/catalog/video_meta.json",
    ]

def have_all_core():
    return all(p.exists() for p in need_paths())

def download_if_needed():
    if have_all_core():
        return
    try:
        import gdown  # type: ignore
    except Exception:
        os.system("pip -q install gdown==5.1.0")
        import gdown  # type: ignore
    ensure_dirs()
    gdown.download_folder(URL, output=str(DATA_DIR), use_cookies=False, remaining_ok=True, quiet=True)
    # extract any tarballs
    for tb in list(DATA_DIR.glob("*.tgz")) + list(DATA_DIR.glob("*.tar.gz")):
        try:
            with tarfile.open(tb, "r:gz") as tf:
                tf.extractall(DATA_DIR)
        except Exception:
            pass
    # move loose files to the expected layout
    mv = {
        "chunks.jsonl": DATA_DIR/"data/chunks/chunks.jsonl",
        "faiss.index":  DATA_DIR/"data/index/faiss.index",
        "metas.pkl":    DATA_DIR/"data/index/metas.pkl",
        "video_meta.json": DATA_DIR/"data/catalog/video_meta.json",
    }
    for name, dst in mv.items():
        src = DATA_DIR/name
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)

def build_offsets_if_missing():
    off = DATA_DIR/"data/chunks/chunks.offsets.npy"
    src = DATA_DIR/"data/chunks/chunks.jsonl"
    if off.exists() or not src.exists():
        return
    pos = 0; offs = []
    with open(src, "rb") as f:
        for ln in f:
            offs.append(pos); pos += len(ln)
    np.save(off, np.array(offs, dtype=np.int64))

def precompute_video_centroids_if_missing():
    """Compute per-video mean embedding without loading everything into RAM.
       Works by reconstructing vectors from the FAISS index one-by-one.
       Outputs:
         - data/index/video_centroids.npy  (float32 [Nvideos, d])
         - data/index/video_ids.txt        (text, one id per line)
         - data/catalog/video_summaries.json  (optional; fallback to titles)
    """
    out_cent = DATA_DIR/"data/index/video_centroids.npy"
    out_ids  = DATA_DIR/"data/index/video_ids.txt"
    out_sum  = DATA_DIR/"data/catalog/video_summaries.json"

    if out_cent.exists() and out_ids.exists():
        return

    # core artifacts
    from sentence_transformers import SentenceTransformer  # ensures model cache exists if needed later
    import faiss, pickle

    idx_path = DATA_DIR/"data/index/faiss.index"
    metas_p  = DATA_DIR/"data/index/metas.pkl"
    catalog_p= DATA_DIR/"data/catalog/video_meta.json"
    if not (idx_path.exists() and metas_p.exists()):
        return

    ix = faiss.read_index(str(idx_path))
    with open(metas_p, "rb") as f:
        mp = pickle.load(f)

    # map: chunk_id -> video_id
    # mp["metas"] is a list aligned with chunk ids
    chunk_to_vid = []
    for m in mp.get("metas", []):
        vid = m.get("video_id") or m.get("vid") or m.get("ytid") or m.get("id")
        chunk_to_vid.append(vid)

    # accumulate sums and counts per video, in a streaming way
    sums = {}     # vid -> np.float64 vector sum
    counts = {}   # vid -> int
    d = ix.d

    # faiss reconstruct is memory safe even when index is large
    for i, vid in enumerate(chunk_to_vid):
        if vid is None:  # skip malformed
            continue
        try:
            v = ix.reconstruct(int(i))
        except Exception:
            # not all FAISS types support reconstruct for deleted ids; skip
            continue
        if vid not in sums:
            sums[vid] = np.zeros(d, dtype=np.float64)
            counts[vid] = 0
        sums[vid] += v.astype(np.float64, copy=False)
        counts[vid] += 1

        # light throttle to avoid long blocking boot logs
        if (i % 200000) == 0 and i > 0:
            print(f"[precompute] processed {i} chunks...")

    if not sums:
        return

    # finalize means
    vids = sorted(sums.keys())
    mat = np.zeros((len(vids), d), dtype=np.float32)
    for row, vid in enumerate(vids):
        mat[row] = (sums[vid] / max(1, counts[vid])).astype(np.float32)

    out_cent.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_cent, mat)
    with open(out_ids, "w", encoding="utf-8") as f:
        for vid in vids:
            f.write(f"{vid}\n")

    # optional lightweight summaries: fall back to titles in catalog
    if catalog_p.exists():
        try:
            cat = json.loads(catalog_p.read_text(encoding="utf-8"))
            id2title = {str(x.get("video_id") or x.get("id")): (x.get("title") or "") for x in cat}
        except Exception:
            id2title = {}
    else:
        id2title = {}

    if not out_sum.exists():
        # store minimal structure {video_id: {"title": ...}}
        sums_payload = {vid: {"title": id2title.get(vid, "")} for vid in vids}
        out_sum.parent.mkdir(parents=True, exist_ok=True)
        out_sum.write_text(json.dumps(sums_payload, ensure_ascii=False), encoding="utf-8")

def main():
    print(f"[bootstrap] DATA_DIR = {DATA_DIR}")
    ensure_dirs()
    download_if_needed()
    build_offsets_if_missing()
    # The key enhancement: do the per-video precompute here if missing.
    try:
        precompute_video_centroids_if_missing()
    except Exception as e:
        # do not block app start
        print(f"[precompute] skipped due to error: {e}")

    print("[bootstrap] BOOTSTRAP OK")

if __name__ == "__main__":
    main()
