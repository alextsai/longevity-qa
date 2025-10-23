# scripts/bootstrap_data.py
import os, sys, tarfile
from pathlib import Path
import numpy as np

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data")).resolve()
URL = os.getenv("DATA_DRIVE_URL", "https://drive.google.com/drive/folders/1J521WtbU_tnArrD3W_Yh7Q6l3XCCKcjH?usp=drive_link")

def ensure_dirs():
    (DATA_DIR / "data/chunks").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "data/index").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "data/catalog").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "models").mkdir(parents=True, exist_ok=True)

def have_all():
    need = [
        DATA_DIR/"data/chunks/chunks.jsonl",
        DATA_DIR/"data/index/faiss.index",
        DATA_DIR/"data/index/metas.pkl",
        DATA_DIR/"data/catalog/video_meta.json",
    ]
    return all(p.exists() for p in need)

def download_if_needed():
    # only if any piece missing
    if have_all(): return
    try:
        import gdown
    except Exception:
        os.system("pip install -q gdown==5.1.0")
        import gdown
    ensure_dirs()
    # download folder or tarball into DATA_DIR
    gdown.download_folder(URL, output=str(DATA_DIR), use_cookies=False, remaining_ok=True, quiet=True)
    # extract tarballs if present
    for tb in list(DATA_DIR.glob("*.tgz")) + list(DATA_DIR.glob("*.tar.gz")):
        try:
            with tarfile.open(tb, "r:gz") as tf:
                tf.extractall(DATA_DIR)
        except Exception:
            pass
    # move loose files into expected layout
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

def build_offsets():
    off = DATA_DIR/"data/chunks/chunks.offsets.npy"
    src = DATA_DIR/"data/chunks/chunks.jsonl"
    if off.exists() or not src.exists(): return
    pos=0; offs=[]
    with open(src, "rb") as f:
        for ln in f:
            offs.append(pos); pos += len(ln)
    np.save(off, np.array(offs, dtype=np.int64))

def cache_model():
    # optional, avoids HF 429
    dst = DATA_DIR/"models/all-MiniLM-L6-v2"
    if (dst/"config.json").exists(): return
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        os.system("pip install -q huggingface_hub")
        from huggingface_hub import snapshot_download
    snapshot_download("sentence-transformers/all-MiniLM-L6-v2",
                      local_dir=str(dst),
                      local_dir_use_symlinks=False,
                      token=os.getenv("HF_TOKEN"))

if __name__ == "__main__":
    ensure_dirs()
    download_if_needed()
    build_offsets()
    cache_model()
    # print a tiny summary for logs
    print("BOOTSTRAP OK; DATA_DIR =", DATA_DIR)
