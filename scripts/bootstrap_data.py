# scripts/bootstrap_data.py
import os, sys, tarfile, zipfile
from pathlib import Path
import shutil
import numpy as np

DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data")).resolve()
URL = os.getenv("DATA_DRIVE_URL", "").strip()

NEEDED = {
    "chunks.jsonl": DATA_DIR/"data/chunks/chunks.jsonl",
    "faiss.index":  DATA_DIR/"data/index/faiss.index",
    "metas.pkl":    DATA_DIR/"data/index/metas.pkl",
    "video_meta.json": DATA_DIR/"data/catalog/video_meta.json",
}

def log(*a): print("[bootstrap]", *a, flush=True)

def ensure_dirs():
    for sub in ["data/chunks","data/index","data/catalog","models"]:
        (DATA_DIR/sub).mkdir(parents=True, exist_ok=True)

def have_all():
    return all(p.exists() for p in NEEDED.values())

def extract_archives(root: Path):
    # .tgz/.tar.gz
    for tb in list(root.rglob("*.tar.gz")) + list(root.rglob("*.tgz")):
        try:
            log("extract", tb)
            with tarfile.open(tb, "r:gz") as tf: tf.extractall(root)
        except Exception as e:
            log("extract failed", tb, e)
    # .zip
    for z in root.rglob("*.zip"):
        try:
            log("unzip", z)
            with zipfile.ZipFile(z) as zf: zf.extractall(root)
        except Exception as e:
            log("unzip failed", z, e)

def try_gdown_folder():
    if not URL: 
        log("DATA_DRIVE_URL not set, skipping gdown")
        return
    try:
        import gdown
    except Exception:
        os.system("pip install -q gdown==5.1.0")
        import gdown
    log("gdown folder ->", URL)
    gdown.download_folder(URL, output=str(DATA_DIR), use_cookies=False, remaining_ok=True, quiet=True)
    extract_archives(DATA_DIR)

def try_copy_from_bundle():
    # fallback if you baked files under repo /bundle/
    root = Path(__file__).resolve().parents[1] / "bundle"
    if not root.exists(): 
        return
    log("bundle fallback at", root)
    def cp(rel, dst):
        src = root / rel
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst); log("copied", src, "->", dst)
    cp("chunks/chunks.jsonl", NEEDED["chunks.jsonl"])
    cp("index/faiss.index",   NEEDED["faiss.index"])
    cp("index/metas.pkl",     NEEDED["metas.pkl"])
    cp("catalog/video_meta.json", NEEDED["video_meta.json"])

def relocate_downloaded():
    # find first match for each needed filename anywhere under DATA_DIR
    for name, dst in NEEDED.items():
        if dst.exists(): 
            continue
        found = next((p for p in DATA_DIR.rglob(name) if p.is_file()), None)
        if found:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(found, dst)
            log("placed", found, "->", dst)

def build_offsets():
    dst = DATA_DIR/"data/chunks/chunks.offsets.npy"
    src = NEEDED["chunks.jsonl"]
    if dst.exists() or not src.exists(): 
        return
    pos=0; offs=[]
    with open(src, "rb") as f:
        for ln in f:
            offs.append(pos); pos += len(ln)
    np.save(dst, np.array(offs, dtype=np.int64))
    log("offsets built", dst)

def cache_model():
    dst = DATA_DIR/"models/all-MiniLM-L6-v2"
    if (dst/"config.json").exists(): 
        return
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        os.system("pip install -q huggingface_hub==0.24.6")
        from huggingface_hub import snapshot_download
    log("caching HF model to", dst)
    snapshot_download("sentence-transformers/all-MiniLM-L6-v2",
                      local_dir=str(dst),
                      local_dir_use_symlinks=False,
                      token=os.getenv("HF_TOKEN"))

if __name__ == "__main__":
    log("DATA_DIR =", DATA_DIR)
    ensure_dirs()
    if not have_all():
        try_gdown_folder()
        extract_archives(DATA_DIR)
        relocate_downloaded()
        try_copy_from_bundle()
    build_offsets()
    if have_all():
        cache_model()
        log("BOOTSTRAP OK")
    else:
        missing = [str(p) for p in NEEDED.values() if not p.exists()]
        log("MISSING after bootstrap:", *missing)
