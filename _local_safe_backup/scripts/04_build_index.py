#!/usr/bin/env python3
import os, json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_FP = "data/chunks/chunks.jsonl"
OUT_DIR   = Path("data/index")
INDEX_FP  = OUT_DIR / "faiss.index"
METAS_FP  = OUT_DIR / "metas.jsonl"

def load_chunks():
    texts, metas = [], []
    with open(CHUNKS_FP, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            # support both old {"text","meta":{video_id,start}} and new {"text","video_id","start"}
            if "meta" in o and "text" in o:
                t = o.get("text","")
                m = o.get("meta") or {}
                vid = m.get("video_id") or o.get("video_id") or ""
                st  = float(m.get("start", 0.0))
                m = {"video_id": vid, "start": st}
            else:
                t = o.get("text","")
                m = {"video_id": o.get("video_id",""), "start": float(o.get("start",0.0))}
            if not t.strip():
                continue
            texts.append(t)
            metas.append(m)
    return texts, metas

def embed_texts(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", batch=1024):
    model = SentenceTransformer(model_name)
    vecs=[]
    for i in tqdm(range(0,len(texts),batch), desc="embed"):
        chunk = texts[i:i+batch]
        E = model.encode(chunk, convert_to_numpy=True, batch_size=batch, show_progress_bar=False)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        E = E / np.clip(norms, 1e-12, None)
        vecs.append(E.astype("float32"))
    if vecs:
        return np.vstack(vecs)
    else:
        return np.zeros((0, model.get_sentence_embedding_dimension()), dtype="float32")

def build_faiss(X, out_fp: Path):
    d = X.shape[1] if X.size else 384
    index = faiss.IndexFlatIP(d)
    if X.shape[0] > 0:
        index.add(X)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_fp))

def write_metas(metas, out_fp: Path):
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fp, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def main():
    if not os.path.exists(CHUNKS_FP):
        raise SystemExit(f"Missing {CHUNKS_FP}")
    texts, metas = load_chunks()
    print(f"[embed] total={len(texts)}")
    X = embed_texts(texts)
    print(f"[embed] vectors={X.shape[0]} dim={X.shape[1] if X.size else 0}")
    build_faiss(X, INDEX_FP)
    write_metas(metas, METAS_FP)
    print(f"[write] index -> {INDEX_FP}")
    print(f"[write] metas -> {METAS_FP}")

if __name__ == "__main__":
    main()
