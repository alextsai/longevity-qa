import json, numpy as np
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FP = ROOT/"data/chunks/chunks.jsonl"
INDEX_FP  = ROOT/"data/index/faiss.index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_chunks():
    chunks=[]
    with open(CHUNKS_FP,"r",encoding="utf-8") as f:
        for line in f:
            j=json.loads(line)
            if j.get("video_id") and j.get("text"):
                chunks.append(j)
    return chunks

def main():
    assert CHUNKS_FP.exists(), "chunks.jsonl missing. Run scripts/03_chunk_vtt.py first."
    INDEX_FP.parent.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print("chunks:", len(texts))
    model = SentenceTransformer(EMBED_MODEL)
    X = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    X = np.asarray(X, dtype="float32")
    print("embeddings:", X.shape)
    idx = faiss.IndexFlatIP(X.shape[1])
    idx.add(X)
    faiss.write_index(idx, str(INDEX_FP))
    print("wrote", INDEX_FP)

if __name__ == "__main__":
    main()
