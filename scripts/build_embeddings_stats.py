# scripts/build_embeddings_stats.py
import json, argparse, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

def iter_chunks(jsonl:Path, max_lines:int|None=None):
    with jsonl.open(encoding="utf-8") as f:
        for i,ln in enumerate(f):
            if max_lines and i>=max_lines: break
            try:
                j=json.loads(ln)
                t=(j.get("text") or "").strip()
                if t: yield t
            except: continue

def main(a):
    jsonl = Path(a.chunks)
    outd  = Path(a.out); outd.mkdir(parents=True, exist_ok=True)
    enc = SentenceTransformer("intfloat/e5-large-v2")

    buf=[]
    for t in iter_chunks(jsonl, max_lines=a.max_lines):
        buf.append(t)
        if len(buf)>=a.batch:
            E = enc.encode(buf, normalize_embeddings=True).astype("float32")
            yield_from(E, outd, flush=False)
            buf.clear()
    if buf:
        E = enc.encode(buf, normalize_embeddings=True).astype("float32")
        yield_from(E, outd, flush=True)

def yield_from(E, outd, flush):
    # Online mean/var accumulator stored in temp files
    s_mu_p = outd/"_mu_tmp.npy"; s_ss_p = outd/"_ss_tmp.npy"; s_n_p = outd/"_n_tmp.txt"
    if s_mu_p.exists():
        mu = np.load(s_mu_p); ss = np.load(s_ss_p); n = int(s_n_p.read_text())
    else:
        mu = np.zeros(E.shape[1], dtype="float64"); ss = np.zeros(E.shape[1], dtype="float64"); n=0
    mu_new = (mu*n + E.sum(0))/ (n + E.shape[0])
    # sum of squares update
    ss_new = ss + (E**2).sum(0)
    n_new  = n + E.shape[0]
    np.save(s_mu_p, mu_new); np.save(s_ss_p, ss_new); (outd/"_n_tmp.txt").write_text(str(n_new))
    if flush:
        var = (ss_new/n_new) - (mu_new**2)
        np.save(outd/"emb_centroids.npy", mu_new.astype("float32"))
        np.save(outd/"emb_cov.npy", np.maximum(var, 1e-6).astype("float32"))

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="/var/data/data/chunks/chunks.jsonl")
    ap.add_argument("--out", required=True, help="/var/data/data/index")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max_lines", type=int, default=None)
    main(ap.parse_args())
