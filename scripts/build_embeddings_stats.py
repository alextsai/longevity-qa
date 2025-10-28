#!/usr/bin/env python3
"""
Create normalized copy of video centroids and covariance for debugging.

Inputs:  data/index/video_centroids.npy
Outputs: data/domain/emb_centroids.npy  (unit-normalized)
         data/domain/emb_cov.npy
"""
import argparse, numpy as np
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--centroids", default="data/index/video_centroids.npy")
    ap.add_argument("--out-dir", default="data/domain")
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    C_path=Path(args.centroids)
    if not C_path.exists():
        print("centroids not found, skipping."); return
    C=np.load(C_path).astype("float32")
    n=np.linalg.norm(C,axis=1,keepdims=True)+1e-12
    Cn=C/n
    np.save(out/"emb_centroids.npy", Cn)
    cov = np.cov(Cn, rowvar=False)
    np.save(out/"emb_cov.npy", cov)
    print(f"Saved: {out/'emb_centroids.npy'}, {out/'emb_cov.npy'}")

if __name__=="__main__":
    main()
