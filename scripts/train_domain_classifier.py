#!/usr/bin/env python3
"""
Train a simple TF-IDF + LogisticRegression domain classifier from labels.csv.
Produces:
- data/domain/domain_model.joblib
- data/domain/scaler.joblib  (dummy for API symmetry)
- data/domain/domain_probs.yaml (per-video class probabilities for inspection)

labels.csv columns: video_id,label
label ∈ {cardio, sleep, nutrition, supplements, metabolic, …}
"""
import argparse, json, re, yaml, joblib
from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def norm(s:str)->str:
    return re.sub(r"\s+"," ", (s or "").strip())

def load_texts(chunks_path:Path, video_meta:Dict[str,Dict], vids:List[str])->Dict[str,str]:
    """
    Aggregate transcript text per video. If chunks.jsonl missing, fall back to titles.
    """
    out={v:"" for v in vids}
    if chunks_path.exists():
        with chunks_path.open(encoding="utf-8") as f:
            for ln in f:
                try:j=json.loads(ln)
                except: continue
                m=j.get("meta") or {}
                vid=m.get("video_id") or m.get("vid") or m.get("ytid") or j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id")
                if vid in out:
                    out[vid] += " " + norm(j.get("text",""))
    # title fallback
    for vid in vids:
        if not out[vid]:
            out[vid]=norm((video_meta.get(vid,{}) or {}).get("title",""))
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--video-meta", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out-dir", required=True)
    args=ap.parse_args()

    chunks=Path(args.chunks)
    video_meta=Path(args.video_meta)
    labels_path=Path(args.labels)
    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    vm = json.loads(video_meta.read_text(encoding="utf-8"))
    rows=[ln.strip().split(",") for ln in labels_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    header = rows[0]
    if [h.strip().lower() for h in header] != ["video_id","label"]:
        rows = [["video_id","label"]] + rows  # best-effort tolerate missing header
    data = rows[1:]

    vids=[r[0] for r in data]; y=[r[1] for r in data]
    texts = load_texts(chunks, vm, vids)
    X=[texts[v] for v in vids]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=200000, ngram_range=(1,2), min_df=2)),
        ("clf", LogisticRegression(max_iter=200, n_jobs=-1, class_weight="balanced"))
    ])
    pipe.fit(X, y)

    joblib.dump(pipe, out/"domain_model.joblib")
    joblib.dump(StandardScaler(with_mean=False, with_std=False), out/"scaler.joblib")

    # export per-video probs for audit
    proba = pipe.predict_proba(X)
    classes=list(pipe.classes_)
    out_rows=[]
    for vid, probs in zip(vids, proba):
        out_rows.append({"video_id": vid, **{c: float(p) for c,p in zip(classes, probs)}})
    (out/"domain_probs.yaml").write_text(yaml.safe_dump(out_rows, allow_unicode=True), encoding="utf-8")

    yhat=pipe.predict(X)
    print(classification_report(y, yhat))
    print(f"Saved: {out/'domain_model.joblib'}, {out/'scaler.joblib'}, {out/'domain_probs.yaml'}")

if __name__=="__main__":
    main()
