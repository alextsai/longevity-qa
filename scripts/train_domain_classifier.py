import argparse, json, csv, re, joblib, yaml
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

def norm(s): 
    return re.sub(r"\s+"," ",(s or "")).strip()

def load_labels(p):
    ymap={}
    with open(p, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            vid=row.get("video_id") or row.get("id")
            lab=row.get("label")
            if vid and lab and lab.strip():
                ymap[vid]=lab.strip().lower()
    return ymap

def load_meta(p):
    if not Path(p).exists(): return {}
    return json.loads(Path(p).read_text(encoding="utf-8"))

def load_texts(chunks_path, video_meta):
    by_vid = defaultdict(list)
    cp = Path(chunks_path)
    if cp.exists():
        with cp.open(encoding="utf-8") as f:
            for ln in f:
                try:j=json.loads(ln)
                except:continue
                m=j.get("meta") or {}
                vid=(m.get("video_id") or m.get("vid") or m.get("ytid") or
                     j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
                if not vid: continue
                t=norm(j.get("text",""))
                if t: by_vid[vid].append(t)
    X={}  # vid -> text
    for vid, lines in by_vid.items():
        X[vid]=norm(" ".join(lines))
    for vid, info in video_meta.items():
        if vid not in X:
            title=norm(info.get("title",""))
            desc=norm(info.get("description","") or info.get("desc",""))
            X[vid]=norm(f"{title}. {desc}")
    return X

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--video-meta", required=True)
    ap.add_argument("--labels", required=True, help="CSV with columns: video_id,label")
    ap.add_argument("--out-dir", required=True)
    args=ap.parse_args()

    out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ymap = load_labels(args.labels)
    vmeta = load_meta(args.video_meta)
    texts = load_texts(args.chunks, vmeta)

    vids=[]; X_text=[]; y=[]
    for vid, lab in ymap.items():
        txt=texts.get(vid,"")
        if not txt: continue
        vids.append(vid); X_text.append(txt); y.append(lab)

    if not X_text:
        raise SystemExit("No training data matched your labels.")

    # small-N safe TF-IDF settings
    docN = len(X_text)
    min_df = 1 if docN < 50 else 2
    max_df = 1.0 if docN < 50 else 0.9

    vec=TfidfVectorizer(ngram_range=(1,2), min_df=min_df, max_df=max_df, max_features=120000)
    clf=LogisticRegression(max_iter=2000, class_weight="balanced")
    pipe=Pipeline([("tfidf", vec), ("clf", clf)])
    pipe.fit(X_text, y)

    # Save model
    joblib.dump(pipe, out/"domain_model.joblib")

    # Save calibrated probs with pure-Python types
    proba = pipe.predict_proba(X_text)
    classes = [str(c) for c in list(pipe.classes_)]
    rows = []
    for vid, probs in zip(vids, proba):
        row = {"video_id": str(vid)}
        for c, p in zip(classes, probs.tolist()):
            row[c] = float(p)
        rows.append(row)

    (out/"domain_probs.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out/"domain_probs.yaml").write_text(
        yaml.safe_dump(rows, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    print(classification_report(y, pipe.predict(X_text)))
    print(f"Saved: {out/'domain_model.joblib'}, {out/'domain_probs.json'}, {out/'domain_probs.yaml'}")

if __name__=="__main__":
    main()
