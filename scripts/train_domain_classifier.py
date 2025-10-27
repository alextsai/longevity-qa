# scripts/train_domain_classifier.py
# Train a simple SVM domain classifier on question text; persists model + thresholds.

import json, re, yaml, argparse
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sentence_transformers import SentenceTransformer

DOMAINS = ["cardio","sleep","nutrition","immune"]

def norm(s:str)->str:
    return re.sub(r"\s+"," ", (s or "").strip().lower())

def load_dataset(p:Path):
    """
    Expects JSONL with fields: {"text": "...", "label": "cardio|sleep|nutrition|immune"}
    Minimal starter: you can bootstrap from your own Q history.
    """
    X,Y=[],[]
    for ln in p.read_text(encoding="utf-8").splitlines():
        if not ln.strip(): continue
        j=json.loads(ln)
        t=norm(j.get("text","")); y=norm(j.get("label",""))
        if t and y in DOMAINS:
            X.append(t); Y.append(y)
    return X,Y

def main(args):
    data_p = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtxt, Ylab = load_dataset(data_p)
    if not Xtxt: raise SystemExit("no training data")

    enc = SentenceTransformer("intfloat/e5-large-v2")
    Xemb = enc.encode(Xtxt, normalize_embeddings=True, batch_size=64).astype("float32")

    y_map = {d:i for i,d in enumerate(DOMAINS)}
    y = np.array([y_map[l] for l in Ylab], dtype=np.int64)

    Xtr, Xte, ytr, yte = train_test_split(Xemb, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)

    clf = SVC(C=4.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    clf.fit(Xtr_s, ytr)

    ypro = clf.predict_proba(Xte_s)
    yhat = ypro.argmax(1)
    rep = classification_report(yte, yhat, target_names=DOMAINS, digits=3)
    print(rep)

    # Per-class OOD threshold = 25th percentile of max-prob when predicted correctly
    maxp = ypro.max(1)
    correct = (yhat==yte)
    thr = {}
    for i,d in enumerate(DOMAINS):
        vals = maxp[(yhat==i)&correct]
        thr[d] = float(np.percentile(vals, 25)) if len(vals)>5 else 0.40

    # Persist
    joblib.dump(clf, out_dir/"domain_model.joblib")
    joblib.dump(scaler, out_dir/"scaler.joblib")
    (out_dir/"label_map.json").write_text(json.dumps({str(i):d for i,d in enumerate(DOMAINS)}, indent=2))
    (out_dir/"ood_threshold.json").write_text(json.dumps(thr, indent=2))

    # Domain priors for transparency
    pri = {d: float((y==i).mean()) for i,d in enumerate(DOMAINS)}
    (out_dir/"domain_probs.yaml").write_text(yaml.safe_dump({"priors":pri, "eval_report":rep}, sort_keys=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to domain_labeled.jsonl")
    ap.add_argument("--out",  required=True, help="output dir, e.g., /var/data/data/models/domain")
    main(ap.parse_args())
