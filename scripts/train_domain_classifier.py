import json, yaml, re, joblib, numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import FeatureUnion
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

DATA = Path(os.getenv("DATA_DIR","/var/data"))
SUM = json.loads((DATA/"data/catalog/video_summaries.json").read_text())
VM  = json.loads((DATA/"data/catalog/video_meta.json").read_text())

# 1) build texts
texts=[]; y=[]
for vid, s in SUM.items():
    txt = " ".join([
        VM.get(vid,{}).get("title",""),
        s.get("summary",""),
        " ".join([b.get("text","") for b in s.get("bullets",[])]),
    ])
    label = s.get("domain")  # start with curated labels; hand-correct file
    if not label: continue
    texts.append(txt); y.append(label)

# 2) features
wvec = TfidfVectorizer(ngram_range=(1,3), min_df=3, max_df=0.9, sublinear_tf=True, analyzer="word")
cvec = TfidfVectorizer(ngram_range=(3,5), min_df=3, analyzer="char_wb")
Xw = wvec.fit_transform(texts); Xc = cvec.fit_transform(texts)

emb = SentenceTransformer("intfloat/e5-large-v2")
Z  = emb.encode([f"passage: {t}" for t in texts], normalize_embeddings=True)
scl = StandardScaler(with_mean=False)
Zs = scl.fit_transform(Z)

from scipy.sparse import hstack
X = hstack([Xw, Xc, Zs])

clf = CalibratedClassifierCV(LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced"))
clf.fit(X, y)

# 3) save
joblib.dump({"w":wvec,"c":cvec,"clf":clf}, DATA/"data/catalog/domain_model.joblib")
joblib.dump(scl, DATA/"data/catalog/scaler.joblib")

# 4) export per-video probabilities
probs={}
for vid,s in SUM.items():
    txt = " ".join([VM.get(vid,{}).get("title",""), s.get("summary",""),
                    " ".join([b.get("text","") for b in s.get("bullets",[])])])
    Xw=wvec.transform([txt]); Xc=cvec.transform([txt]); Zs=scl.transform(emb.encode([f"passage: {txt}"], normalize_embeddings=True))
    Xi=hstack([Xw,Xc,Zs])
    p = dict(zip(clf.classes_, clf.predict_proba(Xi)[0].tolist()))
    probs[vid]={"probs":p}
Path(DATA/"data/catalog").mkdir(parents=True, exist_ok=True)
(DATA/"data/catalog/domain_probs.yaml").write_text(yaml.safe_dump(probs, sort_keys=False))

# 5) embedding distribution for OOD
Zall = emb.encode([f"passage: { ' '.join([VM.get(v,{}).get('title',''), SUM[v].get('summary','')]) }" for v in SUM.keys()], normalize_embeddings=True)
cent = Zall.mean(0); cov = np.cov(Zall.T)
np.save(DATA/"data/catalog/emb_centroids.npy", cent)
np.save(DATA/"data/catalog/emb_cov.npy", cov)
