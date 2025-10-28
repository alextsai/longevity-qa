import argparse, json, re
from pathlib import Path
from collections import defaultdict

def norm(s): return re.sub(r"\s+"," ",(s or "")).strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--terms", nargs="+", default=["apob","ldl","statin","sleep","magnesium","melatonin","pcsk9"])
    args=ap.parse_args()
    pat=re.compile(r"("+"|".join(map(re.escape,args.terms))+r")", re.I)
    idx=defaultdict(set)
    with open(args.chunks, encoding="utf-8") as f:
        for ln in f:
            try:j=json.loads(ln)
            except:continue
            m=j.get("meta") or {}
            vid=(m.get("video_id") or m.get("vid") or m.get("ytid") or
                 j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
            t=norm(j.get("text",""))
            if not vid or not t: continue
            if pat.search(t):
                for term in args.terms:
                    if re.search(rf"\b{re.escape(term)}\b", t, re.I):
                        idx[term].add(vid)
    out={k: sorted(list(v)) for k,v in idx.items()}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved {args.out}")
if __name__=="__main__":
    main()
