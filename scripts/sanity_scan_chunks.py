# scripts/sanity_scan_chunks.py
import json, re, argparse
from pathlib import Path

SPAM = re.compile(r"(sponsor|discount|promo code|subscribe|patreon|newsletter|athletic greens|wealthfront)", re.I)

def main(a):
    inp = Path(a.chunks)
    out_report = Path(a.out_report)
    out_clean = Path(a.out_clean) if a.out_clean else None

    total=0; bad=0; fixed=0
    seen_missing_vid=0; short=0; spam=0
    if out_clean: fout = out_clean.open("w", encoding="utf-8")
    else: fout=None

    with inp.open(encoding="utf-8") as f:
        for ln in f:
            total+=1
            try: j=json.loads(ln)
            except:
                bad+=1; continue
            t=(j.get("text") or "").strip()
            m=(j.get("meta") or {})
            vid=(m.get("video_id") or m.get("vid") or m.get("ytid") or
                 j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
            if not vid:
                seen_missing_vid+=1; continue
            if len(t)<24:
                short+=1; continue
            if SPAM.search(t):
                spam+=1; continue
            if fout:
                m["video_id"]=vid; j["meta"]=m
                fout.write(json.dumps(j, ensure_ascii=False)+"\n")
                fixed+=1

    rep={"total":total,"bad_json":bad,"missing_video_id":seen_missing_vid,"too_short":short,"spam_like":spam,"kept":fixed}
    out_report.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if fout: fout.close()

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="/var/data/data/chunks/chunks.jsonl")
    ap.add_argument("--out_report", required=True, help="/var/data/data/chunks/chunks_sanity_report.json")
    ap.add_argument("--out_clean", help="/var/data/data/chunks/chunks.cleaned.jsonl")
    main(ap.parse_args())
