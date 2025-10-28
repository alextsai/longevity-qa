#!/usr/bin/env python3
"""
Quick scan: missing video_id, creator mapping failures, empty lines.
Prints counts and a small sample for inspection.
"""
import json, re, sys
from pathlib import Path
from collections import Counter

def main(p="data/chunks/chunks.jsonl"):
    p=Path(p)
    if not p.exists():
        print("chunks.jsonl missing"); sys.exit(1)
    n_total=n_empty=n_no_vid=0
    creators=Counter()
    with p.open(encoding="utf-8") as f:
        for ln in f:
            n_total+=1
            try:j=json.loads(ln)
            except: n_empty+=1; continue
            t=(j.get("text") or "").strip()
            if not t: n_empty+=1
            m=j.get("meta") or {}
            vid=(m.get("video_id") or m.get("vid") or m.get("ytid") or
                 j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id"))
            if not vid: n_no_vid+=1
            c=(m.get("channel") or m.get("author") or m.get("uploader") or "").strip().lower()
            if c: creators[c]+=1
    print({"lines":n_total,"empty":n_empty,"no_video_id":n_no_vid,"unique_creators":len(creators)})

if __name__=="__main__":
    main()
