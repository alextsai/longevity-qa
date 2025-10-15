#!/usr/bin/env python3
import json, subprocess
from pathlib import Path

CHUNKS = Path("data/chunks/chunks.jsonl")
META   = Path("data/catalog/video_meta.json")
META.parent.mkdir(parents=True, exist_ok=True)

def collect_needed_ids():
    vids=set()
    with open(CHUNKS,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d=json.loads(line)
            v=d.get("video_id")
            if v: vids.add(v)
    return vids

def fetch_meta(vid):
    try:
        j = subprocess.check_output(
            ["yt-dlp","-J","--ignore-errors", f"https://www.youtube.com/watch?v={vid}"],
            text=True
        )
        d=json.loads(j)
        return {
            "title": (d.get("title") or "Untitled").strip(),
            "channel": (d.get("uploader") or "Unknown channel").strip(),
        }
    except Exception:
        return {"title":"Untitled","channel":"Unknown channel"}

meta={}
if META.exists():
    meta=json.load(open(META,encoding="utf-8"))

seen = collect_needed_ids()
need=[v for v in seen if v not in meta or not meta[v].get("title")]
print(f"[meta] have={len(meta)}  chunks_videos={len(seen)}  to_fetch={len(need)}")
for vid in need:
    info = fetch_meta(vid)
    meta[vid]=info
    print("  +", vid, "->", info["channel"][:40], "|", info["title"][:70])

json.dump(meta, open(META,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
print("[meta] final entries:", len(meta))
