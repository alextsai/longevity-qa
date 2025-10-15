import pandas as pd
from yt_dlp import YoutubeDL
from pathlib import Path
import json, sys, time

channels_fp = Path("data/catalog/channels.csv")
out_fp = Path("data/catalog/videos.csv")
out_fp.parent.mkdir(parents=True, exist_ok=True)

channels = pd.read_csv(channels_fp)
rows = []

ydl_opts = {
    "extract_flat": True,       # do not resolve each video fully
    "skip_download": True,
    "dump_single_json": True
}

def list_entries(url):
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries", [])
    for e in entries:
        if not isinstance(e, dict):
            continue
        vid = e.get("id")
        title = e.get("title") or ""
        if vid is None:
            continue
        yield {
            "video_id": vid,
            "video_title": title,
            "url": f"https://www.youtube.com/watch?v={vid}"
        }

for _, row in channels.iterrows():
    url = row["url"]
    label = row.get("label","")
    print(f"[discover] Scanning: {label} -> {url}")
    try:
        for v in list_entries(url):
            v["channel_label"] = label
            rows.append(v)
    except Exception as ex:
        print(f"[discover][warn] {url}: {ex}")
        continue
    time.sleep(1)

df = pd.DataFrame(rows).drop_duplicates(subset=["video_id"])
df.to_csv(out_fp, index=False)
print(f"[discover] Wrote {out_fp} with {len(df)} rows")
