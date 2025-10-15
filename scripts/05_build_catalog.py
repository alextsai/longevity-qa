#!/usr/bin/env python3
import json, glob, re, subprocess, time, os, sys
from pathlib import Path

def find_ids():
    vtts = glob.glob("data/vtt/*.vtt") or glob.glob("data/raw_vtt/*.vtt")
    ids = []
    for p in vtts:
        vid = Path(p).name.split(".")[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{6,}", vid):
            ids.append(vid)
    return sorted(set(ids))

def yt_info(vid):
    url = f"https://www.youtube.com/watch?v={vid}"
    cmd = ["yt-dlp","--skip-download","--dump-json",url]
    b = os.getenv("YTDLP_COOKIES_BROWSER")
    prof = os.getenv("YTDLP_COOKIES_PROFILE")
    if b and prof:
        cmd = ["yt-dlp", "--cookies-from-browser", f"{b}:{prof}", "--skip-download", "--dump-json", url]
    elif b:
        cmd = ["yt-dlp", "--cookies-from-browser", b, "--skip-download", "--dump-json", url]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout.strip().splitlines()[-1]
        j = json.loads(out)
        title = (j.get("title") or "Untitled").strip()
        creator = (j.get("uploader") or j.get("channel") or "Unknown").strip()
        page = j.get("webpage_url") or url
        return {"id": vid, "title": title, "creator": creator, "url": page}
    except Exception:
        return {"id": vid, "title": "Untitled", "creator": "Unknown", "url": url}

def main():
    os.makedirs("data/catalog", exist_ok=True)
    ids = find_ids()
    if not ids:
        print("No VTTs found in data/vtt or data/raw_vtt"); sys.exit(1)
    cat = {}
    for i, vid in enumerate(ids, 1):
        cat[vid] = yt_info(vid)
        if i % 25 == 0:
            print(f"  … {i}/{len(ids)}")
            time.sleep(0.1)
    with open("data/catalog/catalog.json","w",encoding="utf-8") as f:
        json.dump(cat, f, indent=2, ensure_ascii=False)
    print("catalog entries:", len(cat))
    any_item = next(iter(cat.values()))
    print("sample:", any_item)

if __name__ == "__main__":
    main()