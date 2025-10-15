#!/usr/bin/env python3
"""
Fetch YouTube subtitles (.vtt) for all video IDs in data/catalog/videos_prioritized.csv.

- Accepts catalog header: video_id OR id OR ytid
- Respects env MAX_VIDS (default: 50)
- Uses cookies if provided via:
    YTDLP_COOKIES_BROWSER='chrome[:ProfileName]'  e.g. 'chrome:Default'
  or
    YTDLP_COOKIES_FILE='/path/to/cookies.txt'
- Writes VTTs to: data/raw_vtt/{VIDEO_ID}.en.vtt (or .vtt for the matched lang)
- Logs progress to: data/catalog/subtitles_progress.csv

Requires: yt-dlp installed and on PATH (brew install yt-dlp)
"""

import csv
import os
import sys
import time
import json
import shutil
import subprocess
from pathlib import Path

CATALOG = Path("data/catalog/videos_prioritized.csv")
RAW_VTT_DIR = Path("data/raw_vtt")
PROGRESS_CSV = Path("data/catalog/subtitles_progress.csv")

DEFAULT_MAX_VIDS = 50
SLEEP_ON_ERROR_SEC = 2.0   # small backoff between failures to be gentle

def info(msg: str):
    print(msg, flush=True)

def read_ids_from_catalog() -> list[str]:
    if not CATALOG.exists():
        info(f"[subs] ERROR: catalog not found -> {CATALOG}")
        sys.exit(1)
    ids = []
    with CATALOG.open(newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            vid = (r.get("video_id") or r.get("id") or r.get("ytid") or "").strip()
            # common bad cells include full URLs—extract v param if present
            if vid.startswith("http"):
                # very quick extraction
                # e.g. https://www.youtube.com/watch?v=QmOF0crdyRU
                # also short youtu.be/...
                if "watch?v=" in vid:
                    vid = vid.split("watch?v=")[-1].split("&")[0]
                elif "youtu.be/" in vid:
                    vid = vid.split("youtu.be/")[-1].split("?")[0]
            if vid:
                ids.append(vid)
    return ids

def build_yt_dlp_cmd(video_id: str) -> list[str]:
    """Build a yt-dlp command that downloads subtitles only (no media)."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    cmd = [
        "yt-dlp",
        "--ignore-errors",
        "--no-warnings",
        "--skip-download",
        # try both manual and auto subs; prefer vtt
        "--write-sub",
        "--write-auto-sub",
        "--sub-format", "vtt",
        "--sub-langs", "en.*",        # en, en-US, etc. (adjust if you want all: "all")
        "-o", "%(id)s",               # base name is the video id
        "-P", str(RAW_VTT_DIR),       # output directory
        url,
    ]

    # Add cookies if provided for fewer 429 / age restricted
    cookies_browser = os.getenv("YTDLP_COOKIES_BROWSER", "").strip()
    cookies_file = os.getenv("YTDLP_COOKIES_FILE", "").strip()
    cookies_profile = os.getenv("YTDLP_COOKIES_PROFILE", "").strip()

    if cookies_browser:
        # allow chrome or chrome:Profile
        # if user set profile separately, append it
        arg = cookies_browser
        if cookies_profile and ":" not in arg:
            arg = f"{arg}:{cookies_profile}"
        cmd.insert(1, f"--cookies-from-browser={arg}")
    elif cookies_file and Path(cookies_file).exists():
        cmd.insert(1, f"--cookies={cookies_file}")

    return cmd

def ensure_dirs():
    RAW_VTT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_CSV.parent.mkdir(parents=True, exist_ok=True)

def append_progress(rows: list[dict]):
    write_header = not PROGRESS_CSV.exists()
    with PROGRESS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["video_id", "url", "status", "message", "path"]
        )
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ensure_dirs()

    all_ids = read_ids_from_catalog()
    if not all_ids:
        info("[subs] total videos to attempt: 0 (catalog empty)")
        sys.exit(0)

    try:
        max_vids = int(os.getenv("MAX_VIDS", DEFAULT_MAX_VIDS))
    except ValueError:
        max_vids = DEFAULT_MAX_VIDS

    target_ids = all_ids[:max_vids]
    info(f"[subs] total videos to attempt: {len(target_ids)} (from {CATALOG})")

    ok = 0
    miss = 0
    batched_logs = []

    # quick existing-file check: skip if a .vtt already present
    def has_vtt(vid: str) -> bool:
        # examples: VID.en.vtt, VID.en-US.vtt, sometimes plain VID.vtt
        for p in RAW_VTT_DIR.glob(f"{vid}*.vtt"):
            if p.is_file() and p.stat().st_size > 0:
                return True
        return False

    for i, vid in enumerate(target_ids, 1):
        url = f"https://www.youtube.com/watch?v={vid}"

        # already fetched?
        if has_vtt(vid):
            info(f"[subs] ({i}/{len(target_ids)}) {vid} ✓ (already present)")
            batched_logs.append({
                "video_id": vid, "url": url, "status": "ok", "message": "already_downloaded", "path": str(RAW_VTT_DIR)
            })
            ok += 1
            continue

        cmd = build_yt_dlp_cmd(vid)

        try:
            # capture both stdout and stderr for diagnostics
            res = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
            )
            stdout = res.stdout or ""
            stderr = res.stderr or ""

            # yt-dlp returns 0 on success; but verify a vtt exists
            if res.returncode == 0 and has_vtt(vid):
                info(f"[subs] ({i}/{len(target_ids)}) {vid} ✓")
                batched_logs.append({
                    "video_id": vid, "url": url, "status": "ok", "message": "downloaded", "path": str(RAW_VTT_DIR)
                })
                ok += 1
            else:
                # detect common causes
                msg = "no subtitles or rate-limited"
                text = (stdout + "\n" + stderr).lower()
                if "http error 429" in text or "too many requests" in text:
                    msg = "429 rate-limited"
                elif "sign in to confirm" in text or "age-restricted" in text:
                    msg = "age/consent restricted"
                elif "copyright claim" in text:
                    msg = "copyright restricted"
                elif "subtitles" in text and "unavailable" in text:
                    msg = "subtitles unavailable"
                info(f"[subs] ({i}/{len(target_ids)}) {vid} ✗ ({msg})")
                batched_logs.append({
                    "video_id": vid, "url": url, "status": "miss", "message": msg, "path": ""
                })
                miss += 1
                time.sleep(SLEEP_ON_ERROR_SEC)

        except Exception as e:
            miss += 1
            emsg = f"exception: {e.__class__.__name__}"
            info(f"[subs] ({i}/{len(target_ids)}) {vid} ✗ ({emsg})")
            batched_logs.append({
                "video_id": vid, "url": url, "status": "miss", "message": emsg, "path": ""
            })
            time.sleep(SLEEP_ON_ERROR_SEC)

        # flush logs periodically
        if len(batched_logs) >= 50:
            append_progress(batched_logs)
            batched_logs.clear()

    # flush any remaining logs
    if batched_logs:
        append_progress(batched_logs)

    info(f"[subs] done. ok={ok} miss={miss} saved dir={RAW_VTT_DIR}")
    if ok == 0:
        info("[subs] no subtitles saved. If you keep seeing 429, set MAX_VIDS=10 and try again, "
             "or set YTDLP_COOKIES_BROWSER='chrome:Default' and rerun.")

if __name__ == "__main__":
    main()