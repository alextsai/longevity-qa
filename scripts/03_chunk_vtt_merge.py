#!/usr/bin/env python3
import json
import re
from pathlib import Path
from tqdm import tqdm
import webvtt

VTT_DIR = Path("data/vtt")
OUT_FP  = Path("data/chunks/chunks.jsonl")
MAX_SECONDS = 45
MAX_CHARS   = 700

def id_from_path(p: Path):
    m = re.search(r'([A-Za-z0-9_-]{11})', p.name)
    return m.group(1) if m else ""

def to_seconds(ts: str) -> float:
    ts = ts.replace(',', '.')
    h, m, s = ts.split(':')
    return int(h)*3600 + int(m)*60 + float(s)

OUT_FP.parent.mkdir(parents=True, exist_ok=True)
n=0
with open(OUT_FP, "w", encoding="utf-8") as fout:
    for vtt in tqdm(sorted(VTT_DIR.glob("*.vtt")), desc="merge"):
        vid = id_from_path(vtt)
        if not vid:
            continue
        try:
            cur_text = []
            cur_start = None
            cur_chars = 0
            last_end = None

            for cue in webvtt.read(str(vtt)):
                txt = (cue.text or "").strip()
                if not txt:
                    continue
                s = to_seconds(cue.start)
                e = to_seconds(cue.end)

                if cur_start is None:
                    cur_start = s
                    cur_text = [txt]
                    cur_chars = len(txt)
                    last_end = e
                    continue

                exceed = ((e - cur_start) > MAX_SECONDS) or ((cur_chars + 1 + len(txt)) > MAX_CHARS)
                if exceed:
                    fout.write(json.dumps({
                        "video_id": vid,
                        "start": round(cur_start, 3),
                        "text": " ".join(cur_text)
                    }, ensure_ascii=False) + "\n")
                    n += 1
                    cur_start = s
                    cur_text = [txt]
                    cur_chars = len(txt)
                    last_end = e
                else:
                    cur_text.append(txt)
                    cur_chars += 1 + len(txt)
                    last_end = e

            if cur_text:
                fout.write(json.dumps({
                    "video_id": vid,
                    "start": round(cur_start if cur_start is not None else 0.0, 3),
                    "text": " ".join(cur_text)
                }, ensure_ascii=False) + "\n")
                n += 1
        except Exception:
            pass

print(f"[chunk-merge] wrote {OUT_FP} lines={n}")
