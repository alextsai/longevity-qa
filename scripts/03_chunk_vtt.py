#!/usr/bin/env python3
from pathlib import Path
import re, json
from tqdm import tqdm
import webvtt

VTT_DIRS = [Path("data/vtt"), Path("data/raw_vtt")]
OUT_DIR = Path("data/chunks")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FP = OUT_DIR / "chunks.jsonl"

def ts_to_seconds(ts: str) -> int:
    h, m, s = ts.split(":")
    return int(float(h) * 3600 + float(m) * 60 + float(s))

def ytid_from_name(name: str) -> str | None:
    m = re.search(r"([A-Za-z0-9_-]{11})", name)
    return m.group(1) if m else None

def write_chunk(out_f, ytid, vtt_path, start_ts, end_ts, pieces) -> int:
    text = " ".join(pieces).strip()
    if not text:
        return 0
    rec = {
        "text": text,
        "meta": {
            "ytid": ytid,
            "start": start_ts,
            "end": end_ts,
            "path": str(vtt_path),
            "source_url": (
                f"https://www.youtube.com/watch?v={ytid}&t={ts_to_seconds(start_ts)}s"
                if ytid else None
            ),
        },
    }
    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 1

paths = []
for d in VTT_DIRS:
    if d.exists():
        paths.extend(sorted(d.glob("*.vtt")))

wrote = 0
with OUT_FP.open("w", encoding="utf-8") as out:
    for vtt_path in tqdm(paths, desc="chunk"):
        ytid = ytid_from_name(vtt_path.name)
        try:
            cues = list(webvtt.read(str(vtt_path)))
        except Exception:
            continue

        MAX_CHARS = 1200
        buf = []
        n_chars = 0
        chunk_start = None
        last_end = None

        for c in cues:
            seg = c.text.strip().replace("\n", " ")
            if not seg:
                continue
            if chunk_start is None:
                chunk_start = c.start
            if n_chars + len(seg) > MAX_CHARS and buf:
                wrote += write_chunk(out, ytid, vtt_path, chunk_start, last_end or c.end, buf)
                buf, n_chars, chunk_start = [], 0, c.start
            buf.append(seg)
            n_chars += len(seg)
            last_end = c.end

        if buf:
            wrote += write_chunk(out, ytid, vtt_path, chunk_start or "00:00:00.000", last_end or "00:00:00.000", buf)

print(f"[chunk] wrote {wrote} chunks -> {OUT_FP}")
