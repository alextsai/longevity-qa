import os, re, json, glob, webvtt, math
from tqdm import tqdm

VTT_DIR = "data/vtt"
OUT_FP  = "data/chunks/chunks.jsonl"

def id_from_vtt_path(p: str) -> str:
    b = os.path.basename(p)
    c = b.split('.')[0]
    if re.fullmatch(r'[A-Za-z0-9_-]{11}', c): return c
    m = re.search(r'([A-Za-z0-9_-]{11})', b)
    return m.group(1) if m else ""

def to_seconds(ts: str) -> float:
    # "HH:MM:SS.mmm" -> seconds
    h, m, s = ts.split(':')
    return int(h)*3600 + int(m)*60 + float(s)

os.makedirs(os.path.dirname(OUT_FP), exist_ok=True)
n=0
with open(OUT_FP, "w", encoding="utf-8") as fout:
    vtts = sorted(glob.glob(os.path.join(VTT_DIR, "*.vtt")))
    for vtt_path in tqdm(vtts, desc="chunk"):
        vid = id_from_vtt_path(vtt_path)
        if not vid:
            # skip files we cannot map to a YouTube id
            continue
        try:
            for cue in webvtt.read(vtt_path):
                txt = (cue.text or "").strip()
                if not txt: continue
                start_s = to_seconds(cue.start.replace(',', '.'))
                d = {
                    "video_id": vid,          # REQUIRED for citations
                    "start":    round(start_s, 3),  # numeric seconds
                    "text":     txt
                }
                fout.write(json.dumps(d, ensure_ascii=False)+"\n")
                n += 1
        except Exception as e:
            # bad vtt – skip, but don't crash the run
            pass

print(f"[chunk] wrote {OUT_FP} lines={n}")
