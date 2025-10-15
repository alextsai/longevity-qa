import os, re, json, shutil

SRC="data/chunks/chunks.jsonl"
TMP="data/chunks/chunks.jsonl.tmp"

def pick_video_id(d):
    # try common fields or URLs left by other code paths
    for k in ("video_id","vid","id"): 
        v=d.get(k)
        if isinstance(v,str) and re.fullmatch(r'[A-Za-z0-9_-]{11}', v): 
            return v
    for k in ("video_url","url","source"):
        u=d.get(k) or ""
        m=re.search(r"v=([A-Za-z0-9_-]{11})", u)
        if m: return m.group(1)
    return ""

def pick_start_seconds(d):
    # accept numeric, or parse HH:MM:SS(.mmm)
    s=d.get("start")
    if isinstance(s,(int,float)): return float(s)
    ts = d.get("start_ts") or d.get("start_time") or ""
    m=re.match(r"^(\d+):(\d{2}):(\d{2}(?:\.\d+)?)$", str(ts))
    if m:
        h,mn,ss=m.groups()
        return int(h)*3600 + int(mn)*60 + float(ss)
    return 0.0

fixed=0; total=0
with open(SRC,"r",encoding="utf-8") as fin, open(TMP,"w",encoding="utf-8") as fout:
    for line in fin:
        line=line.strip()
        if not line: continue
        total+=1
        try:
            d=json.loads(line)
        except: 
            continue
        vid = pick_video_id(d)
        if vid: d["video_id"]=vid
        d["start"]=pick_start_seconds(d)
        fout.write(json.dumps(d, ensure_ascii=False)+"\n")
        fixed+=1

shutil.move(TMP, SRC)
print(f"[normalize] wrote {SRC} lines={fixed}/{total}")
