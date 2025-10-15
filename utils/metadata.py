from __future__ import annotations
import json, os
from typing import Dict, Tuple, Any

META_PATH = "data/catalog/video_meta.json"

def _load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_video_meta(path: str = META_PATH) -> Dict[str, Dict[str,str]]:
    meta = _load_json(path)
    if not isinstance(meta, dict): return {}
    fixed={}
    for vid, m in meta.items():
        if not isinstance(m, dict): m={}
        chan=(m.get("channel") or "").strip() or "Unknown channel"
        title=(m.get("title") or "").strip() or "Untitled"
        fixed[str(vid)]={"channel":chan,"title":title}
    return fixed

def _fmt_mmss(seconds): 
    try: s=int(round(float(seconds)))
    except: return "00:00"
    if s<0: s=0
    m,s=divmod(s,60)
    return f"{m:02d}:{s:02d}"

def _start_seconds(src):
    if isinstance(src, dict):
        for k in ("start","start_sec","start_seconds","ts","ts_seconds"):
            if k in src:
                try: return int(float(src[k]))
                except: pass
    try: return int(float(src))
    except: return 0

def _video_id(src):
    if isinstance(src, dict):
        for k in ("video_id","vid","ytid","id"):
            v=src.get(k)
            if v: return str(v)
    if isinstance(src, str): return src
    return ""

def meta_label_for(source: Any, meta: Dict[str, Dict[str,str]] | None = None
                  ) -> Tuple[str,str,str,int]:
    vid=_video_id(source); s=_start_seconds(source)
    if not meta or not isinstance(meta, dict): meta={}
    m=meta.get(vid, {})
    channel=(m.get("channel") or "Unknown channel").strip() or "Unknown channel"
    title=(m.get("title") or "Untitled").strip() or "Untitled"
    label=f"{channel} — {title} — {_fmt_mmss(s)}"
    url=f"https://www.youtube.com/watch?v={vid}&t={max(0,s)}s" if vid else ""
    return label, url, vid, s
