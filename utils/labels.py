# utils/labels.py
from __future__ import annotations
from typing import Tuple, Dict
import json, pathlib

_META_PATH = pathlib.Path("data/catalog/video_meta.json")
_META: Dict[str, Dict[str, str]] | None = None

_PLACEHOLDERS = {"na", "n/a", "none", "null", "-"}

def _load_meta() -> Dict[str, Dict[str, str]]:
    global _META
    if _META is None:
        _META = {}
        if _META_PATH.exists():
            try:
                _META = json.loads(_META_PATH.read_text(encoding="utf-8"))
            except Exception:
                _META = {}
    return _META

def _parse_ts(v) -> float:
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return 0.0
    if isinstance(v, str):
        t = v.strip()
        if not t:
            return 0.0
        try:
            if ":" in t:
                parts = [float(x) for x in t.split(":")]
                if len(parts) == 3:
                    h, m, s = parts
                    return h*3600 + m*60 + s
                if len(parts) == 2:
                    m, s = parts
                    return m*60 + s
            return float(t)
        except Exception:
            return 0.0
    return 0.0

def _mmss(sec: float) -> str:
    sec = max(0, int(round(sec)))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

def _clean(s: str | None) -> str:
    """Turn placeholder-y strings into empty so fallbacks can run."""
    if not s:
        return ""
    t = s.strip()
    return "" if t.lower() in _PLACEHOLDERS else t

def label_and_url(meta: dict) -> Tuple[str, str]:
    """
    Build 'Channel — Title — mm:ss' and a timestamped YouTube URL.
    Treat 'NA' / 'N/A' / 'None' / 'Null' as empty so fallbacks apply.
    """
    vid = (
        meta.get("video_id")
        or meta.get("vid")
        or meta.get("ytid")
        or meta.get("id")
        or ""
    )
    start = _parse_ts(meta.get("start") or meta.get("start_sec") or 0)

    catalog = _load_meta()
    info = catalog.get(vid, {}) if vid else {}
    channel = _clean(info.get("channel"))
    title   = _clean(info.get("title"))

    # fallbacks from the chunk itself
    if not channel:
        channel = _clean(meta.get("channel") or meta.get("uploader") or meta.get("author"))
    if not title:
        title = _clean(meta.get("title") or meta.get("video_title"))

    # last resort: show the ID so it's never "Unknown — Untitled"
    if not channel:
        channel = f"Video {vid[:11]}" if vid else "Video"
    if not title:
        title = "Untitled"

    ts_lbl = _mmss(start)
    url = f"https://www.youtube.com/watch?v={vid}&t={max(0,int(start))}s" if vid else ""
    return f"{channel} — {title} — {ts_lbl}", url