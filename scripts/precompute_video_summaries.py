# scripts/precompute_video_summaries.py
# -*- coding: utf-8 -*-
"""
Precompute video summaries for Health | Nutrition Q&A.

Reads:
- DATA_DIR (env) or defaults to /var/data
- DATA_DIR/data/chunks/chunks.jsonl
- DATA_DIR/data/catalog/video_meta.json (optional)

Writes:
- DATA_DIR/data/catalog/video_summaries.json

Design:
- Topic-aware, multi-section summarization per video using rule-based
  keyword buckets (no LLM call needed).
- Each video gets:
    {
        "title": ...,
        "channel": ...,
        "published_at": ...,
        "tags": [...],                # high-level topics present
        "sections": [                 # topic sections
            {
                "id": "lipids_cardio",
                "topic": "Lipids & Cardiometabolic",
                "bullets": [
                    {"ts": 123.0, "text": "..."},
                    ...
                ],
            },
            ...
        ],
        "bullets": [                  # flattened, general bullets (for app.py)
            {"ts": 45.0, "text": "..."},
            ...
        ],
        "summary": "Short multi-topic summary string..."
    }

The existing app.py only *requires* `bullets` and `summary`, so it remains
fully backward-compatible. The richer sections can be used later.
"""

from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Any

# ---------- Paths ----------

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv("DATA_DIR", "/var/data")).resolve()
DATA_DIR = DATA_ROOT / "data"

CHUNKS_PATH     = DATA_DIR / "chunks" / "chunks.jsonl"
VIDEO_META_JSON = DATA_DIR / "catalog" / "video_meta.json"
VID_SUM_JSON    = DATA_DIR / "catalog" / "video_summaries.json"

# ---------- Basic utils ----------

def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _parse_ts(v) -> float:
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except Exception:
            return 0.0
    try:
        sec = 0.0
        for p in str(v).split(":"):
            sec = sec * 60 + float(p)
        return sec
    except Exception:
        return 0.0

def _truncate(txt: str, limit: int = 280) -> str:
    txt = _normalize_text(txt)
    if len(txt) <= limit:
        return txt
    return txt[:limit].rsplit(" ", 1)[0] + "…"

def _load_video_meta() -> Dict[str, Dict[str, Any]]:
    if VIDEO_META_JSON.exists():
        try:
            return json.loads(VIDEO_META_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

# ---------- Topic buckets ----------

TOPIC_SPEC = {
    "lipids_cardio": {
        "label": "Lipids & Cardiometabolic",
        "keywords": [
            "ldl", "apo b", "apob", "apolipoprotein b", "non-hdl",
            "triglyceride", "triglycerides", "cholesterol", "lipid", "lipids",
            "statin", "statins", "ezetimibe", "pcsk9", "inclisiran",
            "plaque", "atherosclerosis", "coronary", "cac", "calcium score",
            "lp(a)", "lipoprotein(a)", "coronary artery", "cardiovascular",
        ],
        "max_bullets": 5,
    },
    "exercise_training": {
        "label": "Exercise & Training",
        "keywords": [
            "exercise", "training", "resistance training", "strength training",
            "hypertrophy", "reps", "sets", "weightlifting", "lifting",
            "zone 2", "zone two", "vo2", "vo2 max", "aerobic", "anaerobic",
            "cardio", "conditioning", "endurance", "sprinting", "hiit",
        ],
        "max_bullets": 5,
    },
    "sleep_circadian": {
        "label": "Sleep & Circadian",
        "keywords": [
            "sleep", "deep sleep", "slow wave", "rem sleep", "rem",
            "circadian", "circadian rhythm", "chronotype", "insomnia",
            "sleep latency", "sleep hygiene", "sleep quality", "sleep duration",
            "melatonin", "sunlight", "morning light", "blue light",
        ],
        "max_bullets": 4,
    },
    "nutrition_fasting": {
        "label": "Nutrition & Fasting",
        "keywords": [
            "diet", "nutrition", "macronutrient", "protein", "carbohydrate",
            "carb", "fat", "fiber", "glycemic", "glucose", "fructose",
            "sugar", "refined", "processed food", "ultra-processed",
            "fasting", "time-restricted", "time restricted", "intermittent",
            "keto", "ketogenic", "low carb", "mediterranean",
        ],
        "max_bullets": 5,
    },
    "supplements_meds": {
        "label": "Supplements & Medications",
        "keywords": [
            "supplement", "supplements", "vitamin", "magnesium", "omega 3",
            "fish oil", "epa", "dha", "creatine", "berberine", "metformin",
            "rapamycin", "statin", "statins", "niacin", "coq10", "coenzyme q10",
            "glp-1", "semaglutide", "tirzepatide", "ozempic", "mounjaro",
        ],
        "max_bullets": 5,
    },
    "metabolic_weight": {
        "label": "Metabolic & Weight",
        "keywords": [
            "insulin", "insulin resistance", "insulin sensitive",
            "insulin sensitivity", "a1c", "hba1c", "glucose",
            "post-prandial", "postprandial", "cgm", "continuous glucose",
            "metabolic syndrome", "metabolic health", "obesity", "weight loss",
            "body fat", "visceral fat", "waist circumference",
        ],
        "max_bullets": 4,
    },
}

def _topic_hits(text: str) -> Dict[str, int]:
    """Return topic_id -> hit_count for this text."""
    txt = text.lower()
    hits: Dict[str, int] = {}
    for tid, spec in TOPIC_SPEC.items():
        count = 0
        for kw in spec["keywords"]:
            if kw in txt:
                count += 1
        if count > 0:
            hits[tid] = count
    return hits

# ---------- Summarization per video ----------

def summarize_video(vid: str,
                    segments: List[Dict[str, Any]],
                    meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    segments: list of {"ts": float, "text": str}
    """
    # Sort segments by time
    segments = sorted(segments, key=lambda s: float(s.get("ts", 0.0)))

    # Collect topic-specific segments
    topic_segments: Dict[str, List[Dict[str, Any]]] = {tid: [] for tid in TOPIC_SPEC.keys()}
    cleaned_segments: List[Dict[str, Any]] = []

    for seg in segments:
        text = _normalize_text(seg.get("text", ""))
        if not text:
            continue
        ts = float(seg.get("ts", 0.0))

        hits = _topic_hits(text)
        seg_obj = {"ts": ts, "text": text}

        if hits:
            for tid, cnt in hits.items():
                topic_segments[tid].append({
                    "ts": ts,
                    "text": text,
                    "score": float(cnt) * (1.0 + len(text) / 200.0),
                })
        cleaned_segments.append(seg_obj)

    sections: List[Dict[str, Any]] = []
    tags: List[str] = []
    general_bullets: List[Dict[str, Any]] = []

    # Build topic sections
    for tid, spec in TOPIC_SPEC.items():
        segs = topic_segments[tid]
        if not segs:
            continue

        # Deduplicate by text snippet
        seen = {}
        for s in segs:
            key = _truncate(s["text"], limit=140)
            if key not in seen:
                seen[key] = s
        segs = list(seen.values())

        # Sort by score (keyword density * length) then by time
        segs.sort(key=lambda s: (-float(s.get("score", 0.0)), float(s.get("ts", 0.0))))
        max_bullets = int(spec.get("max_bullets", 4))
        top = segs[:max_bullets]

        bullets = []
        for s in top:
            bullets.append({
                "ts": float(s["ts"]),
                "text": _truncate(s["text"]),
            })

        if bullets:
            sections.append({
                "id": tid,
                "topic": spec["label"],
                "bullets": bullets,
            })
            tags.append(spec["label"])
            # contribute top 1–2 to general bullets
            general_bullets.extend(bullets[:2])

    # Fallback if no topic matches at all
    if not sections:
        fallback = []
        for s in cleaned_segments[:8]:
            fallback.append({
                "ts": float(s["ts"]),
                "text": _truncate(s["text"]),
            })
        sections = [{
            "id": "general",
            "topic": "General",
            "bullets": fallback,
        }]
        tags.append("General")
        general_bullets = fallback[:]

    # Ensure we have a reasonably sized general bullet list
    if not general_bullets:
        for s in cleaned_segments[:8]:
            general_bullets.append({
                "ts": float(s["ts"]),
                "text": _truncate(s["text"]),
            })

    # Build summary string from sections
    summary_parts: List[str] = []
    for sec in sections:
        texts = " ".join(b["text"] for b in sec.get("bullets", []))
        if texts:
            summary_parts.append(f"{sec['topic']}: {texts}")
    summary = " ".join(summary_parts)
    if len(summary) > 1400:
        summary = summary[:1400].rsplit(" ", 1)[0] + "…"

    info = meta or {}
    # Normalize published_at field name
    pub = (
        info.get("published_at")
        or info.get("publishedAt")
        or info.get("date")
        or ""
    )

    return {
        "title": info.get("title", ""),
        "channel": (
            info.get("channel")
            or info.get("author")
            or info.get("uploader")
            or info.get("podcaster")
            or ""
        ),
        "published_at": pub,
        "tags": tags,
        "sections": sections,
        "bullets": general_bullets[:8],  # app.py expects this
        "summary": summary,
    }

# ---------- Main pipeline ----------

def main() -> None:
    print(f"[precompute_video_summaries] DATA_DIR = {DATA_ROOT}")
    print(f"[precompute_video_summaries] chunks path = {CHUNKS_PATH}")

    if not CHUNKS_PATH.exists():
        raise SystemExit(f"chunks.jsonl not found at {CHUNKS_PATH}")

    vm = _load_video_meta()
    print(f"[precompute_video_summaries] video_meta.json entries: {len(vm)}")

    videos: Dict[str, List[Dict[str, Any]]] = {}

    # Group transcript segments by video
    total_lines = 0
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for ln in f:
            total_lines += 1
            try:
                j = json.loads(ln)
            except Exception:
                continue

            text = _normalize_text(j.get("text", ""))
            if not text:
                continue

            m = (j.get("meta") or {})
            vid = (
                m.get("video_id") or m.get("vid") or m.get("ytid") or
                j.get("video_id") or j.get("vid") or j.get("ytid") or j.get("id")
            )
            if not vid:
                continue

            ts = _parse_ts(m.get("start", m.get("start_sec", 0)))
            videos.setdefault(vid, []).append({"ts": ts, "text": text})

    print(f"[precompute_video_summaries] total JSONL lines: {total_lines}")
    print(f"[precompute_video_summaries] videos detected: {len(videos)}")

    summaries: Dict[str, Any] = {}
    for i, (vid, segs) in enumerate(videos.items(), start=1):
        if not segs:
            continue
        meta = vm.get(vid, {})
        sv = summarize_video(vid, segs, meta)
        summaries[vid] = sv
        if i % 20 == 0:
            print(f"  summarized {i} videos...")

    VID_SUM_JSON.parent.mkdir(parents=True, exist_ok=True)
    VID_SUM_JSON.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[precompute_video_summaries] wrote {len(summaries)} summaries to {VID_SUM_JSON}")

if __name__ == "__main__":
    main()
