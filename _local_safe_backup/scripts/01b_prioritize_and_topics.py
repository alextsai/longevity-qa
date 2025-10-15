import os, re, csv, math, time
from pathlib import Path
import pandas as pd

inp = Path("data/catalog/videos.csv")
outp = Path("data/catalog/videos_prioritized.csv")

# Topic keyword buckets
TOPICS = {
    "mental": ["stress","anxiety","depress","focus","motivation","mindset","sleep","nsdr","yoga nidra","cognitive","brain"],
    "physical": ["exercise","workout","training","strength","cardio","hypertrophy","mobility","rehab","steps","walking","VO2","HIIT"],
    "nutritional": ["diet","nutrition","food","foods","fasting","protein","fiber","glycemic","omega","vitamin","mineral","salt","sugar"],
    "longevity": ["longevity","lifespan","healthspan","aging","senescence","telomere","mTOR","autophagy","sauna","cold","metformin","rapa"]
}

def tag_topics(title, desc):
    text = f"{title}\n{desc}".lower()
    tags = []
    for k, kws in TOPICS.items():
        if any(kw in text for kw in kws):
            tags.append(k)
    if not tags:
        # heuristic fallbacks
        if "sleep" in text: tags.append("mental")
        if "blood pressure" in text or "hypertension" in text: tags.append("physical")
    return "|".join(sorted(set(tags))) or "other"

def topic_relevance(tags):
    # Longevity-first policy: videos with 'longevity' tag score highest
    s = 0.0
    if "longevity" in tags: s += 1.0
    if "nutritional" in tags: s += 0.6
    if "physical" in tags: s += 0.6
    if "mental" in tags: s += 0.6
    return min(s, 1.8)  # cap

def recency_score(days):
    return math.exp(-max(days,0)/120.0)

df = pd.read_csv(inp)
# Parse age days
from datetime import datetime, timezone
def age_days(iso):
    try:
        dt = datetime.fromisoformat(iso.replace("Z","+00:00"))
        return (datetime.now(timezone.utc) - dt).days
    except Exception:
        return 9999

df["age_days"] = df["publish_date"].apply(age_days)
df["topics"] = df.apply(lambda r: tag_topics(r.get("video_title",""), r.get("description","")), axis=1)
df["topic_relevance"] = df["topics"].apply(topic_relevance)
df["popularity"] = (df["views_total"]+1).apply(lambda v: math.log1p(max(v,0)))
df["recency"] = df["age_days"].apply(recency_score)

# Priority formula: longevity/topic first, then views, then recency
df["priority"] = (df["topic_relevance"]*0.5) + (df["popularity"]*0.3) + (df["recency"]*0.2)

df = df.sort_values(["priority","publish_date","views_total"], ascending=[False, False, False])
df.to_csv(outp, index=False)
print(f"[prioritize] wrote {outp} rows={len(df)}")
