import os, csv, time, re, sys
from pathlib import Path
import requests

API_KEY = os.environ.get("YT_API_KEY", "").strip()
if not API_KEY:
    print("[error] Missing YT_API_KEY in environment. Set it then re-run.")
    sys.exit(1)

channels_csv = Path("data/catalog/channels.csv")
out_csv = Path("data/catalog/videos.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)

def http_get(url, params, retries=6, backoff=1.2):
    last_err = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            # Surface API error details if present
            try:
                j = r.json()
                if "error" in j:
                    msg = j["error"].get("message", "")
                    print(f"[api][{r.status_code}] {msg}")
            except Exception:
                pass
            last_err = requests.HTTPError(f"{r.status_code} {r.reason}")
        except Exception as e:
            last_err = e
        time.sleep((i + 1) * backoff)
    raise last_err

def resolve_channel_and_uploads(url_or_handle: str):
    """
    Return (channel_id, uploads_playlist_id) using channel handle when possible.
    Supports:
      - https://www.youtube.com/@handle
      - https://www.youtube.com/@handle/videos
      - https://www.youtube.com/channel/UCxxxx
    """
    u = url_or_handle.strip()

    # Direct playlist links are handled upstream
    if "/playlist" in u and "list=" in u:
        return (None, None)

    # If channel URL with /channel/UC…
    if "/channel/" in u:
        ch_id = u.split("/channel/")[1].split("/")[0]
        data = http_get(
            "https://www.googleapis.com/youtube/v3/channels",
            {"part": "contentDetails", "id": ch_id, "key": API_KEY},
        )
        items = data.get("items", [])
        if not items:
            return (None, None)
        upid = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        return (ch_id, upid)

    # Try handle like @hubermanlab
    m = re.search(r"@[\w\.\-]+", u)
    if m:
        handle = m.group(0)  # includes '@'
        data = http_get(
            "https://www.googleapis.com/youtube/v3/channels",
            {"part": "id,contentDetails", "forHandle": handle, "key": API_KEY},
        )
        items = data.get("items", [])
        if items:
            ch_id = items[0]["id"]
            upid = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
            return (ch_id, upid)

    # Fallback: search by last path segment
    query = u.rstrip("/").split("/")[-1]
    data = http_get(
        "https://www.googleapis.com/youtube/v3/search",
        {"part": "snippet", "q": query, "type": "channel", "maxResults": 1, "key": API_KEY},
    )
    items = data.get("items", [])
    if not items:
        return (None, None)
    ch_id = items[0]["snippet"]["channelId"]
    data2 = http_get(
        "https://www.googleapis.com/youtube/v3/channels",
        {"part": "contentDetails", "id": ch_id, "key": API_KEY},
    )
    it2 = data2.get("items", [])
    if not it2:
        return (None, None)
    upid = it2[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    return (ch_id, upid)

def list_playlist_videos(playlist_id: str, max_pages=100):
    page_token = None
    videos = []
    for _ in range(max_pages):
        data = http_get(
            "https://www.googleapis.com/youtube/v3/playlistItems",
            {
                "part": "snippet,contentDetails",
                "playlistId": playlist_id,
                "maxResults": 50,
                "pageToken": page_token or "",
                "key": API_KEY,
            },
        )
        for it in data.get("items", []):
            cd = it.get("contentDetails", {})
            sn = it.get("snippet", {})
            vid = cd.get("videoId")
            if not vid:
                continue
            title = sn.get("title", "") or ""
            publishedAt = cd.get("videoPublishedAt") or sn.get("publishedAt", "")
            videos.append(
                {
                    "video_id": vid,
                    "video_title": title,
                    "publish_date": publishedAt,
                }
            )
        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.2)
    return videos

def details_for_ids(ids):
    out = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        data = http_get(
            "https://www.googleapis.com/youtube/v3/videos",
            {"part": "snippet,statistics", "id": ",".join(chunk), "key": API_KEY},
        )
        for it in data.get("items", []):
            vid = it["id"]
            sn = it.get("snippet", {})
            st = it.get("statistics", {})
            out[vid] = {
                "title": sn.get("title", ""),
                "description": sn.get("description", ""),
                "channel_title": sn.get("channelTitle", ""),
                "views_total": int(st.get("viewCount", "0") or 0),
                "like_count": int(st.get("likeCount", "0") or 0),
                "publishedAt": sn.get("publishedAt", ""),
            }
        time.sleep(0.2)
    return out

# ---- main ----
rows = []
with open(channels_csv, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        u = r["url"].strip()
        label = (r.get("label") or "").strip()
        print(f"[api] Channel/playlist: {u}")

        # Direct playlist
        if "list=" in u:
            pid = u.split("list=")[1].split("&")[0]
        else:
            ch_id, pid = resolve_channel_and_uploads(u)
            if not pid:
                print(f"[warn] could not resolve uploads playlist for {u}")
                continue

        try:
            vids = list_playlist_videos(pid)
        except Exception as e:
            print(f"[warn] playlist fetch failed for {u}: {e}")
            continue

        for v in vids:
            v["url"] = f"https://www.youtube.com/watch?v={v['video_id']}"
            v["channel_label"] = label
            rows.append(v)
        time.sleep(0.2)

# Dedupe and enrich
uniq = {}
for r in rows:
    vid = r["video_id"]
    if vid not in uniq:
        uniq[vid] = r

ids = list(uniq.keys())
meta = details_for_ids(ids) if ids else {}

out_fields = [
    "video_id",
    "video_title",
    "url",
    "channel",
    "channel_label",
    "publish_date",
    "views_total",
    "like_count",
    "description",
]

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=out_fields)
    w.writeheader()
    for vid, r in uniq.items():
        m = meta.get(vid, {})
        title = r.get("video_title") or m.get("title", "")
        channel = m.get("channel_title", "") or r.get("channel_label", "")
        publish_date = r.get("publish_date") or m.get("publishedAt", "")
        w.writerow(
            {
                "video_id": vid,
                "video_title": title,
                "url": r["url"],
                "channel": channel,
                "channel_label": r.get("channel_label", ""),
                "publish_date": publish_date,
                "views_total": m.get("views_total", 0),
                "like_count": m.get("like_count", 0),
                "description": m.get("description", ""),
            }
        )

print(f"[api] wrote {out_csv} with {len(uniq)} videos")
