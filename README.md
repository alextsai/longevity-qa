# Longevity Q&A Starter (No-CS background friendly)

Goal: ingest YouTube videos from selected channels, build a searchable knowledge base with timestamped chunks, and run a simple Q&A web UI with citations.

## 0) Prereqs
- macOS or Windows.
- Python 3.10+ installed.
- Disk space ~3–5 GB for transcripts and embeddings.

## 1) Setup
```bash
cd longevity_qa_starter
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Declare channels
Edit `data/catalog/channels.csv` and add channel handles or playlist URLs. Two examples are included.

## 3) Discover videos
This lists each channel's videos with id, title, and URL.
```bash
python scripts/01_discover.py
```
Outputs `data/catalog/videos.csv`.

## 4) Download transcripts (VTT)
This fetches manual subs when available, else auto-subs. No audio/video is downloaded.
```bash
python scripts/02_fetch_subtitles.py
```
VTT files saved in `data/raw_vtt/`.

## 5) Build 1–3 min chunks with metadata
This converts VTT to text and creates 120s chunks with 20s overlap and timestamps.
```bash
python scripts/03_chunk_vtt.py
```
Outputs `data/chunks/chunks.jsonl`.

## 6) Create an embedding index
This encodes chunks and builds a FAISS index for fast retrieval.
```bash
python scripts/04_build_index.py
```
Outputs `data/index/faiss.index` and `data/index/meta.jsonl`.

## 7) Run the Q&A app
```bash
streamlit run app/app.py
```
Open the local URL shown in the terminal. Type a question. You will see top chunks with timestamps and links. Optional: set `OPENAI_API_KEY` to enable answer synthesis on top of retrieval.

## Why chunks?
- Precision: short spans reduce topic drift.
- Citations: link to exact moment via timestamps.
- Safety: grade claims per chunk, not per whole video.
- Updates: re-embed only changed chunks.

## Tips
- If a channel has many videos, you can restrict date in `01_discover.py` filters first.
- If yt-dlp fails on a URL, re-run; YouTube changes HTML often and yt-dlp updates fast.


## (Alt) Discover with YouTube API + prioritize longevity topics
1) Put your key in `.env` as `YT_API_KEY=...`
2) Reload your shell: `source .venv/bin/activate && export $(grep -v '^#' .env | xargs -d '\n')`
3) Run:
```bash
python scripts/01a_fetch_videos_api.py
python scripts/01b_prioritize_and_topics.py
```
Then continue with steps 4–7. Subtitle and chunk scripts will auto-use `videos_prioritized.csv` when present.
