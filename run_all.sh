#!/usr/bin/env bash
set -euo pipefail
echo "[1/5] Discover videos"
python scripts/01_discover.py
echo "[2/5] Fetch subtitles"
python scripts/02_fetch_subtitles.py
echo "[3/5] Chunk VTT"
python scripts/03_chunk_vtt.py
echo "[4/5] Build index"
python scripts/04_build_index.py
echo "[5/5] Launch app (Ctrl+C to stop)"
streamlit run app/app.py
