import os, json, faiss, numpy as np
from pathlib import Path

index_fp = Path("data/index/faiss.index")
meta_fp = Path("data/index/meta.jsonl")
chunks_fp = Path("data/chunks/chunks.jsonl")

problems = []

if not chunks_fp.exists():
    problems.append("Missing data/chunks/chunks.jsonl")
else:
    # Check JSONL structure
    with open(chunks_fp, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                j = json.loads(line)
                for key in ["url","text","start_sec","end_sec"]:
                    if key not in j:
                        problems.append(f"Chunk {i} missing key: {key}")
                        break
            except Exception as e:
                problems.append(f"Invalid JSON on line {i}: {e}")
                break

if not index_fp.exists():
    problems.append("Missing FAISS index at data/index/faiss.index")
if not meta_fp.exists():
    problems.append("Missing metadata at data/index/meta.jsonl")

if problems:
    print("[validate] Issues found:")
    for p in problems:
        print(" -", p)
    raise SystemExit(1)
else:
    print("[validate] All basic checks passed.")
