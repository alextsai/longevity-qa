import pandas as pd, subprocess, os
from pathlib import Path

N = int(os.environ.get("SMOKE_N", "3"))
vfp = Path("data/catalog/videos.csv")
sub_cmd = ["python", "scripts/02_fetch_subtitles.py"]
chk_cmd = ["python", "scripts/03_chunk_vtt.py"]
idx_cmd = ["python", "scripts/04_build_index.py"]

df = pd.read_csv(vfp).head(N)
tmp = Path("data/catalog/videos_smoke.csv")
df.to_csv(tmp, index=False)
print(f"[smoke] Using first {N} videos -> {tmp}")

# Patch the fetch and chunk scripts to read the smoke file if present via env
os.environ["VIDEOS_CSV_OVERRIDE"] = str(tmp)

print("[smoke] Fetching subs...")
subprocess = __import__("subprocess")
env = os.environ.copy()
env["VIDEOS_CSV_OVERRIDE"] = str(tmp)
subprocess.run(["python","scripts/02_fetch_subtitles.py"], check=True, env=env)
print("[smoke] Chunking...")
subprocess.run(["python","scripts/03_chunk_vtt.py"], check=True, env=env)
print("[smoke] Building index...")
subprocess.run(["python","scripts/04_build_index.py"], check=True, env=env)
print("[smoke] OK. Run: streamlit run app/app.py")
