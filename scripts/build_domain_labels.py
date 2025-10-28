import json,csv,re
from pathlib import Path
def norm(s): return re.sub(r"\s+"," ",(s or "")).strip()
meta=Path("data/catalog/video_meta.json")
out=Path("data/catalog/domain_labels.csv")
if not meta.exists(): raise SystemExit("missing data/catalog/video_meta.json")
vm=json.loads(meta.read_text(encoding="utf-8"))
rows=[("video_id","title","channel","label")]
for vid,info in vm.items():
    rows.append((vid, norm(info.get("title","")), norm(info.get("channel","")), ""))  # label blank
with out.open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerows(rows)
print(f"Wrote {out} — fill the 'label' column, one or more labels separated by ';'")
