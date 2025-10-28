import json,csv,re,yaml
from pathlib import Path
def norm(s): return re.sub(r"\s+"," ",(s or "")).strip()
vm_path = Path("data/catalog/video_meta.json")
tax_path= Path("scripts/domain_taxonomy.yaml")
out_csv = Path("data/catalog/domain_labels.csv")
if not vm_path.exists(): raise SystemExit("missing data/catalog/video_meta.json")
vm = json.loads(vm_path.read_text(encoding="utf-8"))
tax = yaml.safe_load(tax_path.read_text(encoding="utf-8"))["labels"]
rows=[["video_id","title","channel","label"]]
for vid,info in vm.items():
    title = norm(info.get("title",""))
    desc  = norm(info.get("description") or info.get("desc") or "")
    chan  = (info.get("channel") or info.get("author") or info.get("uploader") or "") or ""
    bag = f"{title}. {desc}".lower()
    hits=set()
    for lab, kws in tax.items():
        for kw in kws:
            if kw.lower() in bag:
                hits.add(lab); break
    rows.append([vid, info.get("title",""), chan, ";".join(sorted(hits))])
with out_csv.open("w", newline="", encoding="utf-8") as f:
    w=csv.writer(f); w.writerows(rows)
print(f"Wrote {out_csv} — review 'label' column; you can add/remove labels; use semicolons for multi-labels.")
