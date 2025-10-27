# scripts/build_tag_index.py
import json, re, argparse
from pathlib import Path
from collections import Counter

KEYSETS = {
  "cardio": {"ldl","apob","statin","ezetimibe","pcsk9","bempedoic","inclisiran","atherosclerosis","coronary","angiogram","calcium","plaque","triglyceride"},
  "sleep": {"sleep","melatonin","magnesium","insomnia","apnea","circadian","valerian","glycine"},
  "nutrition": {"protein","fasting","fiber","omega","olive","polyphenol","choline","glycemic","sugar","fructose","polyphenols"},
  "immune": {"immune","infection","vitamin d","zinc","omega-3","inflammation","cytokine","antibody"},
}

def toks(s): return re.findall(r"[a-z0-9]+", (s or "").lower())

def main(a):
    vs = json.loads(Path(a.video_summaries).read_text(encoding="utf-8"))
    vm = json.loads(Path(a.video_meta).read_text(encoding="utf-8"))

    out={}
    for vid in set(list(vs.keys()) + list(vm.keys())):
        tags=set()
        title = (vm.get(vid,{}) or {}).get("title","")
        bullets = " ".join([b.get("text","") for b in (vs.get(vid,{}).get("bullets",[]) or [])])
        bag = set(toks(title) + toks(bullets))
        # domain tags
        for dom,keys in KEYSETS.items():
            if len(bag & keys)>0: tags.add(dom)
        # top keywords
        cnt=Counter(toks(title+" "+bullets))
        for w,_ in cnt.most_common(12):
            if len(w)>=4: tags.add(w)
        out[vid]=sorted(tags)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--video_summaries", required=True, help="/var/data/data/catalog/video_summaries.json")
    ap.add_argument("--video_meta", required=True, help="/var/data/data/catalog/video_meta.json")
    ap.add_argument("--out", required=True, help="/var/data/data/catalog/tag_index.json")
    main(ap.parse_args())
