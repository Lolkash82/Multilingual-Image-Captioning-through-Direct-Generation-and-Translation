"""Prepare Multi30K (Task 1, translation variant): captions from GitHub, images from Hugging Face.

Usage: python scripts/prepare_multi30k.py [--skip-images]
Creates data/multi30k/{train,val,test_2016_flickr}.json with entries
{"image": "<file>.jpg", "captions": {"en": ..., "de": ..., "fr": ...}}.
"""
import argparse
import gzip
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import load_config  # noqa: E402

BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1"
SPLITS = ["train", "val", "test_2016_flickr"]
LANGS = ["en", "de", "fr"]


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-images", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    data_dir = Path(cfg["paths"]["data_dir"])
    images_dir = Path(cfg["paths"]["images_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        images = fetch(f"{BASE}/image_splits/{split}.txt").decode("utf-8").split()
        caps = {}
        for lang in LANGS:
            raw = gzip.decompress(fetch(f"{BASE}/raw/{split}.{lang}.gz")).decode("utf-8")
            caps[lang] = [line.strip() for line in raw.split("\n")]
            if caps[lang] and caps[lang][-1] == "":
                caps[lang].pop()
            if len(caps[lang]) != len(images):
                raise SystemExit(f"{split}.{lang}: {len(caps[lang])} captions vs {len(images)} images")
        entries = [{"image": img, "captions": {l: caps[l][i] for l in LANGS}} for i, img in enumerate(images)]
        with open(data_dir / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False)
        avg = {l: round(sum(len(c.split()) for c in caps[l]) / len(images), 1) for l in LANGS}
        print(f"{split}: {len(entries)} entries, avg words {avg}")

    if args.skip_images:
        return
    if not any(images_dir.glob("*.jpg")):
        from huggingface_hub import hf_hub_download
        zip_path = hf_hub_download("nlphuji/flickr30k", "flickr30k-images.zip", repo_type="dataset")
        images_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():
                if member.lower().endswith(".jpg"):
                    (images_dir / Path(member).name).write_bytes(z.read(member))
    missing = []
    for split in SPLITS:
        with open(data_dir / f"{split}.json", encoding="utf-8") as f:
            missing += [e["image"] for e in json.load(f) if not (images_dir / e["image"]).exists()]
    print(f"Images present; missing: {len(missing)}" + (f" e.g. {missing[:3]}" if missing else ""))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
