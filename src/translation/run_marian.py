import argparse
import json
import time
from pathlib import Path

import torch

from ..utils.config import load_config


class MarianTranslator:
    def __init__(self, model_name, device="cuda"):
        from transformers import MarianMTModel, MarianTokenizer

        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def translate(self, texts, batch_size, num_beams, max_length):
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer([t or "." for t in batch], return_tensors="pt", padding=True,
                                 truncation=True).to(self.device)
            gen = self.model.generate(**enc, num_beams=num_beams, max_length=max_length)
            dec = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            out.extend(d if t else "" for d, t in zip(dec, batch))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", default=None, help="EN captions JSON (default outputs/captions/en.json)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc = cfg["translation"]
    outputs = Path(cfg["paths"]["outputs_dir"])
    src_path = Path(args.source) if args.source else outputs / "captions" / "en.json"
    with open(src_path, encoding="utf-8") as f:
        source = json.load(f)
    texts = [r["caption"] for r in source["records"]]
    out_dir = outputs / "translations"
    out_dir.mkdir(parents=True, exist_ok=True)

    for lang, model_name in tc["models"].items():
        translator = MarianTranslator(model_name)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        translated = translator.translate(texts, tc["batch_size"], tc["num_beams"], tc["max_length"])
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        records = [
            {"image": r["image"], "caption": t, "source_en": r["caption"], "references": r["references"]}
            for r, t in zip(source["records"], translated)
        ]
        meta = {"mode": f"translate_{lang}", "language": lang, "model": model_name, "samples": len(records),
                "total_seconds": round(total, 2), "seconds_per_sample": round(total / len(records), 5),
                "source_file": str(src_path), **{k: tc[k] for k in ("batch_size", "num_beams", "max_length")}}
        path = out_dir / f"translate_{lang}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "records": records}, f, ensure_ascii=False, indent=1)
        print(f"{lang}: {total:.1f} s total -> {path}")
        del translator
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
