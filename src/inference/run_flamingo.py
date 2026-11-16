"""Generate captions for one mode. Usage: python -m src.inference.run_flamingo --lang en|de|fr

Writes outputs/captions/{en,direct_de,direct_fr}.json. Progress is appended to a .partial.jsonl
file after every batch, so an interrupted SLURM job can simply be resubmitted.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from ..utils.config import load_config
from ..utils.data import Multi30KLoader
from .flamingo import FlamingoInference


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=["en", "de", "fr"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N test images (smoke test)")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    mode = "en" if args.lang == "en" else f"direct_{args.lang}"
    out_dir = Path(cfg["paths"]["outputs_dir"]) / "captions"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".limit{args.limit}" if args.limit else ""
    final_path = out_dir / f"{mode}{suffix}.json"
    partial_path = out_dir / f"{mode}{suffix}.partial.jsonl"
    batch_size = args.batch_size or cfg["flamingo"]["batch_size"]

    loader = Multi30KLoader(cfg["paths"]["data_dir"], cfg["paths"]["images_dir"])
    ds = cfg["dataset"]
    test = loader.load_split(ds["test_split"])[: args.limit]
    shots = loader.sample_few_shot(ds["fewshot_split"], ds["num_shots"], cfg["seed"])

    done = {}
    if partial_path.exists():
        with open(partial_path, encoding="utf-8") as f:
            done = {r["image"]: r for r in map(json.loads, f)}
        print(f"Resuming: {len(done)} captions already generated")
    todo = [e for e in test if e["image"] not in done]

    engine = FlamingoInference(cfg)
    engine.set_few_shot(shots, args.lang)
    print(f"Mode {mode} | {len(todo)} images | batch {batch_size} | prompt tokens {engine.prompt_tokens}")
    print("Few-shot images:", [s["image"] for s in shots])

    torch.cuda.reset_peak_memory_stats()
    with open(partial_path, "a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(todo), batch_size)):
            batch = todo[i : i + batch_size]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            results = engine.caption_batch([e["image_path"] for e in batch])
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / len(batch)
            for e, r in zip(batch, results):
                rec = {"image": e["image"], **r, "seconds": dt, "references": e["captions"]}
                done[e["image"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

    records = [done[e["image"]] for e in test]
    total = sum(r["seconds"] for r in records)
    meta = {
        "mode": mode, "language": args.lang, "samples": len(records), "batch_size": batch_size,
        "total_seconds": round(total, 1), "seconds_per_sample": round(total / len(records), 4),
        "prompt_tokens": engine.prompt_tokens,
        "avg_generated_tokens": round(sum(r["generated_tokens"] for r in records) / len(records), 2),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "few_shot_images": [s["image"] for s in shots], "prompt": engine.prompt,
        "generation": cfg["generation"], "seed": cfg["seed"],
    }
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "records": records}, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: meta[k] for k in ("samples", "total_seconds", "seconds_per_sample")}))
    print("Saved", final_path)


if __name__ == "__main__":
    main()
