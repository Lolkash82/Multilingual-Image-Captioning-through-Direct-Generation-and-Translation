import argparse
import csv
import json
from pathlib import Path

from ..utils.config import load_config

MODES = {
    "en": ("captions", "en"),
    "direct_de": ("captions", "de"),
    "translate_de": ("translations", "de"),
    "direct_fr": ("captions", "fr"),
    "translate_fr": ("translations", "fr"),
}
NLTK_LANG = {"en": "english", "de": "german", "fr": "french"}


def tokenize(text, lang):
    from nltk import word_tokenize
    try:
        return word_tokenize(text, language=NLTK_LANG[lang])
    except LookupError:
        return text.split()


def meteor(hyps, refs, lang):
    from nltk.translate.meteor_score import meteor_score
    scores = [meteor_score([tokenize(r, lang)], tokenize(h, lang)) for h, r in zip(hyps, refs)]
    return 100 * sum(scores) / len(scores)


def language_accuracy(hyps, lang, seed):
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException
    DetectorFactory.seed = seed
    detected = []
    for h in hyps:
        try:
            detected.append(detect(h))
        except LangDetectException:
            detected.append("none")
    dist = {}
    for d in detected:
        dist[d] = dist.get(d, 0) + 1
    return 100 * dist.get(lang, 0) / len(hyps), dict(sorted(dist.items(), key=lambda kv: -kv[1]))


def evaluate_mode(hyps, refs, lang, cfg, with_bertscore):
    import sacrebleu
    bleu_metric, chrf_metric = sacrebleu.metrics.BLEU(), sacrebleu.metrics.CHRF()
    bleu = bleu_metric.corpus_score(hyps, [refs])
    chrf = chrf_metric.corpus_score(hyps, [refs])
    lang_acc, dist = language_accuracy(hyps, lang, cfg["evaluation"]["langdetect_seed"])
    res = {
        "bleu": bleu.score, "chrf": chrf.score, "meteor": meteor(hyps, refs, lang),
        "lang_acc": lang_acc, "detected_languages": dist,
        "brevity_penalty": bleu.bp,
        "avg_words_hyp": sum(len(h.split()) for h in hyps) / len(hyps),
        "avg_words_ref": sum(len(r.split()) for r in refs) / len(refs),
        "empty_captions": sum(1 for h in hyps if not h.strip()),
        "bleu_signature": str(bleu_metric.get_signature()),
        "chrf_signature": str(chrf_metric.get_signature()),
    }
    if with_bertscore:
        from bert_score import score
        model = cfg["evaluation"]["bertscore_models"][lang]
        _, _, f1 = score(hyps, refs, model_type=model, lang=lang, verbose=False)
        res["bertscore"] = 100 * f1.mean().item()
        res["bertscore_model"] = model
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-bertscore", action="store_true")
    ap.add_argument("--outputs", default=None, help="override outputs directory")
    args = ap.parse_args()
    cfg = load_config(args.config)
    outputs = Path(args.outputs or cfg["paths"]["outputs_dir"])
    metrics_dir = outputs / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for mode, (sub, lang) in MODES.items():
        path = outputs / sub / f"{mode}.json"
        if not path.exists():
            print(f"[skip] {mode}: {path} not found")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        hyps = [r["caption"] for r in data["records"]]
        refs = [r["references"][lang] for r in data["records"]]
        results[mode] = {"samples": len(hyps), "eval_language": lang,
                         **evaluate_mode(hyps, refs, lang, cfg, not args.skip_bertscore),
                         "runtime": {k: data["meta"].get(k) for k in ("total_seconds", "seconds_per_sample")}}
        print(mode, {k: round(v, 2) for k, v in results[mode].items() if isinstance(v, float)})

    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    keys = ["bleu", "chrf", "meteor", "bertscore", "lang_acc"]
    with open(metrics_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "samples", *keys, "seconds_per_sample"])
        for mode, r in results.items():
            w.writerow([mode, r["samples"], *[f"{r[k]:.2f}" if k in r else "" for k in keys],
                        r["runtime"]["seconds_per_sample"]])

    thesis = cfg.get("thesis_table_5_3", {})
    with open(metrics_dir / "comparison_with_thesis.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "metric", "thesis", "reproduced", "difference"])
        for mode, r in results.items():
            for k in keys:
                if k in r and k in thesis.get(mode, {}):
                    w.writerow([mode, k, thesis[mode][k], f"{r[k]:.2f}", f"{r[k] - thesis[mode][k]:+.2f}"])
    print("Saved metrics to", metrics_dir)


if __name__ == "__main__":
    main()
