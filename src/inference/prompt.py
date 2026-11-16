END_OF_CHUNK = "<|endofchunk|>"


def build_prompt(few_shot_examples, language, templates):
    prefix = templates.get(language, templates["en"])
    cap_lang = language if language in ("de", "fr") else "en"
    parts = []
    for ex in few_shot_examples:
        caption = ex["captions"].get(cap_lang, ex["captions"]["en"])
        parts.append(f"<image>{prefix} {caption}{END_OF_CHUNK}")
    parts.append(f"<image>{prefix}")
    return "".join(parts)
