SPECIAL_TOKENS = ["<|endofchunk|>", "<image>", "\n"]
TEMPLATE_LEAKS = ["Output:", "German description:", "French description:"]


def clean_caption(raw):
    caption = raw.strip()
    for marker in SPECIAL_TOKENS + TEMPLATE_LEAKS:
        if marker in caption:
            caption = caption[: caption.index(marker)].strip()
    if len(caption) > 10:
        half = len(caption) // 2
        for i in range(half - 5, half + 5):
            if i > 0 and caption[:i].strip() == caption[i : 2 * i].strip():
                caption = caption[:i].strip()
                break
    while caption.endswith(".."):
        caption = caption[:-1]
    return caption
