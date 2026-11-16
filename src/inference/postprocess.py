"""Caption post-processing, following Listing 2 of the thesis. Pure text, no model needed."""

SPECIAL_TOKENS = ["<|endofchunk|>", "<image>", "\n"]
TEMPLATE_LEAKS = ["Output:", "German description:", "French description:"]


def clean_caption(raw):
    caption = raw.strip()
    # Truncate at the first special token or leaked prompt template.
    for marker in SPECIAL_TOKENS + TEMPLATE_LEAKS:
        if marker in caption:
            caption = caption[: caption.index(marker)].strip()
    # Self-repetition: first half equals second half (+/- 5 characters around the midpoint).
    if len(caption) > 10:
        half = len(caption) // 2
        for i in range(half - 5, half + 5):
            if i > 0 and caption[:i].strip() == caption[i : 2 * i].strip():
                caption = caption[:i].strip()
                break
    # Trim trailing ellipses.
    while caption.endswith(".."):
        caption = caption[:-1]
    return caption
