from src.inference.prompt import build_prompt
from src.inference.postprocess import clean_caption

T = {"en": "Output:", "de": "German description:", "fr": "French description:"}
SHOTS = [{"captions": {"en": "A dog.", "de": "Ein Hund.", "fr": "Un chien."}},
         {"captions": {"en": "A cat.", "de": "Eine Katze.", "fr": "Un chat."}}]


def test_prompt_en():
    assert build_prompt(SHOTS, "en", T) == (
        "<image>Output: A dog.<|endofchunk|><image>Output: A cat.<|endofchunk|><image>Output:")


def test_prompt_de_uses_target_language_captions():
    p = build_prompt(SHOTS, "de", T)
    assert p.count("<image>") == 3 and "Ein Hund." in p and p.endswith("<image>German description:")


def test_template_leak_is_cut():
    assert clean_caption("A man wearing a hat. Output: A dog runs.") == "A man wearing a hat."
    assert clean_caption("Ein Mann. German description: Ein Hund") == "Ein Mann."


def test_special_tokens_are_cut():
    assert clean_caption("A man.<|endofchunk|><image>Output: x") == "A man."
    assert clean_caption("A man.\nA dog.") == "A man."


def test_self_repetition_is_removed():
    assert clean_caption("A dog runs on grass. A dog runs on grass.") == "A dog runs on grass."


def test_trailing_ellipsis_and_clean_text():
    assert clean_caption("A man walks...") == "A man walks."
    assert clean_caption("  A woman reads a book.  ") == "A woman reads a book."
