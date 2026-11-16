"""Pre-download all models into the Hugging Face cache (run on a node WITH internet access)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nltk  # noqa: E402
from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

from src.utils.config import load_config  # noqa: E402

cfg = load_config()
fc = cfg["flamingo"]
hf_hub_download(fc["checkpoint_repo"], fc["checkpoint_file"])
snapshot_download(fc["lang_encoder_path"])
for name in list(cfg["translation"]["models"].values()) + list(set(cfg["evaluation"]["bertscore_models"].values())):
    snapshot_download(name)
import open_clip  # noqa: E402
open_clip.create_model_and_transforms(fc["clip_vision_encoder_path"], pretrained=fc["clip_vision_encoder_pretrained"])
for pkg in ("wordnet", "omw-1.4", "punkt", "punkt_tab"):
    nltk.download(pkg)
print("All models cached.")
