import json
import random
from pathlib import Path


class Multi30KLoader:
    def __init__(self, data_dir, images_dir):
        self.data_dir = Path(data_dir)
        self.images_dir = Path(images_dir)

    def load_split(self, split):
        with open(self.data_dir / f"{split}.json", encoding="utf-8") as f:
            entries = json.load(f)
        for e in entries:
            e["image_path"] = str(self.images_dir / e["image"])
        return entries

    def sample_few_shot(self, split, n, seed):
        entries = self.load_split(split)
        return random.Random(seed).sample(entries, n)
