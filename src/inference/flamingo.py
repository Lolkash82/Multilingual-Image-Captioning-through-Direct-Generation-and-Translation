"""OpenFlamingo few-shot captioning wrapper (reproduction of the thesis FlamingoInference class)."""
import torch
from PIL import Image

from .postprocess import clean_caption
from .prompt import build_prompt

class FlamingoInference:
    def __init__(self, cfg, device="cuda"):
        from huggingface_hub import hf_hub_download
        from open_flamingo import create_model_and_transforms

        fc = cfg["flamingo"]
        self.gen = cfg["generation"]
        self.templates = cfg["prompts"]
        self.device = device
        self.bf16 = fc.get("precision", "fp32") == "bf16"

        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=fc["clip_vision_encoder_path"],
            clip_vision_encoder_pretrained=fc["clip_vision_encoder_pretrained"],
            lang_encoder_path=fc["lang_encoder_path"],
            tokenizer_path=fc["tokenizer_path"],
            cross_attn_every_n_layers=fc["cross_attn_every_n_layers"],
        )
        ckpt = hf_hub_download(fc["checkpoint_repo"], fc["checkpoint_file"])
        missing, unexpected = model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}")
        tokenizer.padding_side = "left"
        self.model = model.to(device).eval()
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.prompt = None
        self.shot_tensors = None

    def _load(self, path):
        return self.image_processor(Image.open(path).convert("RGB"))

    def set_few_shot(self, examples, language):
        """Fix the few-shot context once; it is reused for every query image."""
        self.prompt = build_prompt(examples, language, self.templates)
        self.shot_tensors = [self._load(ex["image_path"]) for ex in examples]
        self.prompt_tokens = len(self.tokenizer(self.prompt)["input_ids"])

    @torch.no_grad()
    def caption_batch(self, image_paths):
        """Returns a list of dicts: caption (cleaned), raw, generated_tokens."""
        # vision_x shape: (batch, num_images, frames=1, C, H, W)
        per_query = [torch.stack(self.shot_tensors + [self._load(p)]) for p in image_paths]
        vision_x = torch.stack(per_query).unsqueeze(2).to(self.device)
        # The prompt is identical for every query, so batching introduces no input padding.
        enc = self.tokenizer([self.prompt] * len(image_paths), return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            out = self.model.generate(
                vision_x=vision_x,
                lang_x=input_ids,
                attention_mask=enc["attention_mask"].to(self.device),
                max_new_tokens=self.gen["max_new_tokens"],
                num_beams=self.gen["num_beams"],
                repetition_penalty=self.gen["repetition_penalty"],
            )
        prompt_len = input_ids.shape[1]
        pad_id = self.tokenizer.pad_token_id
        results = []
        for row in out:
            new_tokens = row[prompt_len:]
            raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            n_gen = int((new_tokens != pad_id).sum()) if pad_id is not None else len(new_tokens)
            results.append({"caption": clean_caption(raw), "raw": raw, "generated_tokens": n_gen})
        return results
