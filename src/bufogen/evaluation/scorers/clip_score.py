import re
from pathlib import Path

import weave
from pydantic import PrivateAttr


def normalized_caption(prompt: str) -> str:
    prompt = re.sub(r"\b(?:bufo|bufos|CoreWeave)\b", "", prompt, flags=re.I)
    return "a green cartoon frog, " + re.sub(r"\s+", " ", prompt).strip()


class ClipScore(weave.Scorer):
    model_id: str = "openai/clip-vit-base-patch32"
    _model: object = PrivateAttr(default=None)
    _processor: object = PrivateAttr(default=None)

    @weave.op()
    def score(self, output: dict) -> dict:
        if "clip_score" in output.get("scores", {}):
            return output["scores"]["clip_score"]
        if not output.get("valid") or not output.get("png_path"):
            return {"similarity": 0.0}
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        if self._model is None:
            self._model = CLIPModel.from_pretrained(self.model_id).eval()
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
        with Image.open(Path(output["png_path"])) as image:
            inputs = self._processor(
                text=[normalized_caption(output["prompt"])],
                images=image.convert("RGB"),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        with torch.inference_mode():
            result = self._model(**inputs)
        similarity = (result.image_embeds * result.text_embeds).sum().item()
        return {"similarity": round(float(similarity), 5)}
