import torch
from transformers import AutoModelForImageTextToText

from .Base import BaseFeatureExtractor


class InternVL3_1B_FeatureExtractor(BaseFeatureExtractor):
    """Vision tower of InternVL3-1B (448 x 448 input)."""

    input_size = 448

    def __init__(self):
        super().__init__()
        self.model = AutoModelForImageTextToText.from_pretrained(
            "OpenGVLab/InternVL3-1B-hf", dtype=torch.bfloat16
        )

    def tokens(self, x):
        pixel_values = self.preprocess(x).to(self.model.device, dtype=torch.bfloat16)
        return self.model.model.vision_tower(pixel_values=pixel_values).last_hidden_state
