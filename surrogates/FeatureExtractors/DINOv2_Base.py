from transformers import AutoModel

from .Base import BaseFeatureExtractor


class DINOv2FeatureExtractor(BaseFeatureExtractor):
    """DINOv2 ViT-B/14."""

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("facebook/dinov2-base")

    def tokens(self, x):
        return self.model(pixel_values=self.preprocess(x)).last_hidden_state
