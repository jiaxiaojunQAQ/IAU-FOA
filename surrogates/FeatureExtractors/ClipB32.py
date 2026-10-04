from transformers import CLIPModel

from .Base import BaseFeatureExtractor


class ClipB32FeatureExtractor(BaseFeatureExtractor):
    """CLIP ViT-B/32."""

    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def tokens(self, x):
        return self.model.vision_model(pixel_values=self.preprocess(x)).last_hidden_state
