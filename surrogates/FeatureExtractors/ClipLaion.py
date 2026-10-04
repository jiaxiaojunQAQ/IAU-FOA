from transformers import CLIPModel

from .Base import BaseFeatureExtractor


class ClipLaionFeatureExtractor(BaseFeatureExtractor):
    """CLIP ViT-G/14 trained on LAION-2B."""

    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("laion/CLIP-ViT-G-14-laion2B-s12B-b42K")

    def tokens(self, x):
        return self.model.vision_model(pixel_values=self.preprocess(x)).last_hidden_state
