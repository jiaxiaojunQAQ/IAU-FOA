import torch
from transformers import AutoModel
from torchvision import transforms

from .Base import BaseFeatureExtractor


class DINOv2FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(DINOv2FeatureExtractor, self).__init__()
        self.model = AutoModel.from_pretrained("facebook/dinov2-base")
        self.normalizer = transforms.Compose(
            [
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def forward(self, x):
        pixel_values = self.normalizer(x)
        outputs = self.model(pixel_values=pixel_values)
        last_hidden_states = outputs.last_hidden_state
        image_features = last_hidden_states[:, 0, :]
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def global_local_features(self, x):
        pixel_values = self.normalizer(x)
        outputs = self.model(pixel_values=pixel_values)
        features = outputs.last_hidden_state

        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = features[:, 1:, :]
        local_feature = local_feature / local_feature.norm(dim=1, keepdim=True)
        return global_feature, local_feature
