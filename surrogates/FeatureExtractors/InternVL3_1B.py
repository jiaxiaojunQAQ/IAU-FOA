import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer
from torchvision import transforms

from .Base import BaseFeatureExtractor


class InternVL3_1B_FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(InternVL3_1B_FeatureExtractor, self).__init__()
        self.normalizer = transforms.Compose(
            [
                transforms.Resize(
                    (448, 448),
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
        self.processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-1B-hf")
        # transformers 5.x renamed torch_dtype -> dtype.
        self.model = AutoModelForImageTextToText.from_pretrained(
            "OpenGVLab/InternVL3-1B-hf", dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B-hf")

    def forward(self, x):
        pixel_values = self.normalizer(x).to(self.model.device, dtype=torch.bfloat16)
        outputs = self.model.get_image_features(pixel_values=pixel_values)
        image_features = outputs / outputs.norm(dim=1, keepdim=True)
        return image_features.float()

    def global_local_features(self, x):
        pixel_values = self.normalizer(x).to(self.model.device, dtype=torch.bfloat16)
        # transformers 5.x nests vision_tower under InternVLForConditionalGeneration.model
        vision_features = self.model.model.vision_tower(
            pixel_values=pixel_values
        ).last_hidden_state
        global_feature = vision_features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = vision_features[:, 1:, :]
        local_feature = local_feature / local_feature.norm(dim=1, keepdim=True)
        return global_feature.float(), local_feature.float()
