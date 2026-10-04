import torch
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel
from .Base import BaseFeatureExtractor, CLIPTextConditionMixin
from torchvision import transforms


class ClipL336FeatureExtractor(CLIPTextConditionMixin, BaseFeatureExtractor):
    def __init__(self):
        super(ClipL336FeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        self._clip_text_repo = "openai/clip-vit-large-patch14-336"
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(336, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(336),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    def forward(self, x):
        # x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x))
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    def get_multilayer_features(self, x, num_layers=3, eps: float = 1e-8):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        
        outputs = self.model.vision_model(
            pixel_values=inputs['pixel_values'],
            output_hidden_states=True
        )
        
        all_hidden_states = outputs.hidden_states
        selected_layers = all_hidden_states[-num_layers:]
        
        features_list = []
        for layer_output in selected_layers:
            global_feature = layer_output[:, 0, :]
            global_feature = global_feature / (global_feature.norm(dim=1, keepdim=True) + eps)
            
            local_feature = layer_output[:, 1:, :]
            local_feature = local_feature / (local_feature.norm(dim=2, keepdim=True) + eps)
            
            patch_sim = F.cosine_similarity(
                local_feature,
                global_feature.unsqueeze(1),
                dim=-1
            )
            patch_sim = torch.clamp(patch_sim, min=0.0)
            patch_importance = patch_sim / (patch_sim.sum(dim=-1, keepdim=True) + eps)
            
            features_list.append((global_feature, local_feature, patch_importance))
            
        return features_list
