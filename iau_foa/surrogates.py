"""White-box surrogate vision encoders.

Every surrogate maps a pixel-space image (B,3,H,W) in [0,255] to
  global feature: (B, D)     L2-normalised [CLS] token
  local features: (B, N, D)  patch tokens
"""
import torch
from torch import nn
from torchvision import transforms
from transformers import AutoModel, AutoModelForImageTextToText, CLIPModel

from .resize import matmul_resize

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class Surrogate(nn.Module):
    input_size = 224

    def __init__(self):
        super().__init__()
        # All surrogates, DINOv2 included, use the CLIP statistics.
        self.normalize = transforms.Normalize(CLIP_MEAN, CLIP_STD)

    def preprocess(self, x):
        size = (self.input_size, self.input_size)
        x = matmul_resize(x, size, "bicubic", True)
        x = torch.clamp(x, 0.0, 255.0) / 255.0
        return self.normalize(x)

    def tokens(self, x):
        """(B, 1+N, D) last hidden state, [CLS] first."""
        raise NotImplementedError

    def global_local_features(self, x):
        features = self.tokens(x)
        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = features[:, 1:, :]
        local_feature = local_feature / local_feature.norm(dim=1, keepdim=True)
        return global_feature.float(), local_feature.float()


class ClipSurrogate(Surrogate):
    repo = None

    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained(self.repo)

    def tokens(self, x):
        return self.model.vision_model(pixel_values=self.preprocess(x)).last_hidden_state


class ClipB16(ClipSurrogate):
    repo = "openai/clip-vit-base-patch16"


class ClipB32(ClipSurrogate):
    repo = "openai/clip-vit-base-patch32"


class ClipLaion(ClipSurrogate):
    repo = "laion/CLIP-ViT-G-14-laion2B-s12B-b42K"


class InternVL3_1B(Surrogate):
    input_size = 448

    def __init__(self):
        super().__init__()
        self.model = AutoModelForImageTextToText.from_pretrained(
            "OpenGVLab/InternVL3-1B-hf", dtype=torch.bfloat16
        )

    def tokens(self, x):
        pixel_values = self.preprocess(x).to(self.model.device, dtype=torch.bfloat16)
        return self.model.model.vision_tower(pixel_values=pixel_values).last_hidden_state


class DINOv2Base(Surrogate):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("facebook/dinov2-base")

    def tokens(self, x):
        return self.model(pixel_values=self.preprocess(x)).last_hidden_state


SURROGATES = {
    "B16": ClipB16,
    "B32": ClipB32,
    "Laion": ClipLaion,
    "InternVL3_1B": InternVL3_1B,
    "DINOv2_Base": DINOv2Base,
}


def build_surrogates(names, device):
    models = []
    for name in names:
        if name not in SURROGATES:
            raise ValueError(f"unknown surrogate: {name} (available: {list(SURROGATES)})")
        models.append(SURROGATES[name]().eval().to(device).requires_grad_(False))
    return models
