import torch
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel
from .Base import BaseFeatureExtractor, CLIPTextConditionMixin
from torchvision import transforms
import torch.nn.functional as F
from typing import List, Tuple, Dict # <-- 解决 List NameError
class ClipB32FeatureExtractor(CLIPTextConditionMixin, BaseFeatureExtractor):
    def __init__(self, target_layers: List[int] = None):
        super(ClipB32FeatureExtractor, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self._clip_text_repo = "openai/clip-vit-base-patch32"
        self.normalizer = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )
        # 定义需要提取特征的层 (例如：4, 8, 12)。CLIP-B/32 也是 12 层。
        self.target_layers = target_layers if target_layers is not None else [10, 11, 12]
        # 确定用于局部特征对齐的最终层索引
        self.final_layer_idx = self.target_layers[-1]

    def forward(self, x):
        # x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x))
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features


    def global_local_features(self, x):
        # x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x))
        # image_features = self.model.get_image_features(**inputs)
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)

        inputs["pixel_values"] = inputs["pixel_values"]
        outputs = self.model.vision_model(pixel_values=inputs['pixel_values'])
        features = outputs.last_hidden_state
        global_feature = features[:, 0, :]
        global_feature = global_feature / global_feature.norm(dim=1, keepdim=True)
        local_feature = features[:, 1:, :]
        local_feature = local_feature / local_feature.norm(dim=1, keepdim=True)
        # features = self.model.get_image_embedding(inputs["pixel_values"])
        # global_feature = features[:, 0, :]
        # local_feature = features[:, 1:, :]
        return global_feature, local_feature


    def get_multilayer_features(self, x, num_layers=3, eps: float = 1e-8):
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"]

        # Use output_hidden_states=True to get all layers
        outputs = self.model.vision_model(
            pixel_values=inputs['pixel_values'],
            output_hidden_states=True
        )

        all_hidden_states = outputs.hidden_states
        selected_layers = all_hidden_states[-num_layers:]

        features_list = []
        for layer_output in selected_layers:
            # layer_output: (B, N+1, D)

            global_feature = layer_output[:, 0, :]
            global_feature = global_feature / (global_feature.norm(dim=1, keepdim=True) + eps)

            local_feature = layer_output[:, 1:, :]
            local_feature = local_feature / (local_feature.norm(dim=2, keepdim=True) + eps)

            # Calculate patch importance using cosine similarity with global feature
            patch_sim = F.cosine_similarity(
                local_feature,
                global_feature.unsqueeze(1),
                dim=-1
            )
            patch_sim = torch.clamp(patch_sim, min=0.0)
            patch_importance = patch_sim / (patch_sim.sum(dim=-1, keepdim=True) + eps)

            features_list.append((global_feature, local_feature, patch_importance))

        return features_list

    def multi_layer_features(self, x, eps: float = 1e-8):
        """
        实现分层全局特征提取和最后一层局部特征提取。

        返回：
            global_features_list: (List[Tensor]): 多个指定层的全局特征 [ (B, D), (B, D), ... ]
            final_local_feature:  (Tensor): 仅最后一层的局部特征 (B, N, D)
        """
        inputs = dict(pixel_values=self.normalizer(x))

        # 关键：设置 output_hidden_states=True 来获取所有中间层的输出
        outputs = self.model.vision_model(
            pixel_values=inputs['pixel_values'],
            output_hidden_states=True
        )

        # outputs.hidden_states 是一个包含 13 个元素的元组 (输入嵌入 + 12 个 Transformer block 的输出)
        hidden_states = outputs.hidden_states

        global_features_list = []
        local_feature_list = []
        for layer_idx in self.target_layers:
            # 确保索引在有效范围内
            if layer_idx < 0 or layer_idx >= len(hidden_states):
                continue

            features = hidden_states[layer_idx]  # (B, 1+N, D)

            global_feature = features[:, 0, :]  # [CLS] token (B, D)
            local_feature = features[:, 1:, :]  # Patch tokens (B, N, D)

            # 归一化并保存全局特征
            global_feature = global_feature / (global_feature.norm(dim=1, keepdim=True) + eps)
            global_features_list.append(global_feature)

            # 仅保存最终层的局部特征

            local_feature = local_feature / (local_feature.norm(dim=2, keepdim=True) + eps)
            local_feature_list.append(local_feature)
        # if final_local_feature is None:
        #     raise ValueError(
        #         f"Final layer index {self.final_layer_idx} not found in hidden states. Check self.target_layers.")

        # 返回多层全局特征列表、最后一层局部特征张量
        return global_features_list, local_feature_list

    def global_local_features_with_attn(self, x, eps: float = 1e-8):
        """
        这里的 attn 不是 transformer 自带的 attention，
        而是用 global CLS 与每个 patch 的余弦相似度来近似“重要性”。

        返回：
            global_feature: (B, D)
            local_feature:  (B, N, D)
            patch_importance: (B, N)  # 类似 attention，用来做筛选/加权
        """
        # 先复用原来的处理逻辑
        inputs = dict(pixel_values=self.normalizer(x))
        outputs = self.model.vision_model(pixel_values=inputs['pixel_values'])
        features = outputs.last_hidden_state  # (B, 1+N, D)

        global_feature = features[:, 0, :]  # (B, D)
        global_feature = global_feature / (global_feature.norm(dim=1, keepdim=True) + eps)

        local_feature = features[:, 1:, :]  # (B, N, D)
        local_feature = local_feature / (local_feature.norm(dim=2, keepdim=True) + eps)

        # -------- 关键：用 CLS–patch 相似度当“注意力” ----------
        # cosine_similarity: (B, N, D) vs (B, 1, D) -> (B, N)
        patch_sim = F.cosine_similarity(
            local_feature,
            global_feature.unsqueeze(1),  # (B, 1, D)
            dim=-1
        )  # (B, N)

        # 为了稳定，只用正的部分（负相似度直接抹掉）
        patch_sim = torch.clamp(patch_sim, min=0.0)

        # 对每一张图内部做归一化，得到类似 attention 的权重
        patch_importance = patch_sim / (patch_sim.sum(dim=-1, keepdim=True) + eps)  # (B, N)

        return global_feature, local_feature, patch_importance


class ClipB32FeatureExtractorOT(BaseFeatureExtractor):
    def __init__(self):
        super(ClipB32FeatureExtractorOT, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.normalizer = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
                transforms.CenterCrop(224),
                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                # CLIP imgs mean and std.
            ]
        )

    def forward(self, x):
        x = torch.clamp(x, min=0, max=1)
        inputs = dict(pixel_values=self.normalizer(x))
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        features = self.model.get_image_embedding(inputs["pixel_values"])
        global_feature = features[:,0,:]
        local_feature = features[:,1:,:]
        return global_feature, local_feature










