import torch
from torch import nn, Tensor
from abc import abstractmethod
from typing import List, Any, Callable, Dict
from kmeans_pytorch import kmeans
import torch.nn.functional as F
from torchvision import transforms as T
import sys
import os
from contextlib import contextmanager
import numpy as np
@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# =============================================================================
# 局部分支的"token -> K 个局部描述子"归约 (Clustering-DET 分析实验)
# =============================================================================
# 方法的局部支路是: N 个 patch token --归约--> K 个局部描述子 --> K x K 局部 OT。
# 本工作用特征空间 K-Means 做这一步归约。要证明"K-Means 这一步是必要的",
# 就得换掉、且只换掉这一步, 其余(K、OT、loss、集成加权、增强)逐字段不动。
#
#   kmeans (ours) : 特征空间 K-Means 质心 —— 语义分组
#   random        : 均匀随机抽 K 个 token 当"中心" —— 完全不分组
#   grid          : 把 patch 网格按行切成 K 条连续横带, 各带内均值池化 —— 固定几何分组,
#                   不看特征
#   all           : 不归约, N 个 token 全上, 走满 N x N OT —— 检验"归约"本身是否有必要
#
# RNG 对齐: kmeans_pytorch.initialize() 每次调用恰好抽一次
# np.random.choice(N, K, replace=False)(全局 numpy 流)。三个替代分支都照抽一次
# (random 用它, grid/all 抽完丢掉), 所以全局 numpy 流各档逐次对齐。
# RandomResizedCrop 走的是 torch 流, kmeans 只在"某簇变空"这一罕见分支上碰
# torch.randint —— 除此之外各档的裁剪逐比特相同(见 clu_check_twin.sh)。
CLUSTER_METHODS = ("kmeans", "random", "grid", "all")


def local_descriptors(embedding_img: Tensor, num_clusters: int, device,
                      method: str = "kmeans") -> Tensor:
    """(N, D) patch token -> (K, D) 局部描述子。method="kmeans" 时与补丁前逐比特一致。"""
    method = (method or "kmeans").lower()
    if method not in CLUSTER_METHODS:
        raise ValueError(f"unknown cluster method: {method}")

    if method == "kmeans":
        if int(num_clusters) == 1:
            # K=1: 所有 token 必然落进同一簇, 中心就是全体均值 —— 这是 kmeans 的不动点,
            # 与初始化无关, 一次迭代即收敛。之所以不进库: 第三方 kmeans_pytorch 在
            # num_clusters=1 时距离矩阵退化成一维, argmin(dis, dim=1) 会 IndexError。
            # 仍把 initialize() 本该消耗的那一次 np.random.choice 抽掉, 让全局 numpy 流
            # 与其它 K 逐次对齐(同 grid/all 两档的做法), 这样各 K 的裁剪仍可配对。
            np.random.choice(int(embedding_img.shape[0]), 1, replace=False)
            return embedding_img.mean(dim=0, keepdim=True).to(device)
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=num_clusters,
                distance='euclidean',
                device=device,
            )
        return cluster_center.to(device)

    # 以下三档: 先把 kmeans 本该消耗的那次 numpy 抽样抽掉, 保证 RNG 流不错位。
    n = int(embedding_img.shape[0])
    k = min(int(num_clusters), n)
    idx = np.random.choice(n, k, replace=False)

    if method == "random":
        # 随机 K 个 token: 有 K 个局部向量、有 K x K OT, 唯独没有"分组"这一步。
        sel = torch.as_tensor(np.sort(idx), device=embedding_img.device, dtype=torch.long)
        return embedding_img.index_select(0, sel).to(device)

    if method == "all":
        # 不归约: K = N, 局部 OT 变成满分辨率 token 级传输。
        return embedding_img.to(device)

    # grid: 按 token 序(行主序)切成 K 条连续横带, 带内均值池化。
    # 用 floor(i*K/N) 分带, 对任意 N(不必是完全平方数)都成立, 且各带非空。
    pos = torch.arange(n, device=embedding_img.device)
    band = (pos * k) // n                                    # (N,) in [0, k)
    onehot = F.one_hot(band, num_classes=k).to(embedding_img.dtype)   # (N, k)
    cnt = onehot.sum(dim=0).clamp(min=1.0).unsqueeze(1)      # (k, 1)
    centers = (onehot.t() @ embedding_img) / cnt             # (k, D)
    return centers.to(device)


class BaseFeatureExtractor(nn.Module):
    # Surrogates that expose a joint image-text space (CLIP family) set this to
    # True via CLIPTextConditionMixin. Non-CLIP surrogates (InternVL3, DINOv2)
    # keep it False, so instruction conditioning falls back to uniform mass.
    supports_text = False

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass


class CLIPTextConditionMixin:
    """Joint-space text encoding + patch projection for CLIP-family extractors.

    Enables instruction-conditioned alignment (IC-SOT): patch tokens are
    projected into CLIP's joint image-text space (MaskCLIP style) so their
    cosine similarity to an encoded instruction/answer is meaningful.

    Host class must provide:
      - self.model:           a transformers CLIPModel (vision + text + projections)
      - self.normalizer:      image preprocessing transform (raw [0,255] -> model input)
      - self._clip_text_repo: HF repo id used to load a tokenizer if none exists
    """

    supports_text = True

    def _clip_device(self):
        return next(self.model.parameters()).device

    def _ensure_tokenizer(self):
        tok = getattr(self, "_clip_tokenizer", None)
        if tok is None:
            proc = getattr(self, "processor", None)
            if proc is not None and getattr(proc, "tokenizer", None) is not None:
                tok = proc.tokenizer
            else:
                from transformers import CLIPTokenizer
                tok = CLIPTokenizer.from_pretrained(self._clip_text_repo)
            self._clip_tokenizer = tok
        return tok

    @torch.no_grad()
    def text_features(self, texts) -> Tensor:
        """Encode instruction/answer text into a single normalized joint embedding.

        Multiple strings (e.g. several target captions) are mean-pooled then
        renormalized. Returns shape (Dj,).
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = [t for t in texts if t]
        if len(texts) == 0:
            texts = [""]
        tok = self._ensure_tokenizer()
        inputs = tok(texts, return_tensors="pt", padding=True,
                     truncation=True, max_length=77)
        inputs = {k: v.to(self._clip_device()) for k, v in inputs.items()}
        out = self.model.get_text_features(**inputs)        # (T, Dj) or output obj
        # transformers >=5 returns a text-model output object instead of the
        # projected joint embedding. Project pooler_output through text_projection
        # so it lands in the same joint space as patch_features_joint (MaskCLIP);
        # transformers <5 already returns the projected (T, Dj) tensor directly.
        if isinstance(out, Tensor):
            feats = out
        else:
            feats = self.model.text_projection(out.pooler_output)
        feats = F.normalize(feats, dim=-1)
        emb = F.normalize(feats.mean(dim=0), dim=0)         # (Dj,)
        return emb

    def patch_features_joint(self, x: Tensor) -> Tensor:
        """Project patch tokens into the joint image-text space (MaskCLIP style).

        Returns shape (N, Dj), L2-normalized, aligned 1:1 with the patch tokens
        used for k-means clustering (same vision_model, same ordering).
        """
        pixel_values = self.normalizer(x)
        vm = self.model.vision_model
        outputs = vm(pixel_values=pixel_values)
        patches = outputs.last_hidden_state[:, 1:, :]       # (B, N, Dv)
        patches = vm.post_layernorm(patches)
        patches = self.model.visual_projection(patches)     # (B, N, Dj)
        patches = F.normalize(patches, dim=-1)
        return patches[0]                                   # (N, Dj)


class EnsembleFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        # features = []
        # for model in self.extractors:
        #     features.append(model(x).squeeze())
        # features = torch.cat(features, dim=0)
        features = {}  # 不拼接，改为字典存储
        for i, model in enumerate(self.extractors):
            features[i] = model(x).squeeze()
        return features

class EnsembleFeatureExtractor_ot(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor],cluster_number=5,
                 cluster_method="kmeans"):
        super(EnsembleFeatureExtractor_ot, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.cluster_number = cluster_number
        # token->K 归约方式(Clustering-DET 分析)。"kmeans" 为默认, 与补丁前逐比特一致。
        self.cluster_method = cluster_method

    def forward(self, x: Tensor) -> Tensor:
        # features = []
        # for model in self.extractors:
        #     features.append(model(x).squeeze())
        # features = torch.cat(features, dim=0)
        features = {}  # 不拼接，改为字典存储
        features_local = {}
        for i, model in enumerate(self.extractors):
            # features[i] = model(x).squeeze()
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            features[i] = x_tensor.squeeze()
            cluster_center = self.get_cluster_center(x_embedding[0],x.device).unsqueeze(0)
            features_local[i]=cluster_center

        return features,features_local

    def get_cluster_center(self, embedding_img,device):
        return local_descriptors(embedding_img, self.cluster_number, device,
                                 getattr(self, "cluster_method", "kmeans"))

class EnsembleFeatureExtractor_ot3(BaseFeatureExtractor):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureExtractor_ot3, self).__init__()
        self.extractors = nn.ModuleList(extractors)

    def forward(self, x: Tensor) -> Tensor:
        # features = []
        # for model in self.extractors:
        #     features.append(model(x).squeeze())
        # features = torch.cat(features, dim=0)
        features = {}  # 不拼接，改为字典存储
        features_local = {}
        for i, model in enumerate(self.extractors):
            # features[i] = model(x).squeeze()
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            features[i] = x_tensor.squeeze()
            cluster_center = self.get_cluster_center(x_embedding[0],x.device).unsqueeze(0)
            features_local[i]=cluster_center

        return features,features_local

    def get_cluster_center(self, embedding_img,device):
        # self.setup_seed(20)
        # for i in range(10):
        # np.random.seed(20)
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=3,
                distance='euclidean',
                device=device
            )
            cluster_center = cluster_center.to(device)
        # print(cluster_ids_x)
        return cluster_center


class EnsembleFeatureLoss(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []

    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        for model in self.extractors:
            self.ground_truth.append(model(x).to(x.device))

    def __call__(self, feature_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        for index, model in enumerate(self.extractors):
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            loss += torch.mean(torch.sum(feature * gt, dim=1))
            
        loss = loss / len(self.extractors)

        return loss


class EnsembleFeatureLoss_OT(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss_OT, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []



    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)



    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss = 0
        loss_local=0
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            # print("gt_local",gt_local.shape)
            # print("feature_local", feature_local.shape)
            loss_local += self.OT(gt_local, feature_local) * 2

            loss += torch.mean(torch.sum(feature * gt, dim=1))

        loss = loss / len(self.extractors)
        loss_local=loss_local / len(self.extractors)

        # print("loss",loss)
        # print("loss_local", loss_local)
        loss=loss+loss_local*0.1
        return loss

    def get_cluster_center(self, embedding_img):
        # self.setup_seed(20)
        # for i in range(10):
        # np.random.seed(20)
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=5,
                distance='euclidean',
                device=embedding_img.device,
            )
            cluster_center = cluster_center.to(embedding_img.device)
        # print(cluster_ids_x)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        # print(src_dis.shape, tgt_dis.shape)
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T


class EnsembleFeatureLoss_OT_Auto(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        super(EnsembleFeatureLoss_OT_Auto, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []



    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)



    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            # print("gt_local",gt_local.shape)
            # print("feature_local", feature_local.shape)
            local_loss = self.OT(gt_local, feature_local) * 2
            feat_loss = torch.mean(torch.sum(feature * gt, dim=1))

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.1 * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        # 初始化 previous_loss_list（首次）
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
            # 归一化 softmax 计算动态权重
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K  # 可选：缩放为 K


        # 初始化 previous_loss_list（首次）
        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        # 加权总损失
        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img):
        # self.setup_seed(20)
        # for i in range(10):
        # np.random.seed(20)
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=5,
                distance='euclidean',
                device=embedding_img.device,
            )
            cluster_center = cluster_center.to(embedding_img.device)
        # print(cluster_ids_x)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        # print(src_dis.shape, tgt_dis.shape)
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T
    
class EnsembleFeatureLoss_OT_foa_attack(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5,
                 uot_reg_m=None, uot_normalize=False, eps=0.1, local_weight=0.2,
                 uot_adaptive=False, uot_adaptive_gamma=1.0, uot_adaptive_norm=True,
                 dynamic_ensemble=True, ensemble_weighting="auto",
                 ensemble_weight_seed=20260817, cluster_method="kmeans"):
        super(EnsembleFeatureLoss_OT_foa_attack, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number
        # token->K 归约方式(Clustering-DET 分析); 必须与 extractor 侧同档, 否则
        # target 侧和 adv 侧的局部描述子不是同一种东西, OT 两边不可比。
        self.cluster_method = cluster_method
        # 局部 OT 项在 `global + local_weight*local` 中的权重(倒 U, 最优≈1.0)。
        self.local_weight = local_weight
        # 局部簇级不平衡 OT。uot_reg_m=None -> 退回平衡 FOA。
        self.uot_reg_m = uot_reg_m
        self.uot_normalize = uot_normalize
        # adaptive(贡献②): adv 侧逐簇按置信度调制松弛; gamma=0 即普通 UOT; norm=均值归一化。
        self.uot_adaptive = uot_adaptive
        self.uot_adaptive_gamma = uot_adaptive_gamma
        self.uot_adaptive_norm = uot_adaptive_norm
        self.eps = eps              # 熵温度(固定)
        # 集成动态加权(贡献③)。False -> 等权求和, 用于 w/o dynamic-ensemble 消融。
        self.dynamic_ensemble = dynamic_ensemble
        # 加权策略分析实验(DynEnsemble-DET)。四档共用同一套 total_losses, 只换 w 的算法,
        # 且一律缩放到 sum(w)=K —— 否则 loss 整体尺度不同会和 alpha 的步长效果混淆。
        #   auto      : 兼容老 config, 由 dynamic_ensemble 决定 dynamic / equal
        #   equal     : w_i = 1
        #   random    : w ~ K·Dirichlet(1,...,1), 每步重采样(与 dynamic 同频率)
        #   loss_prop : w_i = K·L_i / Σ_j L_j, 权重正比于当前对齐分数(L 越大=对得越准)
        #   dynamic   : w = K·softmax(L_i / L_i^prev), 按"本步改善倍率"重分配(本工作)
        mode = str(ensemble_weighting or "auto").lower()
        if mode == "auto":
            mode = "dynamic" if dynamic_ensemble else "equal"
        if mode not in ("equal", "random", "loss_prop", "dynamic"):
            raise ValueError(f"unknown ensemble_weighting: {ensemble_weighting}")
        self.ensemble_weighting = mode
        # 随机档用自带的 RandomState, 不碰全局 np.random —— 全局流被挪一位会让
        # cluster/crop 的采样整体错位, 那样这一档就不只是"换了权重"了。
        self._w_rng = np.random.RandomState(int(ensemble_weight_seed)) if mode == "random" else None


    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feature_local = feature_local_dict[index].squeeze(0)
            # print("gt_local",gt_local.shape)
            # print("feature_local", feature_local.shape)
            # local OT (K x K cluster transport): the only place UOT is meaningful.
            # w/o-OT 消融(local_weight=0): 整块跳过。不跳的话既白算一遍 Sinkhorn, 又会在
            # OT 返回 None(裁剪退化 -> NaN 传输计划)时于 `0.0 * None` 处抛 TypeError。
            # local_weight != 0 时走原路径, 与补丁前逐比特一致。
            local_loss = self.OT(gt_local, feature_local, reg_m=self.uot_reg_m,
                                 eps=self.eps, normalize=self.uot_normalize,
                                 adaptive=self.uot_adaptive) if self.local_weight != 0.0 else None

            # feat_loss = torch.mean(torch.sum(feature * gt, dim=1))
            # global OT is a 1x1 transport -> UOT is a no-op, keep it balanced.
            feat_loss = self.OT(gt, feature, eps=self.eps)

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] if loss_local_list[i] is None
            else loss_list[i] + self.local_weight * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        K = len(self.extractors)

        if self.ensemble_weighting == "equal":
            # w/o dynamic-ensemble 消融: 等权求和(权重和仍为 K, 与动态分支同量级, 使两者
            # 的有效学习率可比 —— 否则 loss 尺度差 K 倍会和 alpha 的效果混淆)。
            weights_softmax = np.ones(K)
        elif self.ensemble_weighting == "random":
            # 随机对照: 每步从单纯形上均匀采一个点再乘 K。与 dynamic 同样是"每步重分配、
            # 和恒为 K", 唯一区别是分配依据与 loss 无关 —— 用来证明增益来自"按改善速度分",
            # 而不是"权重在动"本身。
            w = self._w_rng.dirichlet(np.ones(K))
            weights_softmax = w * K
        elif self.ensemble_weighting == "loss_prop":
            # loss 正比对照: w_i ∝ L_i。这里 L 是被最大化的对齐分数, 所以该档把权重压给
            # "已经对得最准"的 backbone(与 dynamic 的"按改善倍率分"是不同的方向)。
            vals = np.array([float(l.item()) for l in total_losses])
            pos = np.clip(vals, 0.0, None)
            s = pos.sum()
            # 全非正时无从按比例分(会出 0/0 或负权翻转梯度方向), 退回等权。
            weights_softmax = np.ones(K) if s <= 1e-8 else pos / s * K
        else:
            # 初始化 previous_loss_list（首次）
            if len(self.previous_loss_list) == 0:
                self.previous_loss_list = [l.detach() for l in total_losses]

            weights = []
            for i in range(len(self.extractors)):
                ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
                weights.append(ratio)
                # 归一化 softmax 计算动态权重

            T = 1.0
            weights_np = np.array(weights)
            weights_softmax = np.exp(weights_np / T)
            weights_softmax /= np.sum(weights_softmax)
            weights_softmax *= K  # 可选：缩放为 K

            # 初始化 previous_loss_list（首次）
            for i in range(len(self.extractors)):
                self.previous_loss_list[i] = total_losses[i].detach()

        # 加权总损失
        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        return local_descriptors(embedding_img, self.cluster_number, device,
                                 getattr(self, "cluster_method", "kmeans"))

    def OT(self, src_dis, tgt_dis, reg_m=None, eps=0.1, normalize=False, adaptive=False):
        """局部 cluster 级最优传输对齐(贡献①②)。

        src_dis = target 图簇中心(r/u 侧), tgt_dis = adv 图簇中心(c/v 侧)。
        reg_m=None -> 平衡 OT(原始 FOA); reg_m=tau -> 不平衡 OT(松弛边际,贡献②);
        adaptive=True -> 在 adv 侧按置信度自适应调制松弛(见 Sinkhorn)。
        """
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / eps)              # 传输核(eps=熵温度)
            T = self.Sinkhorn(KK, xx, yy, reg_m=reg_m, eps=eps, adaptive=adaptive, sim=sim)
        if torch.isnan(T).any():
            return None
        # 不平衡 OT 下传输计划不再归一, 重归一使局部项与全局项量级可比。
        if normalize:
            T = T / (T.sum() + 1e-8)
        loss = torch.sum(torch.sum(T * sim, dim=(0, 1)))
        return loss

    def Sinkhorn(self, K, u, v, reg_m=None, eps=0.1, adaptive=False, sim=None):
        """不平衡 Sinkhorn。松弛系数 fi=tau/(tau+eps): reg_m=None 时 fi=1(平衡 OT)。

        adaptive(贡献②, 最终版): adv 侧逐簇按"与 target 的最佳匹配相似度"conf 调制:
          conf 均值归一化 -> rm = reg_m·[(1-γ)+γ·conf] -> fi_c = rm/(rm+eps)。
          γ=0 精确退回普通 UOT(UOT 是 adaptive-UOT 的特例, 保证不劣); γ=1 全自适应。
          均值归一化把"整体变松"改为"绕 reg_m 重分配"(高置信更严/低置信更松)。
        """
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        if reg_m is None:
            fi_r = torch.ones_like(u); fi_c = torch.ones_like(v)        # 平衡 OT
        else:
            base = reg_m / (reg_m + eps)
            fi_r = torch.full_like(u, base)                            # target 侧松弛
            if adaptive and sim is not None:                           # adv 侧自适应调制
                conf = sim.max(dim=0).values.clamp(min=0.0)
                if getattr(self, "uot_adaptive_norm", True):
                    conf = conf / (conf.mean() + 1e-8)                 # 均值=1, 绕 reg_m 重分配
                g = float(getattr(self, "uot_adaptive_gamma", 1.0))    # γ=0 即普通 UOT
                rm = reg_m * ((1.0 - g) + g * conf).clamp(min=0.0)
                fi_c = rm / (rm + eps)
            else:
                fi_c = torch.full_like(v, base)
        for i in range(100):
            r0 = r
            r = (u / (K @ c.unsqueeze(-1)).squeeze(-1)) ** fi_r
            c = (v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)) ** fi_c
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T

class EnsembleFeatureLoss_OT_ablation_wo_global(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_global, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number


    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            gt_local = self.ground_truth_local[index].squeeze(0)
            gt = self.ground_truth[index]
            feature = feature_dict[index]
            feature_local = feature_local_dict[index].squeeze(0)
            local_loss = self.OT(gt_local, feature_local)
            feat_loss = torch.mean(torch.sum(feature * gt, dim=1))

            loss_list.append(feat_loss)
            loss_local_list.append(local_loss)

        total_losses = [
            loss_list[i] + 0.2 * loss_local_list[i]
            for i in range(len(self.extractors))
        ]
        # 初始化 previous_loss_list（首次）
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
            # 归一化 softmax 计算动态权重
        
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K  # 可选：缩放为 K


        # 初始化 previous_loss_list（首次）
        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        # 加权总损失
        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number,  # TODO
                distance='euclidean',
                device=device,
            )
            cluster_center = cluster_center.to(device)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        # print(src_dis.shape, tgt_dis.shape)
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T
    
class EnsembleFeatureLoss_OT_ablation_wo_local(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_local, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number


    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feat_loss = self.OT(gt,feature)

            loss_list.append(feat_loss)

        total_losses = [
            loss_list[i]
            for i in range(len(self.extractors))
        ]
        # 初始化 previous_loss_list（首次）
        if len(self.previous_loss_list) == 0:
            self.previous_loss_list = [l.detach() for l in total_losses]

        weights = []
        for i in range(len(self.extractors)):
            ratio = total_losses[i].item() / (self.previous_loss_list[i].item() + 1e-8)
            weights.append(ratio)
            # 归一化 softmax 计算动态权重
        
        T = 1.0
        K = len(weights)
        weights_np = np.array(weights)
        weights_softmax = np.exp(weights_np / T)
        weights_softmax /= np.sum(weights_softmax)
        weights_softmax *= K  # 可选：缩放为 K


        # 初始化 previous_loss_list（首次）
        for i in range(len(self.extractors)):
            self.previous_loss_list[i] = total_losses[i].detach()

        # 加权总损失
        total_loss = sum(
            weights_softmax[i] * total_losses[i]
            for i in range(len(self.extractors))
        )
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number,  # TODO
                distance='euclidean',
                device=device,
            )
            cluster_center = cluster_center.to(device)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        # print(src_dis.shape, tgt_dis.shape)
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T
    
class EnsembleFeatureLoss_OT_ablation_wo_dynamic(nn.Module):
    def __init__(self, extractors: List[BaseFeatureExtractor], cluster_number=5):
        super(EnsembleFeatureLoss_OT_ablation_wo_dynamic, self).__init__()
        self.extractors = nn.ModuleList(extractors)
        self.ground_truth = []
        self.ground_truth_local = []
        self.previous_loss_list=[]
        self.previous_loss_local_list = []
        self.cluster_number = cluster_number


    @torch.no_grad()
    def set_ground_truth(self, x: Tensor):
        self.ground_truth.clear()
        self.ground_truth_local.clear()
        for model in self.extractors:
            x_tensor, x_embedding = model.global_local_features(x.to(x.device))
            x_embedding = x_embedding.squeeze(0)
            cluster_center = self.get_cluster_center(x_embedding,x.device).unsqueeze(0)
            self.ground_truth.append(x_tensor)
            self.ground_truth_local.append(cluster_center)

    def __call__(self, feature_dict: Dict[int, Tensor],feature_local_dict: Dict[int, Tensor], y: Any = None) -> Tensor:
        loss_list = []
        loss_local_list = []
        for index, model in enumerate(self.extractors):
            
            gt = self.ground_truth[index]
            feature = feature_dict[index].unsqueeze(0)
            feat_loss = self.OT(gt,feature)

            loss_list.append(feat_loss)

        total_losses = [
            loss_list[i]
            for i in range(len(self.extractors))
        ]

        # 加权总损失
        total_loss = sum(total_losses)/len(total_losses)
        return total_loss

    def get_cluster_center(self, embedding_img,device):
        with suppress_output():
            cluster_ids_x, cluster_center = kmeans(
                X=embedding_img,
                num_clusters=self.cluster_number,  # TODO
                distance='euclidean',
                device=device,
            )
            cluster_center = cluster_center.to(device)
        return cluster_center

    def OT(self, src_dis, tgt_dis):
        # print(src_dis.shape, tgt_dis.shape)
        src_dis_norm = F.normalize(src_dis, dim=1)
        tgt_dis_norm = F.normalize(tgt_dis, dim=1)
        sim = torch.einsum('md,nd->mn', src_dis_norm, tgt_dis_norm).contiguous()
        wdist = 1 - sim
        xx = torch.full((src_dis.shape[0],), 1.0 / src_dis.shape[0], dtype=sim.dtype, device=sim.device)
        yy = torch.full((tgt_dis.shape[0],), 1.0 / tgt_dis.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            KK = torch.exp(-wdist / 0.1)
            T = self.Sinkhorn(KK, xx, yy)
        if torch.isnan(T).any():
            return None
        sim_op = torch.sum(T * sim, dim=(0, 1))
        loss = torch.sum(sim_op)
        return loss

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = 1e-2
        for i in range(100):
            r0 = r
            r = u / (K @ c.unsqueeze(-1)).squeeze(-1)
            c = v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break
        T = torch.outer(r, c) * K
        return T