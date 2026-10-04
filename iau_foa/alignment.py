"""Dual-granularity feature alignment over a surrogate ensemble.

Per surrogate, the alignment score between the adversarial and the target image is

    L_i = global_i + local_weight * local_i

  global_i : cosine similarity of the [CLS] features
  local_i  : adaptive unbalanced OT between the K-Means centres of the patch tokens

and the ensemble objective is sum_i w_i * L_i with dynamic weights w_i.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .kmeans import kmeans


def cluster_centers(tokens, num_centers):
    """(N, D) patch tokens -> (K, D) K-Means centres. Differentiable w.r.t. tokens."""
    if int(num_centers) == 1:
        # The single-cluster fixed point is the mean. Draw the sample the K-Means
        # initialisation would have drawn, to keep the RNG stream aligned across K.
        np.random.choice(int(tokens.shape[0]), 1, replace=False)
        return tokens.mean(dim=0, keepdim=True)
    return kmeans(tokens, int(num_centers))


class EnsembleExtractor(nn.Module):
    """Returns, per surrogate, the global feature and the K local cluster centres."""

    def __init__(self, surrogates, num_centers):
        super().__init__()
        self.surrogates = nn.ModuleList(surrogates)
        self.num_centers = num_centers

    def forward(self, x):
        features, features_local = {}, {}
        for i, model in enumerate(self.surrogates):
            global_feature, tokens = model.global_local_features(x)
            features[i] = global_feature.squeeze()
            features_local[i] = cluster_centers(tokens[0], self.num_centers).unsqueeze(0)
        return features, features_local


class AlignmentLoss(nn.Module):
    """Alignment score (to be maximised) between adversarial and target features.

    Args:
        num_centers:  K, number of local cluster centres per image.
        local_weight: lambda, weight of the local OT term.
        rho:          marginal relaxation of the unbalanced OT (smaller = looser).
        gamma:        strength of the confidence-adaptive relaxation on the
                      adversarial side; gamma=0 is plain unbalanced OT.
        eps:          entropic temperature of the Sinkhorn kernel.
    """

    def __init__(self, surrogates, num_centers=10, local_weight=2.0, rho=0.2, gamma=1.0, eps=0.1):
        super().__init__()
        self.surrogates = nn.ModuleList(surrogates)
        self.num_centers = num_centers
        self.local_weight = local_weight
        self.rho = rho
        self.gamma = gamma
        self.eps = eps
        self.target_global = []
        self.target_local = []
        self.previous_scores = []

    def reset(self):
        """Forget the dynamic-weighting state (call once per image)."""
        self.previous_scores = []

    @torch.no_grad()
    def set_target(self, x):
        self.target_global.clear()
        self.target_local.clear()
        for model in self.surrogates:
            global_feature, tokens = model.global_local_features(x)
            centers = cluster_centers(tokens.squeeze(0), self.num_centers).unsqueeze(0)
            self.target_global.append(global_feature)
            self.target_local.append(centers)

    def forward(self, features, features_local):
        scores = []
        for i in range(len(self.surrogates)):
            local = self.transport(self.target_local[i].squeeze(0), features_local[i].squeeze(0),
                                   rho=self.rho, normalize=True, adaptive=True)
            # A 1x1 balanced transport: the cosine similarity of the global features.
            score = self.transport(self.target_global[i], features[i].unsqueeze(0))
            # `local` is None when the Sinkhorn iteration diverged (degenerate crop).
            scores.append(score if local is None else score + self.local_weight * local)

        # Dynamic ensemble weighting: w = K * softmax(L_i / L_i^prev). The weights
        # sum to K, so the objective keeps the scale of an unweighted sum.
        K = len(self.surrogates)
        if len(self.previous_scores) == 0:
            self.previous_scores = [s.detach() for s in scores]
        ratios = np.array([scores[i].item() / (self.previous_scores[i].item() + 1e-8) for i in range(K)])
        weights = np.exp(ratios)
        weights /= np.sum(weights)
        weights *= K
        for i in range(K):
            self.previous_scores[i] = scores[i].detach()

        return sum(weights[i] * scores[i] for i in range(K))

    def transport(self, tgt, adv, rho=None, normalize=False, adaptive=False):
        """OT alignment <T, S> between target rows `tgt` (M,D) and adversarial rows `adv` (N,D).

        rho=None gives balanced OT. The plan T is computed without gradient; the
        gradient flows through the similarity matrix S only.
        """
        sim = torch.einsum("md,nd->mn", F.normalize(tgt, dim=1), F.normalize(adv, dim=1)).contiguous()
        u = torch.full((tgt.shape[0],), 1.0 / tgt.shape[0], dtype=sim.dtype, device=sim.device)
        v = torch.full((adv.shape[0],), 1.0 / adv.shape[0], dtype=sim.dtype, device=sim.device)
        with torch.no_grad():
            kernel = torch.exp(-(1 - sim) / self.eps)
            T = self.sinkhorn(kernel, u, v, rho=rho, adaptive=adaptive, sim=sim)
        if torch.isnan(T).any():
            return None
        if normalize:
            # An unbalanced plan no longer sums to one; renormalise so the local
            # term stays on the scale of the global one.
            T = T / (T.sum() + 1e-8)
        return torch.sum(torch.sum(T * sim, dim=(0, 1)))

    def sinkhorn(self, K, u, v, rho=None, adaptive=False, sim=None):
        """Unbalanced Sinkhorn with relaxation exponent f = rho / (rho + eps).

        Adaptive variant: each adversarial cluster j gets its own rho_j, modulated
        by its confidence c_j (best-match similarity to the target clusters,
        normalised to mean one):  rho_j = rho * ((1 - gamma) + gamma * c_j).
        Confident clusters are constrained more tightly, uncertain ones are relaxed.
        """
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        if rho is None:
            f_r = torch.ones_like(u)
            f_c = torch.ones_like(v)
        else:
            f_r = torch.full_like(u, rho / (rho + self.eps))
            if adaptive:
                conf = sim.max(dim=0).values.clamp(min=0.0)
                conf = conf / (conf.mean() + 1e-8)
                rho_c = rho * ((1.0 - self.gamma) + self.gamma * conf).clamp(min=0.0)
                f_c = rho_c / (rho_c + self.eps)
            else:
                f_c = torch.full_like(v, rho / (rho + self.eps))
        for _ in range(100):
            r0 = r
            r = (u / (K @ c.unsqueeze(-1)).squeeze(-1)) ** f_r
            c = (v / (K.t() @ r.unsqueeze(-1)).squeeze(-1)) ** f_c
            if (r - r0).abs().mean().item() < 1e-2:
                break
        return torch.outer(r, c) * K
