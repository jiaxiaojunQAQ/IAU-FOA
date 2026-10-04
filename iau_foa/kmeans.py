"""Differentiable K-Means (Euclidean), adapted from kmeans_pytorch
(https://github.com/subhadarship/kmeans_pytorch, MIT License).

Vendored because the attack depends on the exact behaviour of the upstream master
branch (random re-seeding of empty clusters), which the PyPI release 0.3 lacks.
"""
import numpy as np
import torch


def kmeans(X, num_clusters, tol=1e-4):
    """(N, D) points -> (num_clusters, D) centres.

    The centres are means of rows of X, so gradients flow back to X.
    Initial centres are drawn with the global NumPy RNG.
    """
    X = X.float()
    centers = X[np.random.choice(len(X), num_clusters, replace=False)]
    while True:
        dis = ((X.unsqueeze(dim=1) - centers.unsqueeze(dim=0)) ** 2.0).sum(dim=-1).squeeze()
        choice = torch.argmin(dis, dim=1)
        previous = centers.clone()
        for index in range(num_clusters):
            selected = torch.index_select(X, 0, torch.nonzero(choice == index).squeeze())
            if selected.shape[0] == 0:
                selected = X[torch.randint(len(X), (1,))]
            centers[index] = selected.mean(dim=0)
        shift = torch.sum(torch.sqrt(torch.sum((centers - previous) ** 2, dim=1)))
        if shift ** 2 < tol:
            break
    return centers
