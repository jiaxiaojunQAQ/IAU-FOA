"""The attack loop: sign-gradient ascent on the ensemble alignment score, with the
gradient averaged over visual-invariance-augmented copies of the adversarial image."""
import logging

import torch
from torchvision import transforms

from .resize import matmul_resize

logger = logging.getLogger("iau_foa")

CROP_RATIO = (3.0 / 4.0, 4.0 / 3.0)


def random_resized_crop(x, size, scale):
    """torchvision's RandomResizedCrop (bilinear, antialiased) with a deterministic resize."""
    top, left, h, w = transforms.RandomResizedCrop.get_params(x, scale, CROP_RATIO)
    return matmul_resize(x[..., top:top + h, left:left + w], (size, size), "bilinear", True)


def via_copies(x, num_scales, wb_spread):
    """Visual-invariance augmentation: copies of x (pixel range [0,255]) under
    illumination changes the target model's vision encoder should be invariant to.

    - bidirectional intensity scaling by 2^k, k symmetric around 0
      (num_scales=5 -> {1/4, 1/2, 1, 2, 4}): both darker and brighter copies;
    - a random per-channel gain in [1 - wb_spread, 1 + wb_spread] per copy
      (white-balance perturbation).
    """
    half = num_scales // 2
    copies = []
    for s in [2.0 ** k for k in range(-half, num_scales - half)]:
        xs = x if s == 1.0 else torch.clamp(x * s, 0.0, 255.0)
        if wb_spread > 0.0:
            gain = 1.0 + (torch.rand(1, 3, 1, 1, device=xs.device) * 2 - 1) * wb_spread
            xs = torch.clamp(xs * gain, 0.0, 255.0)
        copies.append(xs)
    return copies


def attack(cfg, extractor, loss_fn, image, target, tag=""):
    """Perturb `image` so that its features align with those of `target`.

    image, target: (1,3,S,S) tensors in [0,255]. Returns the adversarial image in [0,1].
    """
    size = int(cfg.model.input_res)
    scale = [float(s) for s in cfg.model.crop_scale]
    steps = int(cfg.attack.steps)
    alpha, epsilon = cfg.attack.alpha, cfg.attack.epsilon

    loss_fn.reset()
    delta = torch.zeros_like(image, requires_grad=True)

    for step in range(steps):
        with torch.no_grad():
            loss_fn.set_target(random_resized_crop(target, size, scale))

        grad = torch.zeros_like(delta)
        local_sim, global_sim, n_ok = 0.0, 0.0, 0
        for copy in via_copies(image + delta, int(cfg.via.num_scales), float(cfg.via.wb_spread)):
            crop = random_resized_crop(copy, size, scale)
            # A degenerate random crop can make the Sinkhorn iteration diverge and
            # leave the loss disconnected from delta; skip that copy.
            try:
                # The full-view score is only monitored, but it is also the reference
                # the dynamic ensemble weights of the cropped view are computed against.
                score_full = loss_fn(*extractor(copy))
                score_crop = loss_fn(*extractor(crop))
                g = torch.autograd.grad(score_crop, delta)[0]
            except RuntimeError as e:
                logger.warning("%s step=%d skip copy (grad failed): %s", tag, step, e)
                continue
            grad = grad + g
            local_sim += score_crop.item()
            global_sim += score_full.item()
            n_ok += 1
        if n_ok == 0:
            logger.warning("%s step=%d all copies failed, skip step", tag, step)
            continue
        grad = grad / n_ok

        if step % 50 == 0 or step == steps - 1:
            logger.info("%s step=%d max_delta=%.3f mean_delta=%.3f global_similarity=%.5f local_similarity=%.5f",
                        tag, step, delta.abs().max().item(), delta.abs().mean().item(),
                        global_sim / n_ok, local_sim / n_ok)

        delta.data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)

    return torch.clamp((image + delta) / 255.0, 0.0, 1.0)
