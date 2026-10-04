#!/usr/bin/env python3
"""Generate adversarial images.

    python IAU_FOA.py --config config/iau_foa_100.yaml [key=value ...]

Each clean image <clean_dir>/<class>/<name>.* is paired with the target image
<target_dir>/**/<name>.* and written to <output>/<class>/<name>.png.
Existing outputs are skipped, so an interrupted run can simply be restarted.
"""
import argparse
import logging
import os
import random
import sys

# Must be set before CUDA is initialised: makes cuBLAS matmul deterministic.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from iau_foa.attack import attack
from iau_foa.resize import patch_resize
from surrogates import (
    ClipB16FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    DINOv2FeatureExtractor,
    EnsembleFeatureExtractor_ot,
    EnsembleFeatureLoss_OT_foa_attack,
    InternVL3_1B_FeatureExtractor,
)

logger = logging.getLogger("iau_foa")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

SURROGATES = {
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
    "InternVL3_1B": InternVL3_1B_FeatureExtractor,
    "DINOv2_Base": DINOv2FeatureExtractor,
}


def setup_logging(log_file):
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(fmt)
        logger.addHandler(handler)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # TF32 costs ~1e-3 relative precision in matmul, which the resize cannot afford.
    torch.backends.cuda.matmul.allow_tf32 = False
    patch_resize()


def build_surrogates(names, device):
    models = []
    for name in names:
        if name not in SURROGATES:
            raise ValueError(f"unknown surrogate: {name} (available: {list(SURROGATES)})")
        models.append(SURROGATES[name]().eval().to(device).requires_grad_(False))
    return models


def list_pairs(clean_dir, target_dir):
    """[(clean_path, target_path)], clean images in sorted order, matched by file name."""
    targets = {}
    for root, _, files in os.walk(target_dir):
        for f in files:
            if f.lower().endswith(IMAGE_EXTENSIONS):
                targets[os.path.splitext(f)[0]] = os.path.join(root, f)
    pairs = []
    for path, _ in torchvision.datasets.ImageFolder(clean_dir).samples:
        name = os.path.splitext(os.path.basename(path))[0]
        if name not in targets:
            raise FileNotFoundError(f"no target image named {name}.* under {target_dir}")
        pairs.append((path, targets[name]))
    return pairs


def load_image(path, size):
    """Image file -> (1,3,size,size) float tensor in [0,255]."""
    img = Image.open(path).convert("RGB")
    img = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)(img)
    img = transforms.CenterCrop(size)(img)
    return torch.from_numpy(np.array(img, np.uint8)).permute(2, 0, 1).contiguous().float().unsqueeze(0)


def save_image(adv, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torchvision.utils.save_image(adv, tmp, format="png")
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config/iau_foa_100.yaml")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. model.device=cuda:1")
    args = parser.parse_args()
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))

    num_shards, shard_id = int(cfg.data.num_shards), int(cfg.data.shard_id)
    os.makedirs(cfg.data.output, exist_ok=True)
    log_name = "generate.log" if num_shards == 1 else f"generate_shard{shard_id}.log"
    setup_logging(os.path.join(cfg.data.output, log_name))
    logger.info("config:\n%s", OmegaConf.to_yaml(cfg))

    set_deterministic()
    seed_everything(int(cfg.seed))

    device = cfg.model.device
    surrogates = build_surrogates(cfg.model.surrogates, device)
    extractor = EnsembleFeatureExtractor_ot(surrogates, cluster_number=int(cfg.ot.num_centers))
    loss_fn = EnsembleFeatureLoss_OT_foa_attack(
        surrogates,
        cluster_number=int(cfg.ot.num_centers),
        local_weight=float(cfg.ot.local_weight),
        uot_reg_m=float(cfg.ot.rho),
        uot_normalize=True,
        uot_adaptive=True,
        uot_adaptive_gamma=float(cfg.ot.gamma),
        uot_adaptive_norm=True,
        eps=float(cfg.ot.eps),
    )

    pairs = list_pairs(cfg.data.clean_dir, cfg.data.target_dir)[: int(cfg.data.num_samples)]
    for index, (clean_path, target_path) in enumerate(pairs):
        # Multi-GPU sharding: this process handles images with index % num_shards == shard_id.
        if index % num_shards != shard_id:
            continue
        name = os.path.splitext(os.path.basename(clean_path))[0]
        out_path = os.path.join(cfg.data.output, os.path.basename(os.path.dirname(clean_path)), name + ".png")
        if os.path.exists(out_path):
            logger.info("[%d/%d] %s exists, skip", index + 1, len(pairs), out_path)
            continue
        logger.info("[%d/%d] %s -> %s", index + 1, len(pairs), clean_path, target_path)

        # Seed per image, so that the result for image i depends only on i and not on
        # which images were processed before it (sharding and restarts are safe).
        seed_everything((int(cfg.seed) * 1000003 + index) % (2 ** 31 - 1))

        image = load_image(clean_path, int(cfg.model.input_res)).to(device)
        target = load_image(target_path, int(cfg.model.input_res)).to(device)
        adv = attack(cfg, extractor, loss_fn, image, target, tag=f"img{index}")
        save_image(adv[0].cpu(), out_path)

    logger.info("done: %s", cfg.data.output)


if __name__ == "__main__":
    main()
