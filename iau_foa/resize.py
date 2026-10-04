"""Deterministic resize on GPU, written as two matrix multiplications.

Bicubic / antialiased interpolation has no deterministic backward kernel on
CUDA, so `torch.use_deterministic_algorithms(True)` either errors or forces the
resize onto the CPU. Resizing is a linear, separable operator, however:

    Y = A_h @ X @ A_w^T        (A_h: H_out x H_in, A_w: W_out x W_in)

and matmul (forward and backward) is deterministic under
CUBLAS_WORKSPACE_CONFIG=:4096:8. The matrices below reproduce the weights of
`torch.nn.functional.interpolate` (float64 agreement <= 2e-15), so the attack
keeps exact gradients, runs on the GPU, and is reproducible.
"""
import torch
import torch.nn.functional as F


def _cubic(x, A):
    """Cubic convolution kernel k(x): k(0)=1, k(+-1)=k(+-2)=0."""
    x = abs(float(x))
    if x < 1.0:
        return (A + 2.0) * x**3 - (A + 3.0) * x**2 + 1.0
    if x < 2.0:
        return A * x**3 - 5.0 * A * x**2 + 8.0 * A * x - 4.0 * A
    return 0.0


def _linear(x):
    x = abs(float(x))
    return max(0.0, 1.0 - x)


def build_matrix(in_size, out_size, mode="bicubic", antialias=True):
    """1-D resize matrix (out_size x in_size), float64, align_corners=False.

    antialias=False: fixed taps (bicubic: 4 taps, A=-0.75); out-of-range taps are
                     accumulated onto the clamped border pixel.
    antialias=True : PIL-style continuous kernel with filterscale=max(scale, 1);
                     the window is clipped to [0, in_size) and the weights are
                     normalised inside the clipped window. Bicubic uses A=-0.5.
    """
    M = torch.zeros(out_size, in_size, dtype=torch.float64)
    scale = in_size / out_size
    if mode == "bicubic":
        kern, radius, A = _cubic, 2.0, (-0.5 if antialias else -0.75)
    elif mode == "bilinear":
        kern, radius, A = _linear, 1.0, None
    else:
        raise ValueError(f"unsupported mode: {mode}")

    if antialias:
        fs = max(scale, 1.0)
        support = radius * fs
        for i in range(out_size):
            center = (i + 0.5) * scale
            jmin = max(0, int(center - support + 0.5))
            jmax = min(in_size, int(center + support + 0.5))
            w = []
            for j in range(jmin, jmax):
                x = (j - center + 0.5) / fs
                w.append(kern(x, A) if A is not None else kern(x))
            s = sum(w)
            for j, wv in zip(range(jmin, jmax), w):
                M[i, j] += wv / s
    else:
        ntap = int(2 * radius)
        for i in range(out_size):
            center = (i + 0.5) * scale - 0.5
            base = int(torch.floor(torch.tensor(center)).item())
            for k in range(ntap):
                j = base - int(radius) + 1 + k
                x = center - j
                wv = kern(x, A) if A is not None else kern(x)
                M[i, min(max(j, 0), in_size - 1)] += wv
    return M


_CACHE = {}


def matmul_resize(x, out_hw, mode="bicubic", antialias=True):
    """Resize x (B,C,H,W) -> (B,C,*out_hw). Differentiable and deterministic."""
    H, W = x.shape[-2], x.shape[-1]
    key = (H, W, out_hw, mode, antialias, x.device.type, x.dtype)
    if key not in _CACHE:
        Ah = build_matrix(H, out_hw[0], mode, antialias)
        Aw = build_matrix(W, out_hw[1], mode, antialias)
        _CACHE[key] = (Ah.to(x.device, x.dtype), Aw.to(x.device, x.dtype))
    Ah, Aw = _CACHE[key]
    return Ah @ x @ Aw.T


def patch_resize():
    """Route bicubic / bilinear resizing of CUDA tensors through `matmul_resize`.

    Covers `torchvision.transforms.functional.resize` (the `transforms.Resize` of the
    surrogate preprocessing) and bicubic / antialiased `F.interpolate` (DINOv2 resizes
    its position embeddings bicubically). Call forms without an explicit (H, W)
    size fall back to the CPU, which is deterministic as well.
    """
    import torchvision.transforms.functional as TVF

    native_interpolate = F.interpolate
    native_resize = TVF.resize

    def resize(img, size, interpolation=TVF.InterpolationMode.BILINEAR, max_size=None, antialias=True):
        mode = getattr(interpolation, "value", str(interpolation)).lower()
        if not (torch.is_tensor(img) and img.is_cuda) or mode not in ("bilinear", "bicubic"):
            return native_resize(img, size, interpolation, max_size, antialias)
        h, w = img.shape[-2], img.shape[-1]
        if isinstance(size, (list, tuple)) and len(size) == 2:
            out_hw = (int(size[0]), int(size[1]))
        elif h == w and max_size is None:
            s = int(size[0]) if isinstance(size, (list, tuple)) else int(size)
            out_hw = (s, s)
        else:
            return native_resize(img.cpu(), size, interpolation, max_size, antialias).to(img.device)
        if img.dim() == 3:
            return matmul_resize(img.unsqueeze(0), out_hw, mode, bool(antialias)).squeeze(0)
        return matmul_resize(img, out_hw, mode, bool(antialias))

    def interpolate(inp, *args, **kwargs):
        mode = str(kwargs.get("mode", args[2] if len(args) > 2 else "nearest"))
        antialias = bool(kwargs.get("antialias", False))
        if not (torch.is_tensor(inp) and inp.is_cuda and ("cubic" in mode or antialias)):
            return native_interpolate(inp, *args, **kwargs)
        size = kwargs.get("size", args[0] if args else None)
        if isinstance(size, (list, tuple)) and len(size) == 2 and mode in ("bilinear", "bicubic"):
            return matmul_resize(inp, (int(size[0]), int(size[1])), mode, antialias)
        return native_interpolate(inp.cpu(), *args, **kwargs).to(inp.device)

    TVF.resize = resize
    F.interpolate = interpolate
