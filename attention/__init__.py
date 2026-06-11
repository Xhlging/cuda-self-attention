from . import _C

def attention_naive(q, k, v, scale=None):
    """Naive CUDA attention forward (no optimizations).

    Args:
        q, k, v: [B, H, N, D] tensors (float32)
        scale: scaling factor (default: 1/sqrt(D))
    Returns:
        output: [B, H, N, D] tensor
    """
    B, H, N, D = q.shape
    if scale is None:
        scale = D ** -0.5
    return _C.attention_forward_naive(q, k, v, float(scale))


def attention_tiled(q, k, v, scale=None):
    """Tiled + shared memory optimized CUDA attention forward."""
    B, H, N, D = q.shape
    if scale is None:
        scale = D ** -0.5
    return _C.attention_forward_tiled(q, k, v, float(scale))
