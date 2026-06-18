from . import _C


def attention_naive(q, k, v, scale=None):
    """Naive CUDA attention forward (no optimizations, correctness baseline).

    Each block handles one (batch, head), each thread handles one query.
    No shared memory — intentionally the simplest possible baseline.

    Args:
        q, k, v: [B, H, N, D] tensors (float32), N ≤ 1024
        scale: scaling factor (default: 1/sqrt(D))
    Returns:
        output: [B, H, N, D] tensor
    """
    B, H, N, D = q.shape
    if scale is None:
        scale = D ** -0.5
    return _C.attention_forward_naive(q, k, v, float(scale))


def attention_tiled(q, k, v, scale=None):
    """Tiled + shared memory optimized CUDA attention forward.

    Uses shared memory tiling (TILE_N=32), online safe softmax,
    D_PAD bank conflict elimination, and K+V simultaneous load.

    Args:
        q, k, v: [B, H, N, D] tensors (float32)
        scale: scaling factor (default: 1/sqrt(D))
    Returns:
        output: [B, H, N, D] tensor
    """
    B, H, N, D = q.shape
    if scale is None:
        scale = D ** -0.5
    return _C.attention_forward_tiled(q, k, v, float(scale))
