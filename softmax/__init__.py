from . import _C


def softmax_naive(x):
    """Naive CUDA softmax forward (serial per-row, correctness baseline).

    Each thread independently scans the row three times: find max,
    compute exp + sum, then normalize.

    Args:
        x: [*, D] tensor (float32), D ∈ {64, 128, 256, 512, 1024}
    Returns:
        [*, D] tensor, softmax along last dim
    """
    return _C.softmax_forward_naive(x)


def softmax_warp(x):
    """Warp-reduction CUDA softmax forward.

    One warp (32 threads) per row. Each thread processes D/32 elements
    with strided access, then __shfl_down_sync butterfly reduction
    merges 32 partial results into 1, followed by __shfl_sync broadcast.

    Args:
        x: [*, D] tensor (float32), D ∈ {64, 128, 256, 512, 1024}
    Returns:
        [*, D] tensor, softmax along last dim
    """
    return _C.softmax_forward_warp(x)
