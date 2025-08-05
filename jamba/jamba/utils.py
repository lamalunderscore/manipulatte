"""Utility functions for dejavu implementation."""

import torch


def get_topk(
    attn_weights,
    k: int,
    metric: str = "l2",
    do_prefill: bool = False,
) -> torch.Tensor | None:
    """Return topk head indices for each token.

    Uses different metrics as a measure of uniformity. Return topk least-
    uniform heads for each token.

    Args:
        attn_weights (_type_): Weight matrix to calculate topk on.
        k (int): Number of indices to be returned.
        metric (str, optional): Metric to measure uniformity. Defaults to "l2".
        do_prefill (bool, optional): If the sparsification should be calculated during the
            prefill stage. Defaults to False.

    Raises:
        ValueError: If k is bigger than the number of heads, or k < 0.

    Returns:
        torch.Tensor | None: Tensor with topk indices for each token, if k != 0 else None

    """
    batch_size, num_heads, target_length, sequence_length = attn_weights.shape

    metric_map = {
        "l2": lambda x: torch.norm(x, p=2, dim=-1),
        "entropy": lambda x: torch.sum((x + 1e-12) * torch.log2(x + 1e-12), dim=-1),
    }

    if not do_prefill and not target_length == 1:
        k = num_heads  # deactivate sparsification

    if k == 0:
        return None

    if k > num_heads or k < 0:
        raise ValueError(f"k ({k}) cannot exceed number of attention heads ({num_heads})")
    metric_scores = metric_map[metric](attn_weights)

    _, topk_ind = metric_scores.topk(k, dim=1)

    batch_size_topk, k_topk, target_length_topk = topk_ind.shape

    assert k == k_topk, f"Topk k is {k_topk}, but expected {k} (as in attn_weights)."
    assert batch_size == batch_size_topk, (
        f"Topk batch size is {batch_size_topk}, but expected {batch_size} (as in attn_weights)."
    )
    assert target_length == target_length_topk, (
        f"Topk sequence length is {target_length_topk}, but expected {target_length} (as in attn_weights)."
    )
    return topk_ind


def keep_topk(attn_output, topk: torch.Tensor | None) -> torch.Tensor:
    """Mask out indices not mentioned in the topk tensor.

    Args:
        attn_output (_type_): Matrix to be masked.
        topk (torch.Tensor): Tensor including to-be-kept head indices for every token.

    Returns:
        torch.Tensor: Masked version of attn_output.

    """
    if topk is None:
        return attn_output * 0.0

    batch_size, num_heads, target_length, h = attn_output.shape
    batch_size_topk, k, target_length_topk = topk.shape

    assert batch_size_topk == batch_size, (
        "Shape mismatch, topk and attn_output have different batch size."
    )
    assert target_length_topk == target_length, (
        "Shape mismatch, topk and attn_output have different sequence length."
    )

    if k == num_heads:
        return attn_output

    # mask that can be broadcasted onto attn_putput
    # topk already fits dimensions of mask tensor
    mask = torch.zeros(
        batch_size, num_heads, target_length, dtype=torch.bool, device=attn_output.device
    )

    # unmask head indices in topk
    mask.scatter_(
        dim=1,  # head dimension
        index=topk,
        src=torch.ones_like(topk, dtype=torch.bool, device=attn_output.device),
    )
    attn_output = attn_output * mask.unsqueeze(dim=-1)
    return attn_output


__all__ = ["get_topk", "keep_topk"]
