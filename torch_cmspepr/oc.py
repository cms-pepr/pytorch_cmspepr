import torch


# @torch.jit.script
def oc(
    beta: torch.FloatTensor,
    q: torch.FloatTensor,
    x: torch.FloatTensor,
    y: torch.LongTensor,  # Use long for consistency
    batch: torch.LongTensor,  # Use long for consistency
    sB: float = 1.0,
):
    """
    Calculate the object condensation loss function.

    Args:
        beta (torch.FloatTensor): Beta as described in https://arxiv.org/abs/2002.03605;
            simply a sigmoid of the raw model output
        q (torch.FloatTensor): Charge q per node; usually a function of beta.
        x (torch.FloatTensor): Latent clustering space coordinates for every node.
        y (torch.LongTensor): Clustering truth. WARNING: The torch.op expects y to be
            nicely *incremental*. There should not be any holes in it.
        batch (torch.LongTensor): Batch vector to designate event boundaries. WARNING:
            It is expected that batch is *sorted*.

    Returns:
        torch.FloatTensor: A len-5 tensor with the 5 loss components of the OC loss
            function: V_att, V_rep, V_srp, L_beta_cond_logterm, and L_beta_noise. The
            full OC loss is simply the sum of this tensor.
    """
    N = beta.size(0)
    assert beta.dim() == 1
    assert q.dim() == 1
    assert beta.size() == q.size()
    assert x.size(0) == N
    assert y.size(0) == N
    assert batch.size(0) == N
    device = beta.device

    # TEMPORARY: No GPU version available yet.
    assert device == torch.device('cpu')

    # Translate batch vector into row splits
    counts = torch.zeros(batch.max() + 1, dtype=torch.int, device=device)
    counts.scatter_add_(0, batch, torch.ones_like(batch, dtype=torch.int))
    counts = torch.cat((torch.zeros(1, dtype=torch.int, device=device), counts))
    row_splits = torch.cumsum(counts, 0).type(torch.int)

    return torch.ops.oc_cpu.oc_cpu(beta, q, x, y.type(torch.int), row_splits)
