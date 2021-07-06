from typing import Optional, Tuple

import torch


@torch.jit.script
def select_knn(x: torch.Tensor,
               k: int,
               batch_x: Optional[torch.Tensor] = None,
               inmask: Optional[torch.Tensor] = None,
               max_radius: float = 1e9,
               mask_mode: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Finds for each element in :obj:`x` the :obj:`k` nearest points in
    :obj:`x`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        max_radius (float): Maximum distance to nearest neighbours. (default: :obj:`1e9`)
        mask_mode (int): ??? (default: :obj:`1`)

    :rtype: :class:`Tuple`[`LongTensor`,`FloatTensor`]

    .. code-block:: python

        import torch
        from torch_cmspepr import select_knn

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        assign_index = select_knn(x, 2, batch_x)
    """

    x = x.view(-1, 1) if x.dim() == 1 else x
    x = x.contiguous()

    mask: torch.Tensor = torch.ones(x.shape[0], dtype=torch.int32, device=x.device)
    if inmask is not None:
        mask = inmask
    
    row_splits: torch.Tensor = torch.tensor([0, x.shape[0]], dtype=torch.int32, device=x.device)
    if batch_x is not None:
        assert x.size(0) == batch_x.numel()
        batch_size = int(batch_x.max()) + 1

        deg = x.new_zeros(batch_size, dtype=torch.long)
        deg.scatter_add_(0, batch_x, torch.ones_like(batch_x))

        ptr_x = deg.new_zeros(batch_size + 1)
        torch.cumsum(deg, 0, out=ptr_x[1:])

    return torch.ops.torch_cmspepr.select_knn(
        x,
        row_splits,
        mask,
        k,
        max_radius,
        mask_mode,
    )


@torch.jit.script
def knn_graph(x: torch.Tensor, k: int, batch: Optional[torch.Tensor] = None,
              loop: bool = False, flow: str = 'source_to_target',
              cosine: bool = False, num_workers: int = 1) -> torch.Tensor:
    r"""Computes graph edges to the nearest :obj:`k` points.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        cosine (boolean, optional): If :obj:`True`, will use the Cosine
            distance instead of Euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import knn_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = knn_graph(x, k=2, batch=batch, loop=False)
    """

    assert flow in ['source_to_target', 'target_to_source']

    K = k if loop else k + 1
    start = 0 if loop else 1
    
    index_dists = select_knn(x, K, batch) # select_knn is always in "loop" mode
    neighbours, edge_dists = index_dists[0], index_dists[1]
    
    sources = torch.arange(neighbours.shape[0], device=neighbours.device)[:, None].expand(-1, k).contiguous().view(-1)
    targets = neighbours[:,start:].contiguous().view(-1)
    
    edge_index = torch.cat([sources[None, :], targets[None, :]], dim = 0)
    
    if flow == 'source_to_target':
        row, col = edge_index[1], edge_index[0]
    else:
        row, col = edge_index[0], edge_index[1]

    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]

    return torch.stack([row, col], dim=0)


class SelectKnn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, ):
        pass
