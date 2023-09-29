import os.path as osp
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

    # Compute row_splits
    if batch_x is None:
        row_splits: torch.Tensor = torch.tensor([0, x.shape[0]], dtype=torch.int32, device=x.device)
    else:
        assert x.size(0) == batch_x.size(0)
        batch_size = int(batch_x.max()) + 1

        # Get number of hits per event
        counts = torch.zeros(batch_size, dtype=torch.int32, device=x.device)
        counts.scatter_add_(0, batch_x, torch.ones_like(batch_x, dtype=torch.int32))

        # Convert counts to row_splits by using cumsum.
        # row_splits must start with 0 and end with x.size(0), and has length +1 w.r.t.
        # batch_size.
        # e.g. for 2 events with 5 and 4 hits, row_splits would be [0, 5, 9]
        row_splits = torch.zeros(batch_size+1, dtype=torch.int32, device=x.device)
        torch.cumsum(counts, 0, out=row_splits[1:])

    if x.device == torch.device('cpu'):
        return torch.ops.select_knn_cpu.select_knn_cpu(
            x,
            row_splits,
            mask,
            k,
            max_radius,
            mask_mode,
            )
    else:
        return torch.ops.select_knn_cuda.select_knn_cuda(
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
              cosine: bool = False, num_workers: int = 1,
              max_radius: float = 1e9) -> torch.Tensor:
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
    neighbours, edge_dists = select_knn(x, K, batch, max_radius=max_radius) # select_knn is always in "loop" mode
    
    # neighbours has the following (n_neigh x k) structure:
    # [[0,  1,  3, ...],  <-- node 0 connected with 0, 1, 3, ...
    #  [1,  0,  -1, ...]   <-- node 1 connected with 1 and 0
    #  [2, -1, -1, ...]   <-- node 2 connected with 2 and nothing else
    #  ...]
    # Flatten it to a 1-dim tensor; Drop first column if not doing the self loop
    if loop:
        targets = neighbours.flatten()
    else:
        targets = neighbours[:,1:].flatten()

    # Create sources:
    #   <--k--> <--k-->
    # [ 0 0 0 0 1 1 1 1 ... n_nodes n_nodes n_nodes n_nodes]
    sources = torch.repeat_interleave(torch.arange(x.size(0), device=x.device), k)

    if flow == 'source_to_target':
        edge_index = torch.stack((sources, targets))
    else:
        edge_index = torch.stack((targets, sources))

    # Filter out non-edges (target is -1)
    edge_index = edge_index[:,(targets>=0)]

    return edge_index