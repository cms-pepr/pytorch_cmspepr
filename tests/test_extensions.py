import os.path as osp
import torch

# 4 points on a diagonal line with d^2 = 0.1^2+0.1^2 = 0.02 between them.
# 1 point very far away.
nodes = torch.FloatTensor([
    # Event 0
    [.1, .1],
    [.2, .2],
    [.3, .3],
    [.4, .4],
    [100., 100.],
    # Event 1
    [.1, .1],
    [.2, .2],
    [.3, .3],
    [.4, .4]
    ])
row_splits = torch.IntTensor([0, 5, 9])
mask: torch.Tensor = torch.ones(nodes.shape[0], dtype=torch.int32)
max_radius = .2
mask_mode = 1
k = 3 # Including connection with self

# Expected output for k=3, max_radius=0.2 (with loop)
# Always a connection with self, which has distance 0.0
expected_neigh_indices = torch.IntTensor([
    [ 0,  1, -1],
    [ 1,  0,  2],
    [ 2,  1,  3],
    [ 3,  2, -1],
    [ 4, -1, -1],
    [ 5,  6, -1],
    [ 6,  5,  7],
    [ 7,  6,  8],
    [ 8,  7, -1]
    ])
expected_neigh_dist_sq = torch.FloatTensor([
    [0.0, 0.02, 0.00],
    [0.0, 0.02, 0.02],
    [0.0, 0.02, 0.02],
    [0.0, 0.02, 0.00],
    [0.0, 0.00, 0.00],
    [0.0, 0.02, 0.00],
    [0.0, 0.02, 0.02],
    [0.0, 0.02, 0.02],
    [0.0, 0.02, 0.00]
    ])

SO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))

def test_select_knn_op_cpu():
    torch.ops.load_library(osp.join(SO_DIR, 'select_knn_cpu.so'))
    neigh_indices, neigh_dist_sq = torch.ops.select_knn_cpu.select_knn_cpu(
        nodes,
        row_splits,
        mask,
        k,
        max_radius,
        mask_mode,
        )
    print('Expected indices:')
    print(expected_neigh_indices)
    print('Found indices:')
    print(neigh_indices)
    print('Expected dist_sq:')
    print(expected_neigh_dist_sq)
    print('Found dist_sq:')
    print(neigh_dist_sq)
    assert torch.allclose(neigh_indices, expected_neigh_indices)
    assert torch.allclose(neigh_dist_sq, expected_neigh_dist_sq)

def test_select_knn_op_cuda():
    gpu = torch.device('cuda')
    torch.ops.load_library(osp.join(SO_DIR, 'select_knn_cuda.so'))
    neigh_indices, neigh_dist_sq = torch.ops.select_knn_cuda.select_knn_cuda(
        nodes.to(gpu),
        row_splits.to(gpu),
        mask.to(gpu),
        k,
        max_radius,
        mask_mode,
        )
    print('Expected indices:')
    print(expected_neigh_indices)
    print('Found indices:')
    print(neigh_indices)
    print('Expected dist_sq:')
    print(expected_neigh_dist_sq)
    print('Found dist_sq:')
    print(neigh_dist_sq)
    assert torch.allclose(neigh_indices.cpu(), expected_neigh_indices)
    assert torch.allclose(neigh_dist_sq.cpu(), expected_neigh_dist_sq)