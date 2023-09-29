import torch

# 4 points on a diagonal line with d^2 = 0.1^2+0.1^2 = 0.02 between them.
# 1 point very far away.
nodes = torch.FloatTensor(
    [
        # Event 0
        [0.1, 0.1],
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.4],
        [100.0, 100.0],
        # Event 1
        [0.1, 0.1],
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.4],
    ]
)
batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1])

# Expected output for k=3, max_radius=0.2 (with loop)
# Always a connection with self, which has distance 0.0
expected_neigh_indices = torch.IntTensor(
    [
        [0, 1, -1],
        [1, 0, 2],
        [2, 1, 3],
        [3, 2, -1],
        [4, -1, -1],
        [5, 6, -1],
        [6, 5, 7],
        [7, 6, 8],
        [8, 7, -1],
    ]
)
expected_neigh_dist_sq = torch.FloatTensor(
    [
        [0.0, 0.02, 0.00],
        [0.0, 0.02, 0.02],
        [0.0, 0.02, 0.02],
        [0.0, 0.02, 0.00],
        [0.0, 0.00, 0.00],
        [0.0, 0.02, 0.00],
        [0.0, 0.02, 0.02],
        [0.0, 0.02, 0.02],
        [0.0, 0.02, 0.00],
    ]
)
expected_edge_index_noloop = torch.LongTensor(
    [[0, 1, 1, 2, 2, 3, 5, 6, 6, 7, 7, 8], [1, 0, 2, 1, 3, 2, 6, 5, 7, 6, 8, 7]]
)
expected_edge_index_loop = torch.LongTensor(
    [
        [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8],
        [0, 1, 1, 0, 2, 2, 1, 3, 3, 2, 4, 5, 6, 6, 5, 7, 7, 6, 8, 8, 7],
    ]
)


def test_knn_graph_cpu():
    from torch_cmspepr import knn_graph

    # k=2 without loops
    edge_index = knn_graph(nodes, 2, batch, max_radius=0.2)
    print('Found edge_index:')
    print(edge_index)
    print('Expected edge_index:')
    print(expected_edge_index_noloop)
    assert torch.allclose(edge_index, expected_edge_index_noloop)
    # k=3 with loops
    edge_index = knn_graph(nodes, 3, batch, max_radius=0.2, loop=True)
    print('Found edge_index:')
    print(edge_index)
    print('Expected edge_index:')
    print(expected_edge_index_loop)
    assert torch.allclose(edge_index, expected_edge_index_loop)
    # k=3 with loops
    edge_index = knn_graph(
        nodes, 3, batch, max_radius=0.2, loop=True, flow='target_to_source'
    )
    print('Found edge_index:')
    print(edge_index)
    print('Expected edge_index:')
    expected = torch.flip(expected_edge_index_loop, [0])
    print(expected)
    assert torch.allclose(edge_index, expected)


def test_knn_graph_cpu_1dim():
    from torch_cmspepr import knn_graph

    nodes = torch.FloatTensor([0.1, 0.2, 0.5, 0.6])
    edge_index = knn_graph(nodes, 2, max_radius=0.2, loop=True)
    expected = torch.LongTensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 3],
            [0, 1, 1, 0, 2, 3, 3, 2],
        ]
    )
    print('Found edge_index:')
    print(edge_index)
    print('Expected edge_index:')
    print(expected)
    assert torch.allclose(edge_index, expected)


def test_knn_graph_cuda():
    from torch_cmspepr import knn_graph

    gpu = torch.device('cuda')
    nodes_cuda, batch_cuda = nodes.to(gpu), batch.to(gpu)
    # k=2 without loops
    edge_index = knn_graph(nodes_cuda, 2, batch_cuda, max_radius=0.2)
    print('[k=2 no loops] Found edge_index:')
    print(edge_index)
    print('Expected edge_index:')
    print(expected_edge_index_noloop)
    assert torch.allclose(edge_index, expected_edge_index_noloop.to(gpu))
    # k=3 with loops
    edge_index = knn_graph(nodes_cuda, 3, batch_cuda, max_radius=0.2, loop=True)
    print('[k=3 with loops] Found edge_index:')
    print(edge_index)
    print('Expected edge_index:')
    print(expected_edge_index_loop)
    assert torch.allclose(edge_index, expected_edge_index_loop.to(gpu))


def test_select_knn_cpu():
    from torch_cmspepr import select_knn

    neigh_indices, neigh_dist_sq = select_knn(nodes, k=3, batch_x=batch, max_radius=0.2)
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


def test_select_knn_cuda():
    from torch_cmspepr import select_knn

    device = torch.device('cuda')
    neigh_indices, neigh_dist_sq = select_knn(
        nodes.to(device), k=3, batch_x=batch.to(device), max_radius=0.2
    )
    neigh_indices = neigh_indices.cpu()
    neigh_dist_sq = neigh_dist_sq.cpu()
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
