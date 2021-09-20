import pytest
import torch
from torch_scatter import scatter_max
import numpy as np
import torch_cmspepr.objectcondensation as oc
torch.manual_seed(1001)
np.random.seed(1001)


def test_batch_cluster_indices():
    cluster_id_per_event = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    expected_output = torch.LongTensor([0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6])
    # Should work with unordered cluster_id_per_event and batch, so shuffle
    shuffle = torch.randperm(cluster_id_per_event.size(0))
    cluster_id_per_event = cluster_id_per_event[shuffle]
    batch = batch[shuffle]
    expected_cluster_id = expected_output[shuffle]
    expected_n_clusters_per_event = torch.LongTensor([3, 2, 2])
    cluster_id, n_clusters_per_event = oc.batch_cluster_indices(cluster_id_per_event, batch)
    assert torch.all(torch.isclose(cluster_id,expected_cluster_id))
    assert torch.all(torch.isclose(n_clusters_per_event,expected_n_clusters_per_event))


@pytest.fixture
def simple_clustering_problem():
    cluster_index = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    betas = np.random.rand(10) * .01
    betas[np.array([1, 5, 8])] += .15 # Make fake condensation points
    # Make a clustering space that is easy to cluster
    cluster_space_coords = np.random.rand(10,2) + 2.*np.expand_dims(cluster_index, -1)
    return betas, cluster_space_coords

def test_get_clustering_np(simple_clustering_problem):
    output = oc.get_clustering_np(*simple_clustering_problem)
    expected_output = np.array([1, 1, 1, 1, 5, 5, 5, 8, 8, 8])
    np.testing.assert_array_equal(output, expected_output)

def test_get_clustering_torch(simple_clustering_problem):
    betas, cluster_space_coords = simple_clustering_problem
    betas = torch.FloatTensor(betas)
    cluster_space_coords = torch.FloatTensor(cluster_space_coords)
    output = oc.get_clustering(betas, cluster_space_coords)
    expected_output = torch.LongTensor([1, 1, 1, 1, 5, 5, 5, 8, 8, 8])
    assert torch.allclose(output, expected_output)


def test_scatter_count():
    assert torch.allclose(
        torch.LongTensor([3, 2 ,2]),
        oc.scatter_count(torch.Tensor([0, 0, 0, 1, 1, 2, 2]))
        )


def test_scatter_counts_to_indices():
    assert torch.allclose(
        oc.scatter_counts_to_indices(torch.LongTensor([3, 2, 2])),
        torch.LongTensor([0, 0, 0, 1, 1, 2, 2])
        )


def test_make_norm_mask():
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    cluster_id_per_event = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    n_clusters_per_event = scatter_max(cluster_id_per_event, batch)[0] + 1
    assert torch.allclose(
        oc.get_inter_event_norms_mask(batch, n_clusters_per_event),
        torch.LongTensor([
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 1],
            ])
        )


def test_calc_LV_Lbeta_runs():
    """
    Tests whether the calc_LV_Lbeta function returns output.
    Does not perform any checking on that output.
    """
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    cluster_index_per_event = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    N = batch.size(0)
    out = oc.calc_LV_Lbeta(
        torch.rand(N), torch.rand((N, 2)),
        cluster_index_per_event, batch
        )
    assert isinstance(out[0], torch.Tensor)
    assert isinstance(out[1], torch.Tensor)
    assert len(out) == 2


def test_calc_simple_clus_space_loss_runs():
    """
    Tests whether the calc_simple_clus_space_loss function returns output.
    Does not perform any checking on that output.
    """
    cluster_index_per_event = torch.LongTensor([
        0, 0, 0, 1, 1, 2, 2,
        0, 0, 1, 1,
        0, 0, 1, 1, 1, 2, 2,
        ])
    batch = torch.LongTensor([
        0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2
        ])
    cluster_space_coords = torch.rand((batch.size(0), 2))
    L_att, L_rep = oc.calc_simple_clus_space_loss(
        cluster_space_coords, cluster_index_per_event, batch
        )
    assert isinstance(L_att, torch.Tensor)
    assert isinstance(L_rep, torch.Tensor)
