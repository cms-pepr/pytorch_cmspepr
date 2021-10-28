import numpy as np
import torch
import torch_cmspepr.dataset as dataset

def test_incremental_cluster_index():
    input = torch.LongTensor([13, 4, 4, 13, 4, 13, 13, 42, 4, 4])
    assert torch.allclose(
        dataset.incremental_cluster_index(input),
        torch.LongTensor([1, 0, 0, 1, 0, 1, 1, 2, 0, 0])
        )
    # Noise index should get 0 if it is supplied:
    assert torch.allclose(
        dataset.incremental_cluster_index(input, noise_index=13),
        torch.LongTensor([0, 1, 1, 0, 1, 0, 0, 2, 1, 1])
        )
    # 0 should still be reserved for noise_index even if it is not present:
    assert torch.allclose(
        dataset.incremental_cluster_index(input, noise_index=-99),
        torch.LongTensor([2, 1, 1, 2, 1, 2, 2, 3, 1, 1])
        )

def test_incremental_cluster_index_np():
    input = np.array([13, 4, 4, 13, 4, 13, 13, 42, 4, 4])
    np.testing.assert_array_equal(
        dataset.incremental_cluster_index_np(input),
        np.array([1, 0, 0, 1, 0, 1, 1, 2, 0, 0])
        )
    # Noise index should get 0 if it is supplied:
    np.testing.assert_array_equal(
        dataset.incremental_cluster_index_np(input, noise_index=13),
        np.array([0, 1, 1, 0, 1, 0, 0, 2, 1, 1])
        )
    # 0 should still be reserved for noise_index even if it is not present:
    np.testing.assert_array_equal(
        dataset.incremental_cluster_index_np(input, noise_index=-99),
        np.array([2, 1, 1, 2, 1, 2, 2, 3, 1, 1])
        )

def test_noise_reduction():
    input = np.array([9, -1, 9, 9, -1, -1, -1, 9, -1, -1, -1, -1, 9])
    mask = dataset.mask_fraction_of_noise(input, .5)
    assert mask.shape[0] == input.shape[0]
    assert mask.sum() == 5+4
    out = input[mask]
    assert (out == 9).sum() == (input == 9).sum()
    assert (out == -1).sum() == .5*(input == -1).sum()
