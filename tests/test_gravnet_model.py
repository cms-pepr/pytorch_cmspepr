import torch
import torch_cmspepr.gravnet_model as gm

torch.manual_seed(1001)

def test_global_exchange():
    x = torch.rand(10,3)
    batch = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    x_exchanged = gm.global_exchange(x, batch)
    assert x_exchanged.size() == (10, 12)
    # Compare first 4 rows manually with expectation
    assert torch.allclose(x[:4].mean(dim=0), x_exchanged[0,:3])
    assert torch.allclose(x[:4].min(dim=0)[0], x_exchanged[0,3:6])
    assert torch.allclose(x[:4].max(dim=0)[0], x_exchanged[0,6:9])


def test_gravnet_model_runs():
    """
    Tests whether the GravNetModel returns output.
    Does not perform any checking on that output.
    """
    n_hits = 100
    n_events = 5
    batch = (n_events*torch.rand(n_hits)).long()
    model = gm.GravnetModel(4, 3)
    out = model(torch.rand(n_hits, 4).float(), batch)
    assert out.size() == (n_hits, 3)
