import torch
from torch_geometric.data import Data
import torch_cmspepr

import tqdm
import time


def make_random_event(n_nodes=10000, n_events=5):
    model_out = torch.rand((n_nodes, 32))

    # Varying event sizes
    event_fracs = torch.normal(torch.ones(n_events), .1)
    event_fracs /= event_fracs.sum()
    event_sizes = (event_fracs * n_nodes).type(torch.int)
    event_sizes[-1] += n_nodes - event_sizes.sum() # Make sure it adds up to n_nodes

    batch = torch.arange(n_events).repeat_interleave(event_sizes)
    row_splits = torch.cat((torch.zeros(1, dtype=torch.int), torch.cumsum(event_sizes, 0)))

    ys = []
    for i_event in range(n_events):
        n_clusters = torch.randint(3, 8, (1,)).item() # Somewhere between 3 and 8 particles
        cluster_fracs = torch.randint(50, 200, (n_clusters,)).type(torch.float)
        cluster_fracs[0] += 200 # Boost the amount of noise relatively
        cluster_fracs /= cluster_fracs.sum()
        cluster_sizes = (cluster_fracs * event_sizes[i_event]).type(torch.int)
        # Make sure it adds up to n_nodes in this event
        cluster_sizes[-1] += event_sizes[i_event] - cluster_sizes.sum()
        ys.append(torch.arange(n_clusters).repeat_interleave(cluster_sizes))
    y = torch.cat(ys)
    
    y = y.type(torch.int)
    row_splits = row_splits.type(torch.int)
    return model_out, y, batch, row_splits


def test_oc_performance():
    try:
        import cmspepr_hgcal_core.objectcondensation as objectcondensation
    except ImportError:
        print('Install cmspepr_hgcal_core to run this test')
        return

    objectcondensation.ObjectCondensation.beta_term_option = 'short_range_potential'
    objectcondensation.ObjectCondensation.sB = 1.

    t_py = 0.
    t_cpp = 0.
    N = 1000
    for i_test in tqdm.tqdm(range(N)):
        # Don't count prep work in performance
        model_out, y, batch, row_splits = make_random_event()
        data = Data(y=y.type(torch.long), batch=batch)
        beta = torch.sigmoid(model_out[:,0]).contiguous()
        q = objectcondensation.calc_q_betaclip(torch.sigmoid(model_out[:,0])).contiguous()
        x = model_out[:,1:].contiguous()

        t0 = time.perf_counter()
        objectcondensation.oc_loss(model_out, data)
        t1 = time.perf_counter()
        torch_cmspepr.oc(beta, q, x, y, batch)
        t2 = time.perf_counter()

        t_py += t1-t0
        t_cpp += t2-t1

    print(f'Average python time: {t_py/N:.4f}')
    print(f'Average cpp    time: {t_cpp/N:.4f}')
    print(f'Speed up is {t_py/t_cpp:.2f}x')


test_oc_performance()    