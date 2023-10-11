import os.path as osp
from math import log

import torch
from torch_geometric.data import Data

SO_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))


def calc_q_betaclip(beta, qmin=1.0):
    return (beta.clip(0.0, 1 - 1e-4) / 1.002).arctanh() ** 2 + qmin


# Single event
class single:
    # fmt: off
    model_out = torch.FloatTensor([
        # Event 0
        # beta x0    x1        y
        [0.01, 0.40, 0.40],  # 0
        [0.02, 0.10, 0.90],  # 0
        [0.12, 0.70, 0.70],  # 1 <- d_sq to cond point = 0.02^2 + 0.02^2 = 0.0008; d=0.0283
        [0.01, 0.90, 0.10],  # 0
        [0.13, 0.72, 0.72],  # 1 <-- cond point for y=1
        ])
    # fmt: on
    x = model_out[:, 1:].contiguous()
    y = torch.LongTensor([0, 0, 1, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0])
    beta = torch.sigmoid(model_out[:, 0]).contiguous()
    q = calc_q_betaclip(beta)

    @classmethod
    def d(cls, i, j):
        return ((cls.x[i] - cls.x[j]) ** 2).sum()

    # Manual OC:
    @classmethod
    def losses(cls):
        beta = single.beta
        q = single.q
        d = single.d

        V_att = d(2, 4) * q[2] * q[4] / 5.0  # Since d is small, d == d_huber
        V_rep = (
            torch.exp(-4.0 * d(0, 4)) * q[0] * q[4]
            + torch.exp(-4.0 * d(1, 4)) * q[1] * q[4]
            + torch.exp(-4.0 * d(3, 4)) * q[3] * q[4]
        ) / 5.0
        V_srp = -1.0 / (20.0 * d(2, 4) + 1.0) * beta[4] / 2.0
        L_beta_cond_logterm = -0.2 * log(beta[4] + 1e-9)
        L_beta_noise = (beta[0] + beta[1] + beta[3]) / 3.0

        losses_man = torch.FloatTensor(
            [V_att, V_rep, V_srp, L_beta_cond_logterm, L_beta_noise]
        )
        return losses_man


def test_oc_cpu_single():
    torch.ops.load_library(osp.join(SO_DIR, 'oc_cpu.so'))

    losses_cpp = torch.ops.oc_cpu.oc_cpu(
        single.beta,
        single.q,
        single.x,
        single.y.type(torch.int),
        torch.IntTensor([0, 5]),
    )

    losses_man = single.losses()
    print(f'{losses_man=}')
    print(f'{losses_cpp=}')
    assert torch.allclose(losses_cpp, losses_man, rtol=0.001, atol=0.001)


def test_oc_python_single():
    import torch_cmspepr

    losses = torch_cmspepr.oc(single.beta, single.q, single.x, single.y, single.batch)
    losses_man = single.losses()
    print(f'{losses_man=}')
    print(f'{losses=}')
    assert torch.allclose(losses, losses_man, rtol=0.001, atol=0.001)


class multiple:
    # fmt: off
    model_out = torch.FloatTensor([
        # Event 0
        # beta x0    x1       idx y
        [0.01, 0.40, 0.40],  #  0 0
        [0.02, 0.10, 0.90],  #  1 0
        [0.12, 0.70, 0.70],  #  2 1 <- d_sq to cond point = 0.02^2 + 0.02^2 = 0.0008; d=0.0283
        [0.01, 0.90, 0.10],  #  3 0
        [0.13, 0.72, 0.72],  #  4 1 <-- cond point for y=1
        # Event 1
        [0.11, 0.40, 0.40],  #  5 2
        [0.02, 0.10, 0.90],  #  6 0
        [0.12, 0.70, 0.70],  #  7 1 <-- cond point for y=1
        [0.01, 0.90, 0.10],  #  8 0
        [0.13, 0.72, 0.72],  #  9 2 <-- cond point for y=2
        [0.11, 0.72, 0.72],  # 10 1
        ])
    x = model_out[:,1:].contiguous()
    y = torch.LongTensor([
        0, 0, 1, 0, 1,    # Event 0
        2, 0, 1, 0, 2, 1  # Event 1
        ])
    batch = torch.LongTensor([
        0, 0, 0, 0, 0,    # Event 0
        1, 1, 1, 1, 1, 1  # Event 1
        ])
    # fmt: on
    row_splits = torch.IntTensor([0, 5, 11])
    beta = torch.sigmoid(model_out[:, 0]).contiguous()
    q = calc_q_betaclip(beta).contiguous()


def test_oc_cpu_batch():
    torch.ops.load_library(osp.join(SO_DIR, 'oc_cpu.so'))
    try:
        import cmspepr_hgcal_core.objectcondensation as objectcondensation
    except ImportError:
        print('Install cmspepr_hgcal_core to run this test')
        return

    objectcondensation.ObjectCondensation.beta_term_option = 'short_range_potential'
    objectcondensation.ObjectCondensation.sB = 1.0

    loss_py = objectcondensation.oc_loss(
        multiple.model_out, Data(y=multiple.y, batch=multiple.batch)
    )
    losses_py = torch.FloatTensor(
        [
            loss_py["V_att"],
            loss_py["V_rep"],
            loss_py["L_beta_sig"],
            loss_py["L_beta_cond_logterm"],
            loss_py["L_beta_noise"],
        ]
    )
    losses_cpp = torch.ops.oc_cpu.oc_cpu(
        multiple.beta,
        multiple.q,
        multiple.x,
        multiple.y.type(torch.int),
        multiple.row_splits,
    )
    print(losses_py)
    print(losses_cpp)
    # Lots of rounding errors in python vs c++, can't compare too rigorously
    assert torch.allclose(losses_cpp, losses_py, rtol=0.01, atol=0.01)


def test_oc_python_batch():
    import torch_cmspepr

    try:
        import cmspepr_hgcal_core.objectcondensation as objectcondensation
    except ImportError:
        print('Install cmspepr_hgcal_core to run this test')
        return

    objectcondensation.ObjectCondensation.beta_term_option = 'short_range_potential'
    objectcondensation.ObjectCondensation.sB = 1.0

    loss_py = objectcondensation.oc_loss(
        multiple.model_out, Data(y=multiple.y, batch=multiple.batch)
    )
    losses_py = torch.FloatTensor(
        [
            loss_py["V_att"],
            loss_py["V_rep"],
            loss_py["L_beta_sig"],
            loss_py["L_beta_cond_logterm"],
            loss_py["L_beta_noise"],
        ]
    )
    losses = torch_cmspepr.oc(
        multiple.beta,
        multiple.q,
        multiple.x,
        multiple.y.type(torch.int),
        multiple.batch,
    )
    print(losses_py)
    print(losses)
    # Lots of rounding errors in python vs c++, can't compare too rigorously
    assert torch.allclose(losses, losses_py, rtol=0.01, atol=0.01)
