import importlib
import os.path as osp

import torch

__version__ = '0.0.1'

suffix = 'cuda' if torch.cuda.is_available() else 'cpu'

for library in [
        '_version',
        '_select_knn', '_select_knn_grad',
        '_accumulate_knn', '_accumulate_knn_grad',
]:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        f'{library}_{suffix}', [osp.dirname(__file__)]).origin)

if torch.cuda.is_available():  # pragma: no cover
    cuda_version = torch.ops.torch_cmspepr.cuda_version()

    if cuda_version == -1:
        major = minor = 0
    elif cuda_version < 10000:
        major, minor = int(str(cuda_version)[0]), int(str(cuda_version)[2])
    else:
        major, minor = int(str(cuda_version)[0:2]), int(str(cuda_version)[3])
    t_major, t_minor = [int(x) for x in torch.version.cuda.split('.')]

    if t_major != major:
        raise RuntimeError(
            f'Detected that PyTorch and torch_cmspepr were compiled with '
            f'different CUDA versions. PyTorch has CUDA version '
            f'{t_major}.{t_minor} and torch_cmspepr has CUDA version '
            f'{major}.{minor}. Please reinstall the torch_cmspepr that '
            f'matches your PyTorch install.')

from torch_cmspepr.select_knn import select_knn, knn_graph
from torch_cmspepr.gravnet_conv import GravNetConv

__all__ = [
    'select_knn', 'knn_graph',
    'GravNetConv'
]
