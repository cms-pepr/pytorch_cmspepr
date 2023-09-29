import os.path as osp

import torch

from .logger import logger

THISDIR = osp.dirname(osp.abspath(__file__))

__version__ = '1.0.0'


def load_ops(so_file):
    if not osp.isfile(so_file):
        logger.error(f'Could not load op: No file {so_file}')
    else:
        torch.ops.load_library(so_file)

load_ops(osp.join(THISDIR, "../select_knn_cpu.so"))
load_ops(osp.join(THISDIR, "../select_knn_cuda.so"))


from torch_cmspepr.select_knn import select_knn, knn_graph
from torch_cmspepr.gravnet_conv import GravNetConv

__all__ = [
    'select_knn', 'knn_graph',
    'select_knn_thomas_cpu', 'select_knn_thomas_cuda'
    'GravNetConv',
    'logger'
]
