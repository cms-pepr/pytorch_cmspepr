import os.path as osp
import logging
import torch

__version__ = '1.0.0'


def setup_logger(name: str = "cmspepr") -> logging.Logger:
    """Sets up a Logger instance.

    If a logger with `name` already exists, returns the existing logger.

    Args:
        name (str, optional): Name of the logger. Defaults to "demognn".

    Returns:
        logging.Logger: Logger object.
    """
    if name in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
    else:
        fmt = logging.Formatter(
            fmt=(
                "\033[34m[%(name)s:%(levelname)s:%(asctime)s:%(module)s:%(lineno)s]\033[0m"
                " %(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
    return logger

logger = setup_logger()


# Load the extensions as ops
def load_ops(so_file):
    if not osp.isfile(so_file):
        logger.error(f'Could not load op: No file {so_file}')
    else:
        torch.ops.load_library(so_file)

THISDIR = osp.dirname(osp.abspath(__file__))
load_ops(osp.join(THISDIR, "../select_knn_cpu.so"))
load_ops(osp.join(THISDIR, "../select_knn_cuda.so"))


from torch_cmspepr.select_knn import select_knn, knn_graph

__all__ = ['select_knn', 'knn_graph', 'logger']
