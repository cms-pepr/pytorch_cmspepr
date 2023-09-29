# pytorch_cmspepr

pytorch bindings for optimized knn and aggregation kernels


## Example

```python
>>> import torch
>>> import torch_cmspepr

# Two events with 5 nodes and 4 nodes, respectively.
# Nodes here are on a diagonal line in 2D, with d^2 = 0.02 between them.
>>> nodes = torch.FloatTensor([
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
# Designate which nodes belong to which event
>>> batch = torch.LongTensor([0,0,0,0,0,1,1,1,1])

# Generate edges: k=2, max_radius^2 of 0.04
>>> torch_cmspepr.knn_graph(nodes, 2, batch, max_radius=.2)
tensor([[0, 1, 1, 2, 2, 3, 5, 6, 6, 7, 7, 8],
        [1, 0, 2, 1, 3, 2, 6, 5, 7, 6, 8, 7]])

# Generate edges: k=3 with loops allowed
>>> torch_cmspepr.knn_graph(nodes, 3, batch, max_radius=.2, loop=True)
tensor([[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8],
        [0, 1, 1, 0, 2, 2, 1, 3, 3, 2, 4, 5, 6, 6, 5, 7, 7, 6, 8, 8, 7]])

# If CUDA is available, the CUDA version of the knn_graph is used automatically:
>>> gpu = torch.device('cuda') 
>>> torch_cmspepr.knn_graph(nodes.to(gpu), 2, batch.to(gpu), max_radius=.2)
tensor([[0, 1, 1, 2, 2, 3, 5, 6, 6, 7, 7, 8],
        [1, 0, 2, 1, 3, 2, 6, 5, 7, 6, 8, 7]], device='cuda:0')
```


## Installation and requirements

v1 is tested with CUDA 11.7 and pytorch 2.0.
You should verify `nvcc` is available:

```console
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0
```

Also a `gcc` version of 5 or higher is recommended.

The package is not (yet) available on PyPI, so local installation is at the moment the
preferred installation method:

```bash
git clone git@github.com:cms-pepr/pytorch_cmspepr.git
cd pytorch_cmspepr
pip install -e .
```

Installing _only_ the CPU or CUDA extensions is supported:

```bash
FORCE_CPU_ONLY=1 pip install -e .  # Only compile C++ extensions
FORCE_CUDA_ONLY=1 pip install -e .  # Only compile CUDA extenstions
FORCE_CUDA pip install -e .  # Try to compile CUDA extenstion even if no device found
```

If you only want to test the compilation of the extensions:

```bash
python setup.py develop
```

### Containerization

It is recommended to install and run inside a container.
At the time of writing (29 Sep 2023), the [pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel](https://hub.docker.com/layers/pytorch/pytorch/2.0.0-cuda11.7-cudnn8-devel/images/sha256-96ccb2997a131f2455d70fb78dbb284bafe4529aaf265e344bae932c8b32b2a4?context=explore)
docker container works well.

Example Singularity instructions:

```bash
singularity pull docker://pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
singularity run --nv sifs/pytorch_2.0.0-cuda11.7-cudnn8-devel.sif
```

And then once in the container:

```bash
export PYTHONPATH="/opt/conda/lib/python3.10/site-packages"
python -m venv env
source env/bin/activate
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html # Make sure to pick the right torch and CUDA versions here
git clone git@github.com:cms-pepr/pytorch_cmspepr.git
cd pytorch_cmspepr
pip install -e .
```


## Tests

```bash
pip install pytest
pytest tests
```


## Performance

The following profiling code can be used:

```python
import time
import torch
import torch_cmspepr
import torch_cluster
gpu = torch.device('cuda')

def gen(cuda=False):
    # 10k nodes with 5 node features
    x = torch.rand((10000, 5))
    # Split nodes over 4 events with 2500 nodes/evt
    batch = torch.repeat_interleave(torch.arange(4), 2500)
    if cuda: x, batch = x.to(gpu), batch.to(gpu)
    return x, batch

def profile(name, unit):
    t0 = time.time()
    for _ in range(100): unit()
    print(f'{name} took {(time.time() - t0)/100.} sec/evt')

def cpu_cmspepr():
    x, batch = gen()
    torch_cmspepr.knn_graph(x, k=10, batch=batch)
profile('CPU (torch_cmspepr)', cpu_cmspepr)

def cpu_cluster():
    x, batch = gen()
    torch_cluster.knn_graph(x, k=10, batch=batch)
profile('CPU (torch_cluster)', cpu_cmspepr)

def cuda_cmspepr():
    x, batch = gen(cuda=True)
    torch_cmspepr.knn_graph(x, k=10, batch=batch)
profile('CUDA (torch_cmspepr)', cuda_cmspepr)

def cuda_cluster():
    x, batch = gen(cuda=True)
    torch_cluster.knn_graph(x, k=10, batch=batch)
profile('CUDA (torch_cluster)', cpu_cmspepr)
```

On a NVIDIA Tesla P100 with 12GB of RAM, this produces:

```
CPU (torch_cmspepr) took 0.22623349189758302 sec/evt
CPU (torch_cluster) took 0.2259768319129944 sec/evt
CUDA (torch_cmspepr) took 0.026673252582550048 sec/evt
CUDA (torch_cluster) took 0.22262062072753908 sec/evt
```