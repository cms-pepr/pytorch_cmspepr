import time
import torch
import torch_cmspepr
import torch_cluster
import tqdm
gpu = torch.device('cuda')

k = 10

def gen(cuda=False):
    # 10k nodes with 5 node features
    x = torch.rand((10000, 5))
    # Split nodes over 4 events with 2500 nodes/evt
    batch = torch.repeat_interleave(torch.arange(4), 2500)
    if cuda: x, batch = x.to(gpu), batch.to(gpu)
    return x, batch

def profile(name, unit):
    t0 = time.time()
    for _ in tqdm.tqdm(range(10)): unit()
    print(f'{name} took {(time.time() - t0)/100.} sec/evt')

def cpu_cmspepr():
    x, batch = gen()
    torch_cmspepr.knn_graph(x, k, batch=batch)
profile('CPU (torch_cmspepr)', cpu_cmspepr)

def cpu_cluster():
    x, batch = gen()
    torch_cluster.knn_graph(x, k, batch=batch)
profile('CPU (torch_cluster)', cpu_cmspepr)

def cuda_cmspepr():
    x, batch = gen(cuda=True)
    torch_cmspepr.knn_graph(x, k, batch=batch)
profile('CUDA (torch_cmspepr)', cuda_cmspepr)

def cuda_cluster():
    x, batch = gen(cuda=True)
    torch_cluster.knn_graph(x, k, batch=batch)
profile('CUDA (torch_cluster)', cpu_cmspepr)
