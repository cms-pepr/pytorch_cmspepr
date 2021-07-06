import torch

from typing import Optional, Tuple

class SelectKNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                x: torch.Tensor, 
                k: int, 
                inmask: Optional[torch.Tensor] = None, 
                max_radius: float = 1e9, 
                mask_mode: int = 1):

        x = x.view(-1, 1) if x.dim() == 1 else x
        x = x.contiguous()

        mask: torch.Tensor = torch.ones(x.shape[0], dtype=torch.int32, device=x.device)
        if inmask is not None:
            mask = inmask

        row_splits: torch.Tensor = torch.tensor([0, x.shape[0]], dtype=torch.int32, device=x.device)
        
        #if batch_x is not None:
        #    assert x.size(0) == batch_x.numel()
        #    batch_size = int(batch_x.max()) + 1

        #    deg = x.new_zeros(batch_size, dtype=torch.long)
        #    deg.scatter_add_(0, batch_x, torch.ones_like(batch_x))

        #    ptr_x = deg.new_zeros(batch_size + 1)
        #    torch.cumsum(deg, 0, out=ptr_x[1:])


        outputs =  torch.ops.torch_cmspepr.select_knn(coords = x, 
                                                      row_splits = row_splits, 
                                                      mask = mask, 
                                                      n_neighbours = k, 
                                                      max_radius = max_radius, 
                                                      mask_mode = mask_mode)
        idx_tensor, dist_tensor = outputs
        # saving the outputs and inputs for the backward pass
        ctx.save_for_backward(x, idx_tensor, dist_tensor)

        return idx_tensor, dist_tensor

    @staticmethod
    def backward(ctx, gradDistances, gradIdxs):
        x, idx_tensor, dist_tensor = ctx.saved_tensors
        grad_output = torch.ops.torch_cmspepr.select_knn_grad(gradDistances = gradDistances, 
                                                              indices = idx_tensor, 
                                                              distances = dist_tensor, 
                                                              coordinates = x)

        return grad_output
            
