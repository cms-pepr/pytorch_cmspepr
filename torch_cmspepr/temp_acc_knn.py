import torch

class AccumulateKNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                distances: torch.Tensor, 
                features: torch.Tensor, 
                indices: torch.Tensor, 
                n_moments: int = 1, 
                mean_and_max: bool = True):
        output_feat_tensor, output_max_idxs_tensor = torch.ops.torch_cmspepr.accumulate_knn(distances = distances, 
                                                                                            features = features, 
                                                                                            indices = indices, 
                                                                                            n_moments = n_moments, 
                                                                                            mean_and_max = mean_and_max)
        ctx.save_for_backward(distances, features, indices, output_feat_tensor, output_max_idx_tensor)
        
        return output_feat_tensor, output_max_idxs_tensor       

    @staticmethod
    def backward(ctx, 
                 grad_from_out_features,
                 grad_max_idxs):
        distances, features, indices, output_feat_tensor, output_max_idx_tensor = ctx.saved_tensors
        grad_out_distances, grad_out_features = torch.ops.torch_cmspepr.accumulate_knn_grad(grad_from_out_features = grad_from_out_features,
                                                                                    distances = distances,
                                                                                    features = features,
                                                                                    neigh_indices = indices,
                                                                                    max_feat_indices = output_max_idx_tensor)

        return grad_out_distances, grad_out_features
