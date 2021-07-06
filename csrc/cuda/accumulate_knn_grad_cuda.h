#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> accumulate_knn_grad_cuda(torch::Tensor grad_from_out_features,
    torch::Tensor distances,
    torch::Tensor features,
    torch::Tensor neigh_indices,
    torch::Tensor max_feat_indices);


