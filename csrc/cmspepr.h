#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor>
accumulate_knn(torch::Tensor distances,
               torch::Tensor features,
               torch::Tensor indices,
               int64_t n_moments = 1,
               bool mean_and_max = true);

std::tuple<torch::Tensor, torch::Tensor>
accumulate_knn_grad(torch::Tensor grad_from_out_features, 
		    torch::Tensor distances, 
		    torch::Tensor features,
		    torch::Tensor neigh_indices, 
		    torch::Tensor max_feat_indices);

std::tuple<torch::Tensor, torch::Tensor>
select_knn(torch::Tensor coords,
	   torch::Tensor row_splits,
	   torch::Tensor mask,
	   int64_t n_neighbours,
	   double max_radius,
	   int64_t mask_mode = 1);

torch::Tensor
select_knn_grad(torch::Tensor gradDistances,
                torch::Tensor indices,
                torch::Tensor distances,
                torch::Tensor coordinates);


