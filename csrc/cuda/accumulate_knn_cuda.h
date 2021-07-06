#pragma once

#define ACCUMULATE_KNN_EXPONENT 1.

#include <torch/extension.h>
std::tuple<torch::Tensor, torch::Tensor>
accumulate_knn_cuda(torch::Tensor distances,
		    torch::Tensor features,
		    torch::Tensor indices,
		    int n_moments = 1,
		    bool mean_and_max = true);
