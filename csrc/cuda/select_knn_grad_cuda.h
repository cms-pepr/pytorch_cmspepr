#pragma once

#include <torch/extension.h>

torch::Tensor
select_knn_grad_cuda(torch::Tensor gradDistances,
		     torch::Tensor indices,
		     torch::Tensor distances,
		     torch::Tensor coordinates);

