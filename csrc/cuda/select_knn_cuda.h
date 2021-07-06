#pragma once

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor>
select_knn_cuda(torch::Tensor coords,
		torch::Tensor row_splits,
		torch::Tensor mask,
		int64_t n_neighbours,
		double max_radius,
		int64_t mask_mode);

