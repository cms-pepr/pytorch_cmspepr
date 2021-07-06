#include <Python.h>
#include <torch/script.h>

#include "cpu/accumulate_knn_grad_cpu.h"

#ifdef WITH_CUDA
#include "cuda/accumulate_knn_grad_cuda.h"
#endif


#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__accumulate_knn_grad_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__accumulate_knn_grad_cpu(void) { return NULL; }
#endif
#endif

std::tuple<torch::Tensor, torch::Tensor>
accumulate_knn_grad(torch::Tensor grad_from_out_features,
		    torch::Tensor distances,
		    torch::Tensor features,
		    torch::Tensor neigh_indices,
		    torch::Tensor max_feat_indices) {
   if (distances.device().is_cuda()) {
#ifdef WITH_CUDA
     return accumulate_knn_grad_cuda(grad_from_out_features,
				     distances,
				     features,
				     neigh_indices,
				     max_feat_indices);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return accumulate_knn_grad_cpu(grad_from_out_features,
				   distances,
				   features,
				   neigh_indices,
				   max_feat_indices);
  }
}

static auto registry =
  torch::RegisterOperators().op("torch_cmspepr::accumulate_knn_grad", &accumulate_knn_grad);
