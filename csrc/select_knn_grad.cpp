#include <Python.h>
#include <torch/script.h>

#ifdef WITH_CUDA
#include "cuda/select_knn_grad_cuda.h"
#endif


#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__select_knn_grad_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__select_knn_grad_cpu(void) { return NULL; }
#endif
#endif

torch::Tensor
select_knn_grad(torch::Tensor gradDistances,
		torch::Tensor indices,
		torch::Tensor distances,
		torch::Tensor coordinates) {
   if (gradDistances.device().is_cuda()) {
#ifdef WITH_CUDA
     return select_knn_grad_cuda(gradDistances,
				 indices,
				 distances,
				 coordinates);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
     AT_ERROR("cmspepr::select_knn_grad is not supported for CPU execution");
  }
}

static auto registry =
  torch::RegisterOperators().op("torch_cmspepr::select_knn_grad", &select_knn_grad);
