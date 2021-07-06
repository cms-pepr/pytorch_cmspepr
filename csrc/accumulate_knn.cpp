#include <Python.h>
#include <torch/script.h>

#include "cpu/accumulate_knn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/accumulate_knn_cuda.h"
#endif


#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__accumulate_knn_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__accumulate_knn_cpu(void) { return NULL; }
#endif
#endif

std::tuple<torch::Tensor, torch::Tensor>
accumulate_knn(torch::Tensor distances,
	       torch::Tensor features,
	       torch::Tensor indices,
	       int64_t n_moments,
	       bool mean_and_max) {
   if (distances.device().is_cuda()) {
#ifdef WITH_CUDA
     return accumulate_knn_cuda(distances,
				features,
				indices,
				n_moments,
				mean_and_max);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
    return accumulate_knn_cpu(distances,
			      features,
			      indices,
			      n_moments,
			      mean_and_max);
  }
}

static auto registry =
  torch::RegisterOperators().op("torch_cmspepr::accumulate_knn", &accumulate_knn);
