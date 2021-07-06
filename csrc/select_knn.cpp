#include <Python.h>
#include <torch/script.h>

#include "cpu/select_knn_cpu.h"

#ifdef WITH_CUDA
#include "cuda/select_knn_cuda.h"
#endif


#ifdef _WIN32
#ifdef WITH_CUDA
PyMODINIT_FUNC PyInit__select_knn_cuda(void) { return NULL; }
#else
PyMODINIT_FUNC PyInit__select_knn_cpu(void) { return NULL; }
#endif
#endif

std::tuple<torch::Tensor, torch::Tensor>
select_knn(torch::Tensor coords,
           torch::Tensor row_splits,
           torch::Tensor mask,
           int64_t n_neighbours,
           double max_radius,
           int64_t mask_mode) {
   if (coords.device().is_cuda()) {
#ifdef WITH_CUDA
     return select_knn_cuda(coords,
			    row_splits,
			    mask,
			    n_neighbours,
			    max_radius,
			    mask_mode);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
  } else {
     return select_knn_cpu(coords,
			   row_splits,
			   mask,
			   n_neighbours,
			   max_radius,
			   mask_mode);
  }
}

static auto registry =
  torch::RegisterOperators().op("torch_cmspepr::select_knn", &select_knn);
