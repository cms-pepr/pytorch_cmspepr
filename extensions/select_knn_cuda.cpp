#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::tuple<torch::Tensor, torch::Tensor> select_knn_cuda_fn(
    torch::Tensor coords, 
    torch::Tensor row_splits,
    torch::Tensor mask, 
    int64_t n_neighbours, 
    double max_radius,
    int64_t mask_mode
    );

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> select_knn_cuda_interface(
    torch::Tensor coords, 
    torch::Tensor row_splits,
    torch::Tensor mask, 
    int64_t n_neighbours, 
    double max_radius,
    int64_t mask_mode
    ){
  CHECK_INPUT(coords);
  CHECK_INPUT(row_splits);
  CHECK_INPUT(mask);
  return select_knn_cuda_fn(
    coords, row_splits, mask, n_neighbours, max_radius, mask_mode
    );
}

TORCH_LIBRARY(select_knn_cuda, m) {
  m.def("select_knn_cuda", select_knn_cuda_interface);
}