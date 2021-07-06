#include "select_knn_grad_cuda.h"
#include "utils.cuh"
#include "helpers.h"
#include "cuda_helpers.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

template <typename scalar_t>
__global__ static void select_knn_grad_selfloop_kernel(
    const scalar_t *d_grad_dist, // V x N
    const int32_t *d_neigh_indices,
    const scalar_t *d_dist,
    const scalar_t *d_coord,

    scalar_t *d_grad_coord,

    const size_t n_vert,
    const size_t n_neigh,
    const size_t n_coords) {

    size_t i_v = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_v >= n_vert)
        return;

    size_t nu_c = blockIdx.y * blockDim.y + threadIdx.y;
    if (nu_c >= n_coords)
        return;

    const float xinu = d_coord[I2D(i_v, nu_c, n_coords)];

    float self_contrib = 0;
    for (size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++) {

        int k = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        if (k < 0) break;

        const float gik = d_grad_dist[I2D(i_v, i_i_n, n_neigh)];
        const float xknu = d_coord[I2D(k, nu_c, n_coords)];


        self_contrib -= 2. * gik * (xknu - xinu);

    }
    d_grad_coord[I2D(i_v, nu_c, n_coords)] = self_contrib;
}

template <typename scalar_t>
__global__  static void select_knn_grad_neighloop_kernel(
    const scalar_t*d_grad_dist, // V x N
    const int32_t *d_neigh_indices,
    const scalar_t *d_dist,
    const scalar_t *d_coord,

    scalar_t *d_grad_coord,

    const size_t n_vert,
    const size_t n_neigh,
    const size_t n_coords) {


    size_t i_v = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_v >= n_vert)
        return;

    size_t nu_c = blockIdx.y * blockDim.y + threadIdx.y;
    if (nu_c >= n_coords)
        return;

    const float xinu = d_coord[I2D(i_v, nu_c, n_coords)];

    for (size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++) {

        int m = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];
        if (m < 0) break;//padded with -1

        const float gim = d_grad_dist[I2D(i_v, i_i_n, n_neigh)];
        const float xmnu = d_coord[I2D(m, nu_c, n_coords)];

        float add = 2. * gim * (xmnu - xinu);
	atomicAdd(&d_grad_coord[I2D(m, nu_c, n_coords)], add);
//        d_grad_coord[I2D(m, nu_c, n_coords)] += add;

    }
}

torch::Tensor select_knn_grad_cuda(
    torch::Tensor gradDistances, 
    torch::Tensor indices,
    torch::Tensor distances, 
    torch::Tensor coordinates) 
{
    CHECK_CUDA(gradDistances);
    CHECK_CUDA(indices);
    CHECK_CUDA(distances);
    CHECK_CUDA(coordinates);

    cudaSetDevice(gradDistances.get_device());
    
    const auto n_vert = coordinates.size(0);
    const auto n_coords = coordinates.size(1);
    const auto n_neigh = distances.size(1);

    auto output_tensor = torch::zeros({ n_vert, n_neigh },
        torch::TensorOptions().dtype(torch::kFloat32).device(gradDistances.device()));

    grid_and_block gb(n_vert, 256, n_coords, 4);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(output_tensor.scalar_type(), "select_knn_grad_selfloop_kernel", [&] {
        select_knn_grad_selfloop_kernel<scalar_t> << <gb.grid(), gb.block(), 0, stream >> > (
            gradDistances.data_ptr<scalar_t>(),
            indices.data_ptr<int32_t>(),

            distances.data_ptr<scalar_t>(),
            coordinates.data_ptr<scalar_t>(),

            output_tensor.data_ptr<scalar_t>(),

            n_vert,
            n_neigh,
            n_coords);
        });

    cudaDeviceSynchronize();

    AT_DISPATCH_FLOATING_TYPES(output_tensor.scalar_type(), "select_knn_grad_neighloop_kernel", [&] {
        select_knn_grad_neighloop_kernel<scalar_t> << <gb.grid(), gb.block(), 0, stream >> > (
            gradDistances.data_ptr<scalar_t>(),
            indices.data_ptr<int32_t>(),

            distances.data_ptr<scalar_t>(),
            coordinates.data_ptr<scalar_t>(),

            output_tensor.data_ptr<scalar_t>(),

            n_vert,
            n_neigh,
            n_coords);
        });


    return output_tensor;
}
