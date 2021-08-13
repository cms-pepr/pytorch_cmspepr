#include "select_knn_cuda.h"
#include "utils.cuh"
#include "helpers.h"
#include "cuda_helpers.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

template <typename scalar_t>
__device__ scalar_t calculateDistance(
    size_t i_v, 
    size_t j_v, 
    const scalar_t *d_coord, 
    size_t n_coords) 
{
    scalar_t distsq = 0;
    // if (i_v == j_v)
    //     return 0;
    for (size_t i = 0; i < n_coords; i++) {
        scalar_t dist = d_coord[I2D(i_v,i,n_coords)] - d_coord[I2D(j_v,i,n_coords)];
        distsq += dist * dist;
    }

    return distsq;
}

template <typename scalar_t>
__device__ int32_t searchLargestDistance(
    int32_t i_v, 
    scalar_t* d_dist, 
    int32_t n_neigh, 
    scalar_t& maxdist) {

    maxdist = 0;
    int32_t maxidx = 0;
    if (n_neigh < 2)
        return maxidx;
    for (int32_t n = 1; n < n_neigh; n++) { //0 is self
        scalar_t distsq = d_dist[I2D(i_v, n, n_neigh)];
	bool isgreater = distsq > maxdist;
	bool isless = !isgreater;
	maxdist = distsq*isgreater + maxdist*isless;
	maxidx = n*isgreater + maxidx*isless;
    }
    return maxidx;
}

template <typename scalar_t> 
__global__ void set_defaults(
    scalar_t *d_dist,
    int32_t* d_indices,
    int32_t n_vert,
    int32_t n_neigh)
{
    const int32_t i_v = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_v >= n_vert)
        return;
    const int32_t n = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= n_neigh)
        return;

    d_indices[I2D(i_v, n, n_neigh)] = i_v;
    d_dist[I2D(i_v, n, n_neigh)] = 0;
}

template <typename scalar_t> 
__global__ void select_knn_kernel(
    const scalar_t *d_coord,
    const int32_t *d_row_splits,
    const int32_t *d_mask,
    int32_t *d_indices,
    scalar_t *d_dist,

    const int32_t n_vert,
    const int32_t n_neigh,
    const int32_t n_coords,

    const int32_t j_rs,
    const scalar_t max_radius){

    //really no buffering at all here

    const int32_t start_vert = d_row_splits[j_rs];
    const int32_t end_vert = d_row_splits[j_rs + 1];

    const int32_t i_v = blockIdx.x * blockDim.x + threadIdx.x + start_vert;
    if (i_v >= end_vert || i_v >= n_vert)
        return;//this will be a problem with actual RS

    //protection against n_vert<n_neigh
    int32_t nvert_in_row = end_vert - start_vert;
    int32_t max_neighbours = n_neigh;
    //set default to self
    if (nvert_in_row < n_neigh) {
        max_neighbours = nvert_in_row;
    }
    int32_t nfilled = 1;
    int32_t maxidx_local = 0;
    scalar_t maxdistsq = 0;

    int32_t j_v = 0;
    for (j_v = start_vert; j_v < end_vert && nfilled < max_neighbours; j_v++) {
        scalar_t distsq = calculateDistance(i_v, j_v, d_coord, n_coords);
        if (i_v == j_v || (max_radius > 0 && distsq > max_radius)) continue;
        //fill up
        d_indices[I2D(i_v, nfilled, n_neigh)] = j_v;
        d_dist[I2D(i_v, nfilled, n_neigh)] = distsq;
        if (distsq > maxdistsq) {
            maxdistsq = distsq;
            maxidx_local = nfilled;
        }
        nfilled++;
    }
    // for the rest start from where we left off and only do index shuffling
    for(; j_v < end_vert; j_v++ ) {
        scalar_t distsq = calculateDistance(i_v, j_v, d_coord, n_coords);
        if (i_v == j_v || distsq > maxdistsq) continue;

	//replace former max
        d_indices[I2D(i_v, maxidx_local, n_neigh)] = j_v;
        d_dist[I2D(i_v, maxidx_local, n_neigh)] = distsq;
        //search new max
        maxidx_local = searchLargestDistance(i_v, d_dist, n_neigh, maxdistsq);
    }
}
/*
template <typename scalar_t>
__global__ void row_nn_search(
    const scalar_t *d_coord,
    int32_t *d_indices,
    scalar_t *d_dist,    
    const int32_t i_v,
    const int32_t start_vert,
    const int32_t n_vert,
    const int32_t n_neigh,
    const int32_t n_coords,
    const scalar_t max_radius
) {
    const int32_t j_v = blockIdx.x * blockDim.x + threadIdx.x + start_vert;

    scalar_t distsq = calculateDistance(i_v, j_v, d_coord, n_coords);

    if (j_v == i_v) continue;

    if (nfilled < max_neighbours && (max_radius <= 0 || max_radius >= distsq)) {
        d_indices[I2D(i_v, nfilled, n_neigh)] = j_v;
        d_dist[I2D(i_v, nfilled, n_neigh)] = distsq;
        if (distsq > maxdistsq) {
            maxdistsq = distsq;
            maxidx_local = nfilled;
        }
        nfilled++;
        continue;
    }
    if (distsq < maxdistsq) {// automatically applies to max radius
        //replace former max
        d_indices[I2D(i_v, maxidx_local, n_neigh)] = j_v;
        d_dist[I2D(i_v, maxidx_local, n_neigh)] = distsq;
        //search new max
        maxidx_local = searchLargestDistance(i_v, d_dist, n_neigh, maxdistsq);
    }
}
*/

std::tuple<torch::Tensor, torch::Tensor> select_knn_cuda(
    torch::Tensor coords,
    torch::Tensor row_splits,
    torch::Tensor mask,
    int64_t n_neighbours,
    double max_radius,
    int64_t mask_mode)
{
    CHECK_CUDA(coords);
    CHECK_CUDA(row_splits);
    CHECK_CUDA(mask);

    const auto n_vert = coords.size(0);
    const auto n_coords = coords.size(1);
    const auto n_rs = row_splits.size(0);
    const auto n_neigh = n_neighbours;

    if (max_radius > 0) {
        max_radius *= max_radius;
    }

    auto output_dist_tensor = torch::zeros({ n_vert, n_neighbours },
        torch::TensorOptions().dtype(coords.dtype()).device(coords.device()));
    auto output_idx_tensor = torch::zeros({ n_vert, n_neighbours },
        torch::TensorOptions().dtype(torch::kInt32).device(coords.device()));

    // get the grid and block values for parallel CUDA programming

    grid_and_block gb(n_vert, 256, n_neighbours, 4);

    AT_DISPATCH_FLOATING_TYPES(coords.type(), "set_defaults", ([&] {
        set_defaults <scalar_t> <<<gb.grid(), gb.block()>>> (
            output_dist_tensor.data_ptr<scalar_t>(),
            output_idx_tensor.data_ptr<int32_t>(),
            n_vert,
            n_neigh);
    }));

    std::vector<int32_t> cpu_rowsplits(n_rs);
    cudaMemcpy(&cpu_rowsplits.at(0), row_splits.data_ptr<int32_t>(), n_rs * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int32_t j_rs = 0; j_rs < n_rs - 1; j_rs++) {
        int32_t nvert_rs = cpu_rowsplits.at(j_rs + 1) - cpu_rowsplits.at(j_rs);
        grid_and_block gb(nvert_rs, 1024);

        AT_DISPATCH_FLOATING_TYPES(coords.type(), "select_knn_kernel", ([&] {
            select_knn_kernel <scalar_t> <<<gb.grid(), gb.block()>>> (
                    coords.data_ptr<scalar_t>(),
                    row_splits.data_ptr<int32_t>(),
                    mask.data_ptr<int32_t>(),
                    output_idx_tensor.data_ptr<int32_t>(),
                    output_dist_tensor.data_ptr<scalar_t>(),
                    
                    n_vert,
                    n_neigh,
                    n_coords,
                    
                    j_rs,
                    max_radius);
            }));
    }
    
    return std::make_tuple(output_idx_tensor, output_dist_tensor);

}
