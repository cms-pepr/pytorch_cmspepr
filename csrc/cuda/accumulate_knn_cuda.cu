#include "accumulate_knn_cuda.h"
#include "utils.cuh"
#include "helpers.h"
#include "cuda_helpers.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t distanceWeight(const scalar_t& distsq) 
{
    return exp(-1. * ACCUMULATE_KNN_EXPONENT * distsq);
}

template <typename scalar_t>
__global__ void acc_knn_kernel(
    const scalar_t*d_distances,
    const scalar_t*d_feat,
    const int32_t *d_idxs,

    scalar_t *d_out_feat,
    int32_t *d_out_maxidxs,

    int64_t n_vert,
    int64_t n_neigh,
    int64_t n_feat,

    int64_t n_out_feat,

    int64_t n_moments,
    bool mean_and_max) {

    size_t i_v = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i_f = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_v >= n_vert || i_f >= n_feat)
        return;

    float t_mean = 0;
    float t_max = 0;
    int max_i_n_gidx = 0;

    //parallel over neigh? (requires atmomic add later)
    for (size_t i_n = 0; i_n < n_neigh; i_n++) {

        int nidx = d_idxs[I2D(i_v, i_n, n_neigh)];

        if (nidx < 0) continue;

        float vnf = d_feat[I2D(nidx, i_f, n_feat)];
        float distsq = d_distances[I2D(i_v, i_n, n_neigh)];
        float wfeat = vnf * distanceWeight(distsq);
        t_mean += wfeat;
        if (mean_and_max && (wfeat >= t_max || !i_n)) {
            max_i_n_gidx = nidx;
            t_max = wfeat;
        }
    }
    t_mean /= (float)n_neigh;

    d_out_feat[I2D(i_v, i_f, n_out_feat)] = t_mean;
    if (mean_and_max) {
        d_out_maxidxs[I2D(i_v, i_f, n_feat)] = max_i_n_gidx; //just used for gradient
        d_out_feat[I2D(i_v, i_f + n_feat, n_out_feat)] = t_max;
    }

}

std::tuple<torch::Tensor, torch::Tensor>
accumulate_knn_cuda(torch::Tensor distances, 
                    torch::Tensor features, 
                    torch::Tensor indices,	      
                    int n_moments,
                    bool mean_and_max) {

    // Check for the device
    CHECK_CUDA(distances);
    CHECK_CUDA(indices);
    CHECK_CUDA(features);
   
    cudaSetDevice(distances.get_device());

    const auto n_vert = distances.size(0);
    const auto n_neigh = indices.size(1);
    const auto n_coords = distances.size(1);
    const auto n_feat = features.size(1);

    assert(n_vert == indices.size(0) && n_vert == features.size(0));
    assert(n_neigh == distances.size(1));

    int64_t n_out_feat = n_feat;
    if (mean_and_max) {
        n_out_feat *= 2;
    }
	auto output_feat_tensor = torch::zeros({ n_vert,n_out_feat },
       				 torch::TensorOptions().dtype(torch::kFloat32).device(distances.device()));
    	auto output_max_idxs_tensor = torch::zeros({ n_vert,n_feat },
        			torch::TensorOptions().dtype(torch::kInt32).device(distances.device()));    

    auto stream = at::cuda::getCurrentCUDAStream();

    grid_and_block par(n_vert, 512, n_feat, 2);
    
    AT_DISPATCH_FLOATING_TYPES(output_feat_tensor.scalar_type(), "acc_knn_kernel", ([&] {
        acc_knn_kernel<scalar_t> << <par.grid(), par.block(), 0, stream >> > (
            distances.data_ptr<scalar_t>(),
            features.data_ptr<scalar_t>(),
            indices.data_ptr<int32_t>(),

            output_feat_tensor.data_ptr<scalar_t>(),
            output_max_idxs_tensor.data_ptr<int32_t>(),

            n_vert,
            n_neigh,
            n_feat,

            n_out_feat,

            n_moments,
            mean_and_max
            );
        }));

    cudaDeviceSynchronize();

    return std::make_tuple(output_feat_tensor, output_max_idxs_tensor);

}
