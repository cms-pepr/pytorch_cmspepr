#include "accumulate_knn_cuda.h"
#include "accumulate_knn_grad_cuda.h"
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
__device__ __forceinline__ static float distanceWeight(const scalar_t& distsq) {
    return exp(-1. * ACCUMULATE_KNN_EXPONENT * distsq);
}

template <typename scalar_t>
__global__ static void set_feature_grad_zero(
    scalar_t *d_out_grad_features,
    int64_t n_vert,
    int64_t n_feat
) {

    const size_t i_v = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i_f = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_v >= n_vert || i_f >= n_feat)
        return;

    d_out_grad_features[I2D(i_v, i_f, n_feat)] = 0;

}

template <typename scalar_t>
__global__ static void calc_feature_gradients(
    const scalar_t *d_grad_from_out_features,
    const int32_t *d_max_feat_indices,
    const int32_t *d_neigh_indices,
    const scalar_t *d_distances,

    const int n_vert,
    const int n_feat,
    const int n_neigh,

    const int n_grad_from_out_feat,

    scalar_t *d_out_grad_features,
    bool mean_and_max)
{
    const size_t i_v = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t nu_f = blockIdx.y * blockDim.y + threadIdx.y;
    if (i_v >= n_vert || nu_f >= n_feat)
        return;


    const float ginu = d_grad_from_out_features[I2D(i_v, nu_f, n_grad_from_out_feat)];
    float ginu_max = 0;
    int max_for_iv = -1;
    if (mean_and_max) {
        ginu_max = d_grad_from_out_features[I2D(i_v, nu_f + n_feat, n_grad_from_out_feat)];
        max_for_iv = d_max_feat_indices[I2D(i_v, nu_f, n_feat)];
    }


    bool firstself = true;
    for (size_t i_i_n = 0; i_i_n < n_neigh; i_i_n++) {

        int m_v = d_neigh_indices[I2D(i_v, i_i_n, n_neigh)];

        if (m_v < 0) continue;

        const float distsq_im = d_distances[I2D(i_v, i_i_n, n_neigh)];

        const float weight_im = distanceWeight(distsq_im);

        //if weight_im > some number?
        //     for (size_t nu_f = 0; nu_f < n_feat; nu_f++){

        float mean_contrib = ginu / (float)n_neigh * weight_im;
        float max_contrib = 0;
        if (m_v == max_for_iv) {
            if (m_v == i_v) {
                if (firstself) {//count self just once
                    max_contrib = ginu_max * weight_im;
                    firstself = false;
                }
            }
            else {
                max_contrib = ginu_max * weight_im;
            }
        }

        //ATOMIC because of m_v which can occur in different threads. this is slow.. but needs to be atomic at least here...
        atomicAdd(&d_out_grad_features[I2D(m_v, nu_f, n_feat)], mean_contrib + max_contrib);


    }
}

template <typename scalar_t>
__global__ static void calc_distance_gradients(
    const scalar_t *d_grad_from_out_features,
    const int32_t *d_max_feat_indices,
    const int32_t *d_neigh_indices,
    const scalar_t *d_distances,
    const scalar_t *d_feat,

    const int64_t n_vert,
    const int64_t n_feat,
    const int64_t n_neigh,

    const int n_grad_from_out_feat,

    scalar_t *d_out_grad_distances,
    bool mean_and_max) 
{
    const size_t m = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t l = blockIdx.y * blockDim.y + threadIdx.y;

    if (m >= n_vert || l >= n_neigh)
        return;

    int l_g = d_neigh_indices[I2D(m, l, n_neigh)];
    if (l_g < 0) {
        d_out_grad_distances[I2D(m, l, n_neigh)] = 0;
        return;
    }

    float mean_contrib = 0;
    float max_contrib = 0;

    float dml = d_distances[I2D(m, l, n_neigh)]; //dlm == dml
    float expml = distanceWeight(dml);

    for (size_t b_f = 0; b_f < n_feat; b_f++) {
        __syncthreads();

        bool firstself = true; ///To be checked!!! this needs to be per feature and stored!

        float gmb = d_grad_from_out_features[I2D(m, b_f, n_grad_from_out_feat)];
        float gmbmax = 0;
        if (mean_and_max)
            gmbmax = d_grad_from_out_features[I2D(m, b_f + n_feat, n_grad_from_out_feat)];
        float flb = d_feat[I2D(l_g, b_f, n_feat)];

        mean_contrib += gmb * flb * expml;
        int maxform = -1;
        if (mean_and_max)
            maxform = d_max_feat_indices[I2D(m, b_f, n_feat)];
        if (l_g == maxform) {
            if (l_g == m) {
                if (firstself) {
                    max_contrib += gmbmax * flb * expml;
                    firstself = false;
                }
            }
            else {
                max_contrib += gmbmax * flb * expml;
            }
        }

    }
    mean_contrib *= -ACCUMULATE_KNN_EXPONENT / (float)n_neigh;
    max_contrib *= -ACCUMULATE_KNN_EXPONENT;

    d_out_grad_distances[I2D(m, l, n_neigh)] = mean_contrib + max_contrib;

}

std::tuple<torch::Tensor, torch::Tensor> accumulate_knn_grad_cuda(
    torch::Tensor grad_from_out_features, 
    torch::Tensor distances, 
    torch::Tensor features,
    torch::Tensor neigh_indices, 
    torch::Tensor max_feat_indices) 
{
    CHECK_CUDA(grad_from_out_features);
    CHECK_CUDA(distances);
    CHECK_CUDA(features);
    CHECK_CUDA(neigh_indices);
    CHECK_CUDA(max_feat_indices);

    cudaSetDevice(distances.get_device());

    int64_t n_in_grad_feat = grad_from_out_features.size(1);
    int64_t n_vert = grad_from_out_features.size(0);
    int64_t n_neigh = neigh_indices.size(1);
    int64_t n_feat = features.size(1);

    // int n_moments = 0;
    bool mean_and_max = false;

    if (n_in_grad_feat > n_feat) {
        mean_and_max = true;
    }

    // declaring output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(distances.device());
    auto out_grad_distances = torch::zeros({ n_vert, n_neigh }, options);

    auto optionsFeat = torch::TensorOptions().dtype(torch::kFloat32).device(distances.device());
    auto out_grad_features = torch::zeros({ n_vert, n_feat }, optionsFeat);

    grid_and_block feat_par(n_vert, 64, n_feat, 8);
    if (n_feat >= 32)
        feat_par = grid_and_block(n_vert, 16, n_feat, 32);
    if (n_feat >= 64)
        feat_par = grid_and_block(n_vert, 8, n_feat, 64);
    if (n_feat >= 128)
        feat_par = grid_and_block(n_vert, 4, n_feat, 128);

    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(out_grad_features.type(), "set_feature_grad_zero", ([&] {
        set_feature_grad_zero<scalar_t> << < feat_par.grid(), feat_par.block(), 0, stream >> > (
            out_grad_features.data_ptr<scalar_t>(),
            n_vert,
            n_feat);
        }));

    cudaDeviceSynchronize();

    AT_DISPATCH_FLOATING_TYPES(out_grad_features.type(), "calc_feature_gradients", ([&] {
        calc_feature_gradients<scalar_t> << < feat_par.grid(), feat_par.block(), 0, stream >> > (
            grad_from_out_features.data_ptr<scalar_t>(),
            max_feat_indices.data_ptr<int32_t>(),
            neigh_indices.data_ptr<int32_t>(),
            distances.data_ptr<scalar_t>(),

            n_vert,
            n_feat,
            n_neigh,

            n_in_grad_feat,

            out_grad_features.data_ptr<scalar_t>(),
            mean_and_max);
        }));

    cudaDeviceSynchronize();
    
    grid_and_block neigh_par(n_vert, 128, n_neigh, 4);

    AT_DISPATCH_FLOATING_TYPES(out_grad_features.type(), "calc_distance_gradients", ([&] {
        calc_distance_gradients<scalar_t> << <neigh_par.grid(), neigh_par.block(), 0, stream >> > (
            grad_from_out_features.data_ptr<scalar_t>(),
            max_feat_indices.data_ptr<int32_t>(),
            neigh_indices.data_ptr<int32_t>(),
            distances.data_ptr<scalar_t>(),
            features.data_ptr<scalar_t>(),

            n_vert,
            n_feat,
            n_neigh,

            n_in_grad_feat,

            out_grad_distances.data_ptr<scalar_t>(),
            mean_and_max);
        }));

    return std::make_tuple(out_grad_distances, out_grad_features);
}

