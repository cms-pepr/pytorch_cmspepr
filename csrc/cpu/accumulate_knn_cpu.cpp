#include "accumulate_knn_cpu.h"
#include "helpers.h"
#include "utils.h"

#include <torch/extension.h>
#include <string> //size_t, just for helper function
#include <cmath>
//#include <iostream> //remove later DEBUG FIXME

static inline float distanceWeight(const float& distsq) {
    return exp(-1. * ACCUMULATE_KNN_EXPONENT * distsq);
}

void compute(const float_t *d_distances,
    const float_t *d_feat,
    const int32_t *d_idxs,

    float_t *d_out_feat,
    int32_t *d_out_maxidxs,

    size_t n_vert,
    size_t n_neigh,
    size_t n_feat,

    size_t n_out_feat,

    size_t n_moments,
    bool mean_and_max)
{
    for (size_t i_v = 0; i_v < n_vert; i_v++) {

        for (size_t i_f = 0; i_f < n_feat; i_f++) {

            float t_mean = 0;
            float t_max = 0;
            int max_i_n_gidx = 0;

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
                d_out_maxidxs[I2D(i_v, i_f, n_feat)] = max_i_n_gidx;
                d_out_feat[I2D(i_v, i_f + n_feat, n_out_feat)] = t_max;
            }

        }
    }
}

std::tuple<torch::Tensor, torch::Tensor>
accumulate_knn_cpu(torch::Tensor distances, 
		   torch::Tensor features, 
		   torch::Tensor indices,
		   int n_moments, 
		   bool mean_and_max)
{
    const auto n_vert = distances.size(0);
    const auto n_neigh = indices.size(1);
    const auto n_coords = distances.size(1);
    const auto n_feat = features.size(1);

    assert(n_vert == indices.size(0) && n_vert == features.size(0));
    assert(n_neigh == distances.size(1));

    int64_t n_out_feat = n_feat;
    if (mean_and_max) {
        n_out_feat *= 2;    }

    auto output_feat_tensor = torch::zeros({ n_vert,n_out_feat }, 
                         torch::TensorOptions().dtype(torch::kFloat32));
    auto output_max_idxs_tensor = torch::zeros({ n_vert,n_feat }, 
                                  torch::TensorOptions().dtype(torch::kInt32));

    auto distances_data = distances.data_ptr<float_t>();
    auto features_data = features.data_ptr<float_t>();
    auto indices_data = indices.data_ptr<int32_t>();

    auto output_feat_tensor_data = output_feat_tensor.data_ptr<float_t>();
    auto output_max_idxs_data = output_max_idxs_tensor.data_ptr<int32_t>();

    compute(distances_data, 
        features_data, 
        indices_data,
        output_feat_tensor_data, 
        output_max_idxs_data,
        n_vert, 
        n_neigh, 
        n_feat, 
        n_out_feat,
        n_moments, 
        mean_and_max);

    return std::make_tuple(output_feat_tensor, output_max_idxs_tensor);
}
