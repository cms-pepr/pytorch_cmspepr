#include "accumulate_knn_cpu.h"
#include "accumulate_knn_grad_cpu.h"
#include <helpers.h>
#include "utils.h"

#include <torch/extension.h>
#include <string> //size_t, just for helper function
#include <cmath>

inline static float distanceWeight(const float& distsq) {
    return exp(-1. * ACCUMULATE_KNN_EXPONENT * distsq);
}

static void set_feature_grad_zero(
    float_t *d_out_grad_features,
    size_t n_vert,
    size_t n_feat)
{

    for (size_t i_v = 0; i_v < n_vert; i_v++) {
        for (size_t i_f = 0; i_f < n_feat; i_f++)
            d_out_grad_features[I2D(i_v, i_f, n_feat)] = 0;
    }
}

static void calc_feature_gradients(
    const float_t *d_grad_from_out_features,
    const int32_t *d_max_feat_indices,
    const int32_t *d_neigh_indices,
    const float_t *d_distances,

    const size_t n_vert,
    const size_t n_feat,
    const size_t n_neigh,

    const int32_t n_grad_from_out_feat,

    float_t *d_out_grad_features,
    bool mean_and_max)
{
    for (size_t i_v = 0; i_v < n_vert; i_v++) {
        for (size_t nu_f = 0; nu_f < n_feat; nu_f++) {

            const float ginu = d_grad_from_out_features[I2D(i_v, nu_f, n_grad_from_out_feat)];
            float ginu_max = 0;
            int32_t max_for_iv = -1;
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
                d_out_grad_features[I2D(m_v, nu_f, n_feat)] += mean_contrib + max_contrib;

                //     }
            }
        }
    }
}

static void calc_distance_gradients(
    const float_t *d_grad_from_out_features,
    const int32_t *d_max_feat_indices,
    const int32_t *d_neigh_indices,
    const float_t *d_distances,
    const float_t *d_feat,

    const size_t n_vert,
    const size_t n_feat,
    const size_t n_neigh,

    const int n_grad_from_out_feat,

    float_t *d_out_grad_distances,
    bool mean_and_max)
{
    for (size_t m = 0; m < n_vert; m++) {

        for (size_t l = 0; l < n_neigh; l++) {


            int l_g = d_neigh_indices[I2D(m, l, n_neigh)];
            if (l_g < 0) {
                d_out_grad_distances[I2D(m, l, n_neigh)] = 0;
                continue;
            }

            float mean_contrib = 0;
            float max_contrib = 0;

            float dml = d_distances[I2D(m, l, n_neigh)]; //dlm == dml
            float expml = distanceWeight(dml);

            for (size_t b_f = 0; b_f < n_feat; b_f++) {

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
    }
}


void compute(const float_t *d_grad_from_out_features,
    const float_t *d_distances,
    const float_t *d_feat,
    const int32_t *d_max_feat_indices,
    const int32_t *d_neigh_indices,

    float_t *d_out_grad_distances,
    float_t *d_out_grad_features,

    const size_t n_vert,
    const size_t n_neigh,
    const size_t n_feat,

    const int n_grad_from_out_feat,

    int n_moments,
    bool mean_and_max)
{
    //set zero
    set_feature_grad_zero(
        d_out_grad_features,
        n_vert,
        n_feat);

    //compute feature gradients
    calc_feature_gradients(
        d_grad_from_out_features,
        d_max_feat_indices,
        d_neigh_indices,
        d_distances,

        n_vert,
        n_feat,
        n_neigh,

        n_grad_from_out_feat,

        d_out_grad_features,
        mean_and_max);

    // compute distance gradients
    calc_distance_gradients(
        d_grad_from_out_features,
        d_max_feat_indices,
        d_neigh_indices,
        d_distances,
        d_feat,

        n_vert,
        n_feat,
        n_neigh,

        n_grad_from_out_feat,

        d_out_grad_distances,
        mean_and_max);

}

std::tuple<torch::Tensor, torch::Tensor> accumulate_knn_grad_cpu(
    torch::Tensor grad_from_out_features,
    torch::Tensor distances, 
    torch::Tensor features,
    torch::Tensor neigh_indices, 
    torch::Tensor max_feat_indices)
{
    CHECK_CPU(grad_from_out_features);
    CHECK_CPU(distances);
    CHECK_CPU(features);
    CHECK_CPU(neigh_indices);
    CHECK_CPU(max_feat_indices);

    const auto n_in_grad_feat = grad_from_out_features.size(1);
    const auto n_vert = grad_from_out_features.size(0);
    const auto n_neigh = neigh_indices.size(1);
    const auto n_feat = features.size(1);

    int n_moments = 0;
    bool mean_and_max = false;

    if (n_in_grad_feat > n_feat) {
        mean_and_max = true;
    }

    // declaring output tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto out_grad_distances = torch::zeros({ n_vert, n_neigh }, options);

    auto optionsFeat = torch::TensorOptions().dtype(torch::kFloat32);
    auto out_grad_features = torch::zeros({ n_vert, n_feat }, optionsFeat);

    // Input pointers to the compute function
    auto grad_from_out_feat_data = grad_from_out_features.data_ptr<float_t>();
    auto distances_data = distances.data_ptr<float_t>();
    auto features_data = features.data_ptr<float_t>();
    auto max_feat_indices_data = max_feat_indices.data_ptr<int32_t>();
    auto neigh_indices_data = neigh_indices.data_ptr<int32_t>();

    auto out_grad_dist_data = out_grad_distances.data_ptr<float_t>();
    auto out_grad_feat_data = out_grad_features.data_ptr<float_t>();

    // calling compute 
    compute(grad_from_out_feat_data, 
        distances_data, 
        features_data,
        max_feat_indices_data, 
        neigh_indices_data, 
        out_grad_dist_data,
        out_grad_feat_data, 
       
        n_vert, 
        n_neigh, 
        n_feat, 
        
        n_in_grad_feat,
        
        n_moments, 
        mean_and_max);

    return std::make_tuple(out_grad_distances, out_grad_features);
}


