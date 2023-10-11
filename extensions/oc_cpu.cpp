#include <torch/extension.h>

// #include <string> //size_t, just for helper function
#include <cmath>
// #include <iostream>

#define CHECK_CPU(x) AT_ASSERTM(x.device().is_cpu(), #x " must be CPU tensor")
#define I2D(i,j,Nj) j + Nj*i

/*
Returns the squared distance between two nodes in clustering space.
*/
float calc_dist_sq(
    const size_t i, // index of node i
    const size_t j, // index of node j
    const float_t *x, // node feature matrix
    const size_t ndim // number of dimensions 
    ){
    float_t distsq = 0;
    if (i == j) return 0;
    // std::cout << "dist_sq i=" << i << " j=" << j << std::endl;
    for (size_t idim = 0; idim < ndim; idim++) {
        float_t dist = x[I2D(i,idim,ndim)] - x[I2D(j,idim,ndim)];
        // std::cout
        //     << "  idim=" << idim
        //     << " x[" << i << "][" << idim << "]=" << x[I2D(i,idim,ndim)]
        //     << " x[" << j << "][" << idim << "]=" << x[I2D(j,idim,ndim)]
        //     << " d=" << dist
        //     << " d_sq=" << dist*dist
        //     << std::endl;
        distsq += dist * dist;
    }
    // std::cout << "  d_sq_sum=" << distsq << std::endl;
    return distsq;
    }


void oc_kernel(
    // Global event info
    const float_t* beta, // beta per node
    const float_t* q,    // charge per node
    const float_t* x,    // cluster space coordinates
    const size_t n_dim_cluster_space, // Number of dimensions of the cluster space
    const int32_t* cond_indices,     // indices of the condensation points
    const int32_t* cond_counts,      // nr of nodes connected to the cond point
    const size_t cond_indices_start, // row split start for cond points
    const size_t cond_indices_end,   // row split end for cond points
    const int32_t* which_cond_point, // (n_nodes,) array pointing to the cond point index
    const int32_t n_nodes, // Number of nodes in the event of this node

    // To be parallellized over
    const size_t i_node, // index of the node in question

    // Outputs:
    float_t * V_att,
    float_t * V_rep,
    float_t * V_srp
    ){

    int32_t i_cond = which_cond_point[i_node];

    // std::cout
    //     << "i_node=" << i_node
    //     << " i_cond=" << i_cond
    //     << " q[i_node]=" << q[i_node]
    //     << " cond_start=" << cond_indices_start
    //     << " cond_end=" << cond_indices_end
    //     << " n_nodes=" << n_nodes
    //     << std::endl;

    // V_att and V_srp
    if (i_cond == -1 || i_node == (size_t)i_cond){
        // Noise node, or a condensation point itself
        // std::cout << "  Noise hit or cond point, V_att/V_srp set to 0." << std::endl;
        *V_att = 0.;
        *V_srp = 0.;
        }
    else {
        float d_sq = calc_dist_sq(i_node, i_cond, x, n_dim_cluster_space);
        float d = sqrt(d_sq);
        float_t d_huber = d+0.00001 <= 4.0 ?  d_sq  :  2.0 * 4.0 * (d - 4.0) ;
        *V_att = d_huber * q[i_node] * q[i_cond] / (float)n_nodes;
        // V_srp must still be normalized! This is done in the V_rep loop because the
        // normalization numbers are easier to access there.
        *V_srp = 1. / (20.*d_sq + 1.);
        // std::cout << "  d_huber for i_node " << i_node << ": "
        //     << d_huber
        //     << "; d_sq=" << d_sq
        //     << "; V_att=" << *V_att
        //     << "; V_srp=" << *V_srp
        //     << std::endl;
        }

    // V_rep
    *V_rep = 0.;
    for (size_t i=cond_indices_start; i<cond_indices_end; i++) {
        int32_t i_cond_other = cond_indices[i];
        if (i_cond_other == i_cond){
            // Still have to normalize V_srp; this is a convenient albeit awkward time
            // to do so.
            *V_srp *= -beta[i_cond] / (float)cond_counts[i] / (float)(cond_indices_end-cond_indices_start);
            // Should not repulse from own cond point, so skip V_rep calculation
            continue;
            }
        float d_sq = calc_dist_sq(i_node, i_cond_other, x, n_dim_cluster_space);
        float V_rep_this = exp(-4.0 * d_sq) * q[i_node] * q[i_cond_other];
        if (V_rep_this < 0.) V_rep_this = 0.;
        *V_rep += V_rep_this;
        }
    *V_rep /= (float)n_nodes;
    }


torch::Tensor
oc_cpu(
        torch::Tensor beta_tensor,
        torch::Tensor q_tensor,
        torch::Tensor x_tensor,
        torch::Tensor y_tensor,
        torch::Tensor row_splits_tensor
        ){

    const size_t n_nodes = q_tensor.size(0);
    const auto n_dim_cluster_space = x_tensor.size(1);
    const size_t n_events = row_splits_tensor.size(0) - 1;

    // std::cout
    //     << "n_nodes=" << n_nodes
    //     << " n_dim_cluster_space=" << n_dim_cluster_space
    //     << " n_events=" << n_events
    //     << std::endl;

    auto beta = beta_tensor.data_ptr<float_t>();
    auto q = q_tensor.data_ptr<float_t>();
    auto x = x_tensor.data_ptr<float_t>();
    auto y = y_tensor.data_ptr<int32_t>();
    auto row_splits = row_splits_tensor.data_ptr<int32_t>();

    // Determine number of condensation points per event (and total)
    size_t n_cond = 0;
    int32_t* n_cond_per_event = (int32_t *)malloc(n_events * sizeof(int32_t));
    for (size_t i_event=0; i_event<n_events; i_event++) {
        int32_t y_max = 0;
        for (int32_t i_node=row_splits[i_event]; i_node<row_splits[i_event+1]; i_node++)
            if (y[i_node] > y_max) y_max = y[i_node];
        n_cond_per_event[i_event] = y_max;
        n_cond += y_max;
    }

    // Determine the row splits for cond points
    // e.g. n_cond_per_event = [2, 4, 1]
    // then cond_indices_row_splits = [0, 2, 6, 7]
    int32_t* cond_indices_row_splits = (int32_t *)malloc((n_events+1) * sizeof(int32_t));
    cond_indices_row_splits[0] = 0;
    for (size_t i_event=0; i_event<n_events; i_event++)
        cond_indices_row_splits[i_event+1] = cond_indices_row_splits[i_event] + n_cond_per_event[i_event];

    // Determine the condensation point indices per event
    // (basically scatter_max(scatter_max(...)) )
    // O(N) complexity, but could kernelize at a later stage.
    int32_t* cond_indices = (int32_t *)malloc(n_cond * sizeof(int32_t));
    int32_t* cond_counts = (int32_t *)malloc(n_cond * sizeof(int32_t));
    size_t i_cond_indices_filler = 0;
    int32_t* which_cond_point = (int32_t *)malloc(n_nodes * sizeof(int32_t));
    for (size_t i_event=0; i_event<n_events; i_event++) {
        // Open up two arrays, both sized nr of cond points in this event:
        // - q_max, which holds the max charge found so far per cond point
        // - count, which holds the number of nodes per cluster / cond point
        // - i_max, which holds the index where the max charge was found.
        size_t n_cond_this_event = n_cond_per_event[i_event];
        float_t* q_max = (float_t *)malloc(n_cond_this_event * sizeof(float_t));
        int32_t* count = (int32_t *)malloc(n_cond_this_event * sizeof(int32_t));
        for(size_t i=0; i<n_cond_this_event; i++){
            q_max[i] = 0.;
            count[i] = 0;
            }
        int32_t* i_max = (int32_t *)malloc(n_cond_this_event * sizeof(int32_t));
        // Loop over nodes in event, overwrite q_max and i_max when necessary
        for (int32_t i_node=row_splits[i_event]; i_node<row_splits[i_event+1]; i_node++){
            int32_t y_node = y[i_node];
            if (y_node == 0) continue; // Bkg nodes don't belong to a cond point
            count[y_node-1]++;
            if (q[i_node] > q_max[y_node-1]){
                // std::cout
                //     << "i_node=" << i_node
                //     << " y_node-1=" << y_node-1
                //     << " q[i_node]=" << q[i_node]
                //     << " > q_max[y_node-1]=" << q_max[y_node-1]
                //     << "\n Updating i_max[y_node-1] to " << i_node
                //     << std::endl;
                q_max[y_node-1] = q[i_node];
                i_max[y_node-1] = i_node;
                }
            }

        // Loop over nodes in event, use i_max to determine per node to which
        // cond point it belongs
        for (int32_t i_node=row_splits[i_event]; i_node<row_splits[i_event+1]; i_node++){
            int32_t y_node = y[i_node];
            if (y_node == 0){
                // Bkg nodes don't belong to a cond point
                which_cond_point[i_node] = -1;
                }
            else {            
                which_cond_point[i_node] = i_max[y_node-1];
                }
            }
        // Copy the i_max and count info to the global cond_indices/counts array
        for(size_t i=0; i<n_cond_this_event; i++){
            cond_indices[i_cond_indices_filler] = i_max[i];
            cond_counts[i_cond_indices_filler] = count[i];
            i_cond_indices_filler++;
            }
        free(q_max);
        free(i_max);
        free(count);
    }

    // Debug printout

    // std::cout << "n_cond_per_event =";
    // for (size_t i=0; i<n_events; i++) std::cout << " " << n_cond_per_event[i];
    // std::cout << std::endl;

    // std::cout << "cond_indices_row_splits =";
    // for (size_t i=0; i<n_events+1; i++) std::cout << " " << cond_indices_row_splits[i];
    // std::cout << std::endl;

    // std::cout << "cond_indices =";
    // for (size_t i=0; i<n_cond; i++) std::cout << " " << cond_indices[i];
    // std::cout << std::endl;

    // std::cout << "cond_counts =";
    // for (size_t i=0; i<n_cond; i++) std::cout << " " << cond_counts[i];
    // std::cout << std::endl;

    // std::cout << "which_cond_point =";
    // for (size_t i=0; i<n_nodes; i++) std::cout << " " << which_cond_point[i];
    // std::cout << std::endl;


    // Prepare output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    auto losses_tensor = torch::zeros({ 5 }, options);
    auto losses = losses_tensor.data_ptr<float_t>();


    float* V_att = (float *)malloc(n_nodes * sizeof(float));
    float* V_rep = (float *)malloc(n_nodes * sizeof(float));
    float* V_srp = (float *)malloc(n_nodes * sizeof(float));

    // Loop over events
    for (size_t i_event=0; i_event<n_events; i_event++) {
        size_t cond_start = cond_indices_row_splits[i_event];
        size_t cond_end = cond_indices_row_splits[i_event+1];

        size_t node_start = row_splits[i_event];
        size_t node_end = row_splits[i_event+1];

        // Loop over nodes
        for (size_t i_node=node_start; i_node<node_end; i_node++) {
            oc_kernel(
                // Global event info
                beta,
                q,
                x,
                n_dim_cluster_space,
                cond_indices,
                cond_counts,
                cond_start,
                cond_end,
                which_cond_point,
                node_end-node_start,
                // This node (to be parallellized in a CUDA kernel)
                i_node,
                // Output
                &(V_att[i_node]),
                &(V_rep[i_node]),
                &(V_srp[i_node])
                );
            }
        }

    // L_beta_cond_logterm and L_beta_noise
    // L_beta_cond_logterm = (-0.2 * torch.log(beta_cond + 1e-9)).mean()
    // L_beta_noise = sB * beta[is_noise].mean(); sB multiplication done in Python
    float L_beta_cond_logterm = 0.;
    float L_beta_noise = 0.;
    for (size_t i_event=0; i_event<n_events; i_event++) {
        // L_beta_cond_logterm
        size_t cond_start = cond_indices_row_splits[i_event];
        size_t cond_end = cond_indices_row_splits[i_event+1];
        float n_cond_this_event = cond_end - cond_start;
        for (size_t i_cond=cond_start; i_cond<cond_end; i_cond++) {
            float beta_cond = beta[cond_indices[i_cond]];
            L_beta_cond_logterm += -0.2 * log(beta_cond + 0.000000001) / (float)n_cond_this_event;
            }
        // L_beta_noise
        float L_beta_noise_this_event = 0.;
        int n_noise_this_event = 0;
        for (int32_t i_node=row_splits[i_event]; i_node<row_splits[i_event+1]; i_node++) {
            if (y[i_node] == 0){
                L_beta_noise_this_event += beta[i_node];
                n_noise_this_event++;
                }
            }
        if (n_noise_this_event>0)
            L_beta_noise += L_beta_noise_this_event / (float)n_noise_this_event ;
        }
    losses[3] = L_beta_cond_logterm / (float)n_events;
    losses[4] = L_beta_noise / (float)n_events;

    free(n_cond_per_event);
    free(cond_indices_row_splits);
    free(cond_indices);
    free(cond_counts);
    free(which_cond_point);

    float V_att_sum = 0.;
    float V_rep_sum = 0.;
    float V_srp_sum = 0.;
    for (size_t i_node=0; i_node<n_nodes; i_node++){
        V_att_sum += V_att[i_node];
        V_rep_sum += V_rep[i_node];
        V_srp_sum += V_srp[i_node];
        }

    // for (size_t i=0; i<n_nodes; i++) std::cout << "V_att[" << i << "]=" << V_att[i] << std::endl;
    // for (size_t i=0; i<n_nodes; i++) std::cout << "V_rep[" << i << "]=" << V_rep[i] << std::endl;
    // for (size_t i=0; i<n_nodes; i++) std::cout << "V_srp[" << i << "]=" << V_srp[i] << std::endl;

    losses[0] = V_att_sum / (float)n_events;
    losses[1] = V_rep_sum / (float)n_events;
    losses[2] = V_srp_sum / (float)n_events;

    free(V_att);
    free(V_rep);
    free(V_srp);

    return losses_tensor;
}

TORCH_LIBRARY(oc_cpu, m) {
  m.def("oc_cpu", oc_cpu);
}