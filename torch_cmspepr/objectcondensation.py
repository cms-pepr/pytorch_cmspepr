from typing import Tuple, Union
import numpy as np
import torch
from torch_scatter import scatter_max, scatter_add


def assert_no_nans(x):
    """
    Raises AssertionError if there is a nan in the tensor
    """
    assert not torch.isnan(x).any()


# FIXME: Use a logger instead of this
DEBUG = False
def debug(*args, **kwargs):
    if DEBUG: print(*args, **kwargs)


def calc_LV_Lbeta(
    beta: torch.Tensor, cluster_space_coords: torch.Tensor, # Predicted by model
    cluster_index_per_event: torch.Tensor, # Truth hit->cluster index
    batch: torch.Tensor,
    # From here on just parameters
    qmin: float = .1,
    s_B: float = .1,
    noise_cluster_index: int = 0, # cluster_index entries with this value are noise/noise
    beta_stabilizing = 'soft_q_scaling',
    huberize_norm_for_V_belonging = True,
    return_components = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], dict]:
    """
    Calculates the L_V and L_beta object condensation losses.

    Concepts:
    - A hit belongs to exactly one cluster (cluster_index_per_event is (n_hits,)),
      and to exactly one event (batch is (n_hits,))
    - A cluster index of `noise_cluster_index` means the cluster is a noise cluster.
      There is typically one noise cluster per event. Any hit in a noise cluster
      is a 'noise hit'. A hit in an object is called a 'signal hit' for lack of a
      better term.
    - An 'object' is a cluster that is *not* a noise cluster. 

    beta_stabilizing: Choices are ['paper', 'clip', 'soft_q_scaling']:
        paper: beta is sigmoid(model_output), q = beta.arctanh()**2 + qmin
        clip:  beta is clipped to 1-1e-4, q = beta.arctanh()**2 + qmin
        soft_q_scaling: beta is sigmoid(model_output), q = (clip(beta)/1.002).arctanh()**2 + qmin

    huberize_norm_for_V_belonging: Huberizes the norms when used in the attractive potential

    Note this function has modifications w.r.t. the implementation in 2002.03605:
    - The norms for V_repulsive are now Gaussian (instead of linear hinge)
    - The 'signal' contribution to L_beta is overhauled with a potential-like term
    """
    device = beta.device

    # ________________________________
    # Calculate a bunch of needed counts and indices locally

    # cluster_index: unique index over events
    # E.g. cluster_index_per_event=[ 0, 0, 1, 2, 0, 0, 1], batch=[0, 0, 0, 0, 1, 1, 1]
    #      -> cluster_index=[ 0, 0, 1, 2, 3, 3, 4 ]
    cluster_index, n_clusters_per_event = batch_cluster_indices(cluster_index_per_event, batch)
    n_clusters = n_clusters_per_event.sum()
    n_hits, cluster_space_dim = cluster_space_coords.size()
    batch_size = batch.max()+1
    n_hits_per_event = scatter_count(batch)

    # Index of cluster -> event (n_clusters,)
    batch_cluster = scatter_counts_to_indices(n_clusters_per_event)

    # Per-hit boolean, indicating whether hit is sig or noise
    is_noise = cluster_index_per_event == noise_cluster_index
    is_sig = ~is_noise
    n_hits_sig = is_sig.sum()
    n_sig_hits_per_event = scatter_count(batch[is_sig])

    # Per-cluster boolean, indicating whether cluster is an object or noise
    is_object = scatter_max(is_sig.long(), cluster_index)[0].bool()
    is_noise_cluster = ~is_object

    # FIXME: This assumes bkg_cluster_index == 0!!
    # Not sure how to do this in a performant way in case bkg_cluster_index != 0
    object_index_per_event = cluster_index_per_event[is_sig] - 1
    object_index, n_objects_per_event = batch_cluster_indices(object_index_per_event, batch[is_sig])
    n_hits_per_object = scatter_count(object_index)
    batch_object = batch_cluster[is_object]
    n_objects = is_object.sum()

    assert object_index.size() == (n_hits_sig,)
    assert is_object.size() == (n_clusters,)
    assert torch.all(n_hits_per_object > 0)
    assert object_index.max()+1 == n_objects

    # ________________________________
    # L_V term

    # Calculate q
    if beta_stabilizing == 'paper':
        q = beta.arctanh()**2 + qmin
    elif beta_stabilizing == 'clip':
        beta = beta.clip(0., 1-1e-4)
        q = beta.arctanh()**2 + qmin
    elif beta_stabilizing == 'soft_q_scaling':
        q = (beta.clip(0., 1-1e-4)/1.002).arctanh()**2 + qmin
    else:
        raise ValueError(f'beta_stablizing mode {beta_stabilizing} is not known')
    assert_no_nans(q)
    assert q.device == device
    assert q.size() == (n_hits,)

    # Calculate q_alpha, the max q per object, and the indices of said maxima
    q_alpha, index_alpha = scatter_max(q[is_sig], object_index)
    assert q_alpha.size() == (n_objects,)

    # Get the cluster space coordinates and betas for these maxima hits too
    x_alpha = cluster_space_coords[is_sig][index_alpha]
    beta_alpha = beta[is_sig][index_alpha]
    assert x_alpha.size() == (n_objects, cluster_space_dim)
    assert beta_alpha.size() == (n_objects,)

    # Connectivity matrix from hit (row) -> cluster (column)
    # Index to matrix, e.g.:
    # [1, 3, 1, 0] --> [
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 1],
    #     [0, 1, 0, 0],
    #     [1, 0, 0, 0]
    #     ]
    M = torch.nn.functional.one_hot(cluster_index).long()

    # Anti-connectivity matrix; be sure not to connect hits to clusters in different events!
    M_inv = get_inter_event_norms_mask(batch, n_clusters_per_event) - M

    # Throw away noise cluster columns; we never need them
    M = M[:,is_object]
    M_inv = M_inv[:,is_object]
    assert M.size() == (n_hits, n_objects)
    assert M_inv.size() == (n_hits, n_objects)

    # Calculate all norms
    # Warning: Should not be used without a mask!
    # Contains norms between hits and objects from different events
    # (n_hits, 1, cluster_space_dim) - (1, n_objects, cluster_space_dim)
    #   gives (n_hits, n_objects, cluster_space_dim)
    norms = (cluster_space_coords.unsqueeze(1) - x_alpha.unsqueeze(0)).norm(dim=-1)
    assert norms.size() == (n_hits, n_objects)

    # -------
    # Attractive potential term

    # First get all the relevant norms: We only want norms of signal hits
    # w.r.t. the object they belong to, i.e. no noise hits and no noise clusters.
    # First select all norms of all signal hits w.r.t. all objects, mask out later
    norms_att = norms[is_sig]

    # Power-scale the norms
    if huberize_norm_for_V_belonging:
        # Huberized version (linear but times 4)
        # Be sure to not move 'off-diagonal' away from zero
        # (i.e. norms of hits w.r.t. clusters they do _not_ belong to)
        norms_att = huber(norms_att+1e-5, 4.)
    else:
        # Paper version is simply norms squared (no need for mask)
        norms_att = norms_att**2
    assert norms_att.size() == (n_hits_sig, n_objects)

    # Now apply the mask to keep only norms of signal hits w.r.t. to the object
    # they belong to
    norms_att *= M[is_sig]

    # Final potential term
    # (n_sig_hits, 1) * (1, n_objects) * (n_sig_hits, n_objects)
    V_attractive = q[is_sig].unsqueeze(-1) * q_alpha.unsqueeze(0) * norms_att
    assert V_attractive.size() == (n_hits_sig, n_objects)

    # Sum over hits, then sum per event, then divide by n_hits_per_event, then sum over events
    V_attractive = scatter_add(V_attractive.sum(dim=0), batch_object) / n_hits_per_event
    assert V_attractive.size() == (batch_size,)
    L_V_attractive = V_attractive.sum()

    # -------
    # Repulsive potential term

    # Get all the relevant norms: We want norms of any hit w.r.t. to 
    # objects they do *not* belong to, i.e. no noise clusters.
    # We do however want to keep norms of noise hits w.r.t. objects
    # Power-scale the norms: Gaussian scaling term instead of a cone
    # Mask out the norms of hits w.r.t. the cluster they belong to
    norms_rep = torch.exp(-4.*norms**2) * M_inv
    
    # (n_sig_hits, 1) * (1, n_objects) * (n_sig_hits, n_objects)
    V_repulsive = q.unsqueeze(1) * q_alpha.unsqueeze(0) * norms_rep
    # No need to apply a V = max(0, V); by construction V>=0
    assert V_repulsive.size() == (n_hits, n_objects)

    # Sum over hits, then sum per event, then divide by n_hits_per_event, then sum up events
    L_V_repulsive = (scatter_add(V_repulsive.sum(dim=0), batch_object)/n_hits_per_event).sum()
    L_V = L_V_attractive + L_V_repulsive

    # ________________________________
    # L_beta term

    # -------
    # L_beta noise term
    
    n_noise_hits_per_event = scatter_count(batch[is_noise])
    L_beta_noise = s_B * ((scatter_add(beta[is_noise], batch[is_noise])) / n_noise_hits_per_event).sum()

    # -------
    # L_beta signal term
    
    # First collect the norms: We only want norms of hits w.r.t. the object they
    # belong to (like in V_attractive)
    # Apply transformation first, and then apply mask to keep only the norms we want,
    # then sum over hits, so the result is (n_objects,)
    norms_beta_sig = (1./(20.*norms[is_sig]**2+1.) * M[is_sig]).sum(dim=0)
    assert torch.all(norms_beta_sig >= 1.) and torch.all(norms_beta_sig <= n_hits_per_object)
    # Subtract from 1. to remove self interaction, divide by number of hits per object
    norms_beta_sig = (1. - norms_beta_sig) / n_hits_per_object
    assert torch.all(norms_beta_sig >= -1.) and torch.all(norms_beta_sig <= 0.)
    norms_beta_sig *= beta_alpha
    # Conclusion:
    # lower beta --> higher loss (less negative)
    # higher norms --> higher loss

    # Sum over objects, divide by number of objects per event, then sum over events
    L_beta_norms_term = (scatter_add(norms_beta_sig, batch_object) / n_objects_per_event).sum()
    assert L_beta_norms_term >= -batch_size and L_beta_norms_term <= 0.

    # Logbeta term: Take -.2*torch.log(beta_alpha[is_object]+1e-9), sum it over objects,
    # divide by n_objects_per_event, then sum over events (same pattern as above)
    # lower beta --> higher loss
    L_beta_logbeta_term = (
        scatter_add(-.2*torch.log(beta_alpha+1e-9), batch_object) / n_objects_per_event
        ).sum()

    # Final L_beta term
    L_beta_sig = L_beta_norms_term + L_beta_logbeta_term
    L_beta = L_beta_noise + L_beta_sig

    # ________________________________
    # Returning
    # Also divide by batch size here

    if return_components or DEBUG:
        components = dict(
            L_V = L_V / batch_size,
            L_V_attractive = L_V_attractive / batch_size,
            L_V_repulsive = L_V_repulsive / batch_size,
            L_beta = L_beta / batch_size,
            L_beta_noise = L_beta_noise / batch_size,
            L_beta_sig = L_beta_sig / batch_size,
            L_beta_norms_term = L_beta_norms_term / batch_size,
            L_beta_logbeta_term = L_beta_logbeta_term / batch_size,
            )
    if DEBUG:
        debug(formatted_loss_components_string(components))
    return components if return_components else (L_V/batch_size, L_beta/batch_size)


def formatted_loss_components_string(components: dict) -> str:
    """
    Formats the components returned by calc_LV_Lbeta
    """
    total_loss = components['L_V']+components['L_beta']
    fractions = { k : v/total_loss for k, v in components.items() }
    fkey = lambda key: f'{components[key]:+.4f} ({100.*fractions[key]:.1f}%)'
    return (
        'L_V+L_beta = {L:.4f}'
        '\n  L_V                 = {L_V}'
        '\n    L_V_attractive      = {L_V_attractive}'
        '\n    L_V_repulsive       = {L_V_repulsive}'
        '\n  L_beta              = {L_beta}'
        '\n    L_beta_noise        = {L_beta_noise}'
        '\n    L_beta_sig          = {L_beta_sig}'
        '\n      L_beta_norms_term   = {L_beta_norms_term}'
        '\n      L_beta_logbeta_term = {L_beta_logbeta_term}'
        .format(L=total_loss,**{k : fkey(k) for k in components})
        )


def huber(d, delta):
    """
    See: https://en.wikipedia.org/wiki/Huber_loss#Definition
    Multiplied by 2 w.r.t Wikipedia version (aligning with Jan's definition)
    """
    return torch.where(torch.abs(d)<=delta, d**2, 2.*delta*(torch.abs(d)-delta))


def batch_cluster_indices(cluster_id: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Turns cluster indices per event to an index in the whole batch

    Example:

    cluster_id = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    -->
    offset = torch.LongTensor([0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 5, 5, 5])
    output = torch.LongTensor([0, 0, 1, 1, 2, 3, 3, 4, 4, 4, 5, 5, 6])
    """
    device = cluster_id.device
    assert cluster_id.device == batch.device
    # Count the number of clusters per entry in the batch
    n_clusters_per_event = scatter_max(cluster_id, batch, dim=-1)[0] + 1
    # Offsets are then a cumulative sum
    offset_values_nozero = n_clusters_per_event[:-1].cumsum(dim=-1)
    # Prefix a zero
    offset_values = torch.cat((torch.zeros(1, device=device), offset_values_nozero))
    # Fill it per hit
    offset = torch.gather(offset_values, 0, batch).long()
    return offset + cluster_id, n_clusters_per_event


def get_clustering_np(betas: np.array, X: np.array, tbeta: float=.1, td: float=1.) -> np.array:
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes numpy arrays as input.
    """
    n_points = betas.shape[0]
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = np.nonzero(select_condpoints)[0]
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[np.argsort(-betas[select_condpoints])]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = np.arange(n_points)
    clustering = -1 * np.ones(n_points, dtype=np.int32)
    for index_condpoint in indices_condpoints:
        d = np.linalg.norm(X[unassigned] - X[index_condpoint], axis=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint
        unassigned = unassigned[~(d < td)]
    return clustering


def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=.1, td=1.):
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes torch.Tensors as input.
    """
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = torch.arange(n_points)
    clustering = -1 * torch.ones(n_points, dtype=torch.long)
    for index_condpoint in indices_condpoints:
        d = torch.norm(X[unassigned] - X[index_condpoint], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint
        unassigned = unassigned[~(d < td)]
    return clustering


def scatter_count(input: torch.Tensor):
    """
    Returns ordered counts over an index array

    Example:
    input:  [0, 0, 0, 1, 1, 2, 2]
    output: [3, 2, 2]

    Index assumptions like in torch_scatter, so:
    scatter_count(torch.Tensor([1, 1, 1, 2, 2, 4, 4]))
    >>> tensor([0, 3, 2, 0, 2])
    """
    return scatter_add(torch.ones_like(input, dtype=torch.long), input.long())


def scatter_counts_to_indices(input: torch.LongTensor) -> torch.LongTensor:
    """
    Converts counts to indices. This is the inverse operation of scatter_count
    Example:
    input:  [3, 2, 2]
    output: [0, 0, 0, 1, 1, 2, 2]
    """
    return torch.repeat_interleave(torch.arange(input.size(0), device=input.device), input).long()


def get_inter_event_norms_mask(batch: torch.LongTensor, nclusters_per_event: torch.LongTensor):
    """
    Creates mask of (nhits x nclusters) that is only 1 if hit i is in the same event as cluster j

    Example:
    cluster_id_per_event = torch.LongTensor([0, 0, 1, 1, 2, 0, 0, 1, 1, 1, 0, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2])

    Should return:
    torch.LongTensor([
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        ])
    """
    device = batch.device
    # Following the example:
    # Expand batch to the following (nhits x nevents) matrix (little hacky, boolean mask -> long):
    # [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
    batch_expanded_as_ones = (batch == torch.arange(batch.max()+1, dtype=torch.long, device=device).unsqueeze(-1) ).long()
    # Then repeat_interleave it to expand it to nclusters rows, and transpose to get (nhits x nclusters)
    return batch_expanded_as_ones.repeat_interleave(nclusters_per_event, dim=0).T
