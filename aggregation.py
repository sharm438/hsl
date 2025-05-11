import torch
import numpy as np
import pdb

def federated_aggregation(local_models, weights):
    stack = torch.stack(local_models)
    avg = torch.sum(stack * weights.view(-1,1), dim=0)
    return avg

def p2p_aggregation(spoke_wts, W):
    """
    W: shape [num_spokes, num_spokes]
    spoke_wts: shape [num_spokes, dim]
    returns: shape [num_spokes, dim]
    """
    return torch.mm(W, spoke_wts)

def p2p_local_aggregation(node_wts, outdegree, return_W=False):
    """
    node_wts: shape [num_nodes, dim]
    outdegree: int
    return_W: bool -> if True, also return the effective mixing matrix
    """
    num_nodes = node_wts.shape[0]
    dim = node_wts.shape[1]
    updated_wts = torch.zeros_like(node_wts)

    # If returning the mixing matrix, create NxN zero matrix
    W_local = None
    if return_W:
        W_local = torch.zeros((num_nodes, num_nodes), device=node_wts.device)

    for i in range(num_nodes):
        # Create a pool of neighbors excluding the node itself
        all_neighbors = torch.arange(num_nodes, device=node_wts.device)
        all_neighbors = all_neighbors[all_neighbors != i]  # Exclude self

        # Pick exactly 'outdegree' random neighbors from the pool
        if outdegree <= num_nodes - 1:
            chosen_neighbors = all_neighbors[torch.randperm(all_neighbors.size(0))[:outdegree]]
        else:
            chosen_neighbors = all_neighbors[torch.randint(0, all_neighbors.size(0), (outdegree,))]

        # Include the node itself
        chosen = torch.cat((chosen_neighbors, torch.tensor([i], device=node_wts.device)))
        neighbors_models = node_wts[chosen]
        
        # Compute the mean vector and update weights
        mean_vec = neighbors_models.mean(dim=0)
        updated_wts[i] = mean_vec

        if return_W:
            # Each row i => 1/|chosen| for columns in chosen
            for c in chosen:
                W_local[i, c] = 1.0

    if return_W:
        # Convert each row i into row-stochastic
        for i in range(num_nodes):
            row_sum = torch.sum(W_local[i])
            if row_sum > 0:
                W_local[i] /= row_sum
        return updated_wts, W_local
    else:
        return updated_wts

