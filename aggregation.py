import torch
import numpy as np

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
        # pick outdegree random neighbors
        if outdegree <= num_nodes - 1:
            chosen = torch.randperm(num_nodes, device=node_wts.device)[:outdegree]
        else:
            chosen = torch.randint(low=0, high=num_nodes, size=(outdegree,), device=node_wts.device)

        # also include self
        chosen = torch.cat((chosen, torch.tensor([i], device=node_wts.device)))
        chosen = torch.unique(chosen)
        neighbors_models = node_wts[chosen]
        mean_vec = neighbors_models.mean(dim=0)
        updated_wts[i] = mean_vec

        if return_W:
            # Each row i => 1/|chosen| for columns in chosen
            for c in chosen:
                W_local[i, c] = 1.0
    if return_W:
        # convert each row i into row-stochastic
        for i in range(num_nodes):
            row_sum = torch.sum(W_local[i])
            if row_sum > 0:
                W_local[i] /= row_sum
        return updated_wts, W_local
    else:
        return updated_wts