import torch
import numpy as np

def federated_aggregation(local_models, weights):
    """
    Classic Federated averaging: weighted sum of local models.
    local_models: list of 1D parameter vectors (torch.Tensor)
    weights: 1D vector of weights (must match length of local_models)
    """
    stack = torch.stack(local_models)
    # Weighted average
    avg = torch.sum(stack * weights.view(-1,1), dim=0)
    return avg

def p2p_aggregation(spoke_wts, W):
    """
    EL Oracle (p2p) aggregation: 
    Multiply the matrix of models by adjacency W (k-regular).
    spoke_wts: shape [num_spokes, model_dim]
    W: shape [num_spokes, num_spokes], row-stochastic adjacency
    Returns: updated model vectors for each node
    """
    return torch.mm(W, spoke_wts)

def p2p_local_aggregation(node_wts, outdegree):
    """
    EL Local style aggregation:
    - node_wts: [num_nodes, model_dim]
    - Each node i chooses 'outdegree' random neighbors (plus itself), 
      and averages all involved models. 
    - Indegree can be arbitrary (some node might be chosen by many, 
      or none) => no guarantee on in-degree.
    Returns updated node weights.
    """
    num_nodes = node_wts.shape[0]
    dim = node_wts.shape[1]
    updated_wts = torch.zeros_like(node_wts)
    
    for i in range(num_nodes):
        # pick outdegree random neighbors (no replacement if possible)
        if outdegree <= num_nodes-1:
            chosen = torch.randperm(num_nodes, device=node_wts.device)[:outdegree]
        else:
            # if outdegree is bigger than available nodes, sample with replacement
            chosen = torch.randint(low=0, high=num_nodes, size=(outdegree,))

        # Combine chosen neighbors + itself
        # (If i is not already in 'chosen', add it)
        chosen = torch.cat((chosen, torch.tensor([i], device=node_wts.device)))
        chosen = torch.unique(chosen)
        # Average
        neighbors_models = node_wts[chosen]
        updated_wts[i] = neighbors_models.mean(dim=0)
    return updated_wts
