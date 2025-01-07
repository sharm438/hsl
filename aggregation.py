import torch
import numpy as np
import pdb

def federated_aggregation(local_models, weights):
    stack = torch.stack(local_models)
    avg = torch.sum(stack * weights.view(-1,1), dim=0)
    return avg

def p2p_aggregation(spoke_wts, W):
    return torch.mm(W, spoke_wts)

def p2p_local_aggregation(node_wts, outdegree):
    num_nodes = node_wts.shape[0]
    dim = node_wts.shape[1]
    updated_wts = torch.zeros_like(node_wts)
    
    for i in range(num_nodes):
        # pick outdegree random neighbors
        if outdegree <= num_nodes-1:
            chosen = torch.randperm(num_nodes, device=node_wts.device)[:outdegree]
        else:
            chosen = torch.randint(low=0, high=num_nodes, size=(outdegree,), device=node_wts.device)
        chosen = torch.cat((chosen, torch.tensor([i], device=node_wts.device)))
        chosen = torch.unique(chosen)
        neighbors_models = node_wts[chosen]
        updated_wts[i] = neighbors_models.mean(dim=0)
    return updated_wts
