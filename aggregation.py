import torch

def federated_aggregation(local_models, weights):
    stack = torch.stack(local_models)
    avg = torch.sum(stack*weights.view(-1,1),dim=0)
    return avg

def p2p_aggregation(spoke_wts, W):
    return torch.mm(W, spoke_wts)

# HSL handled in main
