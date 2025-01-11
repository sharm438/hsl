import argparse
import torch
import os
import time
import utils
import models
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser()
    # (same arguments as in main or can be omitted if not called directly)
    return parser.parse_args()

def local_train_worker_inline(
    node_id,
    global_wts,
    local_data,
    local_label,
    inp_dim,
    out_dim,
    net_name,
    num_local_iters,
    batch_size,
    lr,
    device,
    sample_type,
    rr_indices
):
    """
    Inline version of local training (no multiprocessing).
    """
    # Rebuild local model from global weights
    local_model = utils.vec_to_model(global_wts, net_name, inp_dim, out_dim, device)
    local_model.train()

    # We'll iterate 'num_local_iters' times over mini-batches
    idx_start = rr_indices[node_id]
    for _ in range(num_local_iters):
        if sample_type == 'round_robin':
            # take the next batch sequentially
            end_idx = idx_start + batch_size
            if end_idx > local_data.shape[0]:
                # wrap around
                end_idx = batch_size
                idx_start = 0
            x = local_data[idx_start:end_idx].to(device)
            y = local_label[idx_start:end_idx].to(device)
            idx_start = end_idx
        else:
            # random
            indices = torch.randint(0, local_data.shape[0], (batch_size,), device=local_data.device)
            x = local_data[indices].to(device)
            y = local_label[indices].to(device)

        # one forward/backward pass
        loss_fn = torch.nn.CrossEntropyLoss()
        out = local_model(x)
        loss = loss_fn(out, y)

        local_model.zero_grad()
        loss.backward()
        for p in local_model.parameters():
            p.data -= lr * p.grad

    rr_indices[node_id] = idx_start
    # return new weights
    return utils.model_to_vec(local_model).cpu()
