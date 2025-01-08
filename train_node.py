# train_node.py
import torch
import torch.optim as optim
import torch.nn as nn
import models
import utils
import numpy as np

def local_train_worker_inline(node_id, global_wts, local_data, local_label, 
                              inp_dim, out_dim, net_name,
                              num_local_iters, batch_size, lr, device,
                              sample_type, rr_indices):
    """
    SEQUENTIAL local training for 'num_local_iters' mini-batches.
    If sample_type == "random", pick random samples each iteration.
    If sample_type == "round_robin", we do wrap-around via modulo.
    'rr_indices[node_id]' is a persistent pointer so that each call continues
    from where the previous round left off.
    """
    train_device = device
    
    # Reconstruct the model from global_wts
    model = utils.vec_to_model(global_wts.to(train_device),
                               net_name, inp_dim, out_dim, train_device)
    criterion = nn.CrossEntropyLoss().to(train_device)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    data_size = local_data.shape[0]
    if data_size == 0:
        # Edge case: if some node truly has no data, return unchanged
        return utils.model_to_vec(model).cpu()

    # Persistent pointer for round-robin
    rr_idx = rr_indices[node_id]

    for _ in range(num_local_iters):
        if sample_type == "random":
            # If batch_size > data_size, we do replace=True
            if batch_size <= data_size:
                idxs = np.random.choice(data_size, batch_size, replace=False)
            else:
                idxs = np.random.choice(data_size, batch_size, replace=True)
        elif sample_type == "round_robin":
            # --- ONLY DO WRAP-AROUND VIA MODULO ---
            idxs = [(rr_idx + i) % data_size for i in range(batch_size)]
            rr_idx = (rr_idx + batch_size) % data_size

        # Now index safely
        batch_x = local_data[idxs].to(train_device)
        batch_y = local_label[idxs].to(train_device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Save updated rr_idx for next call
    rr_indices[node_id] = rr_idx

    # Return updated weights as a CPU tensor
    updated_wts = utils.model_to_vec(model).cpu()
    return updated_wts
