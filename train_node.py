import torch
import torch.optim as optim
import torch.nn as nn
import models
import utils
import numpy as np

def local_train_worker(node_id, global_wts, local_data, local_label, inp_dim, out_dim, net_name,
                       num_local_iters, batch_size, lr, device, sample_type, rr_indices, return_dict):
    train_device = torch.device('cpu')
    model = utils.vec_to_model(global_wts.to(train_device), net_name, inp_dim, out_dim, train_device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    data_size = local_data.shape[0]

    if data_size == 0:
        updated_wts = utils.model_to_vec(model).cpu()
        return_dict[node_id] = updated_wts
        return

    rr_idx = rr_indices[node_id]

    for _ in range(num_local_iters):
        if sample_type == "random":
            if batch_size <= data_size:
                idxs = np.random.choice(data_size, batch_size, replace=False)
            else:
                idxs = np.random.choice(data_size, batch_size, replace=True)
            batch_x = local_data[idxs].to(train_device)
            batch_y = local_label[idxs].to(train_device)
        elif sample_type == "round_robin":
            if rr_idx + batch_size <= data_size:
                minibatch = range(rr_idx, rr_idx+batch_size)
                rr_idx += batch_size
            else:
                remain = (rr_idx + batch_size) - data_size
                minibatch = list(range(rr_idx, data_size))
                if remain>0:
                    minibatch += list(range(0,remain))
                rr_idx = remain

            batch_x = local_data[list(minibatch)].to(train_device)
            batch_y = local_label[list(minibatch)].to(train_device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    rr_indices[node_id] = rr_idx
    updated_wts = utils.model_to_vec(model).cpu()
    return_dict[node_id] = updated_wts
