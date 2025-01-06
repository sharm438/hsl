import argparse
import torch
import torch.multiprocessing as mp
import os
import time
import json
from copy import deepcopy
import utils
import aggregation
import train_node
import models

####################################
# Explanation of the two P2P methods
# 1) p2p (EL Oracle): 
#    Uses a centrally maintained k-regular graph W. 
#    Each node i has exactly k neighbors out (fixed outdegree=k) 
#    and k neighbors in (fixed indegree=k), so total in/out = k. 
# 2) p2p_local (EL Local): 
#    Each node i chooses k neighbors (fixed outdegree=k), 
#    but does NOT constrain how many other nodes may link in. 
#    Indegree is thus not necessarily fixed. 
#    This can be done in a distributed manner with no central coordinator.
####################################

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default='experiment')
    parser.add_argument("--dataset", type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--bias", type=float, default=0.0)
    parser.add_argument("--aggregation", type=str, default='fedsgd', 
                        choices=['fedsgd', 'p2p', 'p2p_local', 'hsl'])
    parser.add_argument("--num_spokes", type=int, default=10)
    parser.add_argument("--num_hubs", type=int, default=1)
    parser.add_argument("--num_rounds", type=int, default=100)
    parser.add_argument("--num_local_iters", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--eval_time", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)

    # Original p2p outdegree
    parser.add_argument("--k", type=int, default=2, 
                        help="Outdegree for the centrally-coordinated (p2p) topology")

    # For the old code, we had a single 'spoke_budget'; now replaced in HSL by separate b_hs, b_sh, etc.
    # We'll keep it for backward compatibility if needed, but set a default:
    parser.add_argument("--spoke_budget", type=int, default=1)

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--sample_type", type=str, default="round_robin", 
                        choices=["round_robin","random"])
    parser.add_argument("--bidirectional", type=int, default=0, 
                        help="Not used in new HSL logic, kept for backward compatibility")
    parser.add_argument("--self_weights", type=int, default=0, 
                        help="Not used in new HSL logic, kept for backward compatibility")

    # New HSL parameters:
    parser.add_argument("--b_hs", type=int, default=2, 
                        help="Step 1: each hub chooses b_hs spokes (fixed hub indegree = b_hs)")
    parser.add_argument("--b_hh", type=int, default=1, 
                        help="Step 2: each hub gossips with b_hh other hubs in EL Local style")
    parser.add_argument("--b_sh", type=int, default=2, 
                        help="Step 3: each spoke chooses b_sh hubs (fixed spoke indegree = b_sh)")

    return parser.parse_args()

def main(args):
    aggregator_device = torch.device('cuda:'+str(args.gpu) if args.gpu>=0 and torch.cuda.is_available() else 'cpu')
    outputs_path = os.path.join(os.getcwd(), "outputs")
    if not os.path.isdir(outputs_path):
        os.makedirs(outputs_path)
    filename = os.path.join(outputs_path, args.exp)

    # Load data
    trainObject = utils.load_data(args.dataset, args.batch_size, args.fraction)
    data, labels = trainObject.train_data, trainObject.train_labels
    test_data = trainObject.test_data
    inp_dim = trainObject.num_inputs
    out_dim = trainObject.num_outputs
    lr = args.lr
    batch_size = trainObject.batch_size
    net_name = trainObject.net_name

    # Distribute data among spokes
    distributedData = utils.distribute_data_noniid((data, labels), args.num_spokes, 
                                                   args.bias, inp_dim, out_dim, 
                                                   net_name, aggregator_device)
    distributed_data = distributedData.distributed_input
    distributed_label = distributedData.distributed_output
    node_weights = distributedData.wts  # For fedsgd weighting

    # Global model
    global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)

    # Round-robin index management
    manager = mp.Manager()
    rr_indices = manager.dict()
    for node_id in range(args.num_spokes):
        ds_size = distributed_data[node_id].shape[0]
        if ds_size > 0:
            distributed_data[node_id] = distributed_data[node_id].cpu()
            distributed_label[node_id] = distributed_label[node_id].cpu()
            perm = torch.randperm(ds_size)
            distributed_data[node_id] = distributed_data[node_id][perm]
            distributed_label[node_id] = distributed_label[node_id][perm]
        rr_indices[node_id] = 0

    metrics = {
        'round': [],
        'global_acc': [],
        'global_loss': [],
        'local_acc': [],
        'local_loss': [],
        'spoke_acc': []
    }

    return_dict = manager.dict()

    # Main training loop
    for rnd in range(args.num_rounds):
        global_wts = utils.model_to_vec(global_model).cpu()
        node_return = manager.dict()
        processes = []

        # Run local training on CPU (spokes)
        for node_id in range(args.num_spokes):
            p = mp.Process(target=train_node.local_train_worker,
                           args=(node_id, global_wts, 
                                 distributed_data[node_id], distributed_label[node_id],
                                 inp_dim, out_dim, net_name, 
                                 args.num_local_iters, batch_size, lr, 'cpu',
                                 args.sample_type, rr_indices, node_return))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # Each spoke now has updated weights
        if args.aggregation == 'fedsgd':
            # Classic FedSGD aggregator: Weighted average
            local_models = [node_return[i] for i in range(args.num_spokes)]
            aggregated_wts = aggregation.federated_aggregation(local_models, node_weights)
            global_model = utils.vec_to_model(aggregated_wts.to(aggregator_device), 
                                              net_name, inp_dim, out_dim, aggregator_device)

            # Evaluate
            if (rnd+1) % args.eval_time == 0:
                traced_model = torch.jit.trace(deepcopy(global_model),
                                               torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device))
                loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                metrics['round'].append(rnd+1)
                metrics['global_loss'].append(loss)
                metrics['global_acc'].append(acc)
                print(f"[Round {rnd+1}] FedSGD => Global Acc: {acc:.4f}, Loss: {loss:.4f}")

        elif args.aggregation == 'p2p':
            # EL Oracle with a central k-regular adjacency
            W = utils.create_k_random_regular_graph(args.num_spokes, args.k, aggregator_device)
            spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)])
            spoke_wts = spoke_wts.to(aggregator_device)
            spoke_wts = aggregation.p2p_aggregation(spoke_wts, W)

            # For consistency, define global model as spoke 0's model
            global_model = utils.vec_to_model(spoke_wts[0], net_name, inp_dim, out_dim, aggregator_device)

            if (rnd+1)%args.eval_time==0:
                local_accs, local_losses = [], []
                for node_id in range(args.num_spokes):
                    node_model = utils.vec_to_model(spoke_wts[node_id], 
                                                    net_name, inp_dim, out_dim, aggregator_device)
                    traced_model = torch.jit.trace(deepcopy(node_model),
                                                   torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device))
                    loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                    local_accs.append(acc)
                    local_losses.append(loss)
                metrics['round'].append(rnd+1)
                metrics['local_acc'].append(local_accs)
                metrics['local_loss'].append(local_losses)
                print(f"[Round {rnd+1}] P2P (EL Oracle) => Local Acc range: [{min(local_accs):.4f}, {max(local_accs):.4f}]")

        elif args.aggregation == 'p2p_local':
            # EL Local: fixed outdegree = k, indefinite indegree
            spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)]).to(aggregator_device)
            updated_spoke_wts = aggregation.p2p_local_aggregation(spoke_wts, args.k)  
            # Same logic: define global model as spoke 0's model
            global_model = utils.vec_to_model(updated_spoke_wts[0], net_name, inp_dim, out_dim, aggregator_device)

            if (rnd+1)%args.eval_time==0:
                local_accs, local_losses = [], []
                for node_id in range(args.num_spokes):
                    node_model = utils.vec_to_model(updated_spoke_wts[node_id], 
                                                    net_name, inp_dim, out_dim, aggregator_device)
                    traced_model = torch.jit.trace(deepcopy(node_model),
                                                   torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device))
                    loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                    local_accs.append(acc)
                    local_losses.append(loss)
                metrics['round'].append(rnd+1)
                metrics['local_acc'].append(local_accs)
                metrics['local_loss'].append(local_losses)
                print(f"[Round {rnd+1}] P2P Local => Local Acc range: [{min(local_accs):.4f}, {max(local_accs):.4f}]")

        elif args.aggregation == 'hsl':
            ##################################
            # New 3-step HSL:
            # Step 1: Hubs gather from b_hs spokes
            # Step 2: Hubs do "EL Local" among themselves with b_hh
            # Step 3: Spokes gather from b_sh hubs
            ##################################
            spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)]).to(aggregator_device)

            # --- Step 1: Hub Aggregation from spokes (fixed hub indegree = b_hs) ---
            # Each hub chooses b_hs spokes at random. If b_hs > num_spokes, we sample with replacement.
            hub_wts = []
            for hub_id in range(args.num_hubs):
                if args.b_hs <= args.num_spokes:
                    chosen_spoke_ids = torch.randperm(args.num_spokes)[:args.b_hs]
                else:
                    chosen_spoke_ids = torch.randint(low=0, high=args.num_spokes, size=(args.b_hs,))
                # average those chosen spokes
                chosen_models = spoke_wts[chosen_spoke_ids]
                hub_agg = chosen_models.mean(dim=0)
                hub_wts.append(hub_agg)
            hub_wts = torch.stack(hub_wts)  # shape [num_hubs, model_dim]

            # --- Step 2: Hubs do "EL Local" among themselves with outdegree = b_hh ---
            # We'll re-use the p2p_local_aggregation but for 'num_hubs' hubs
            hub_wts = aggregation.p2p_local_aggregation(hub_wts, args.b_hh)

            # --- Step 3: Spokes gather from b_sh hubs (fixed spoke indegree = b_sh) ---
            final_spoke_wts = []
            for spoke_id in range(args.num_spokes):
                if args.b_sh <= args.num_hubs:
                    chosen_hub_ids = torch.randperm(args.num_hubs)[:args.b_sh]
                else:
                    chosen_hub_ids = torch.randint(low=0, high=args.num_hubs, size=(args.b_sh,))
                chosen_hub_models = hub_wts[chosen_hub_ids]
                # average them
                spoke_agg = chosen_hub_models.mean(dim=0)
                final_spoke_wts.append(spoke_agg)
            final_spoke_wts = torch.stack(final_spoke_wts)

            # For logging, define the global model as the average of final spokes
            global_model = utils.vec_to_model(final_spoke_wts.mean(dim=0), net_name, inp_dim, out_dim, aggregator_device)

            # Evaluate
            if (rnd+1)%args.eval_time==0:
                spoke_accs=[]
                for s_id in range(args.num_spokes):
                    s_model = utils.vec_to_model(final_spoke_wts[s_id], 
                                                 net_name, inp_dim, out_dim, aggregator_device)
                    traced_model = torch.jit.trace(deepcopy(s_model),
                                                   torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device))
                    loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                    spoke_accs.append(acc)
                metrics['round'].append(rnd+1)
                metrics['spoke_acc'].append(spoke_accs)
                print(f"[Round {rnd+1}] HSL => Spoke Acc range: [{min(spoke_accs):.4f}, {max(spoke_accs):.4f}]")

    with open(filename + '_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Training complete.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
