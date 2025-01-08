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

# mp.set_start_method("spawn", force=True) # Only needed if you use multiprocessing

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
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--spoke_budget", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--sample_type", type=str, default="round_robin", choices=["round_robin","random"])

    # CHANGED #1: Removed --bidirectional and --self_weights entirely

    # NEW #1: A single seed argument for data + model
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for data splitting + model init. 0 or negative => random")

    # HSL parameters
    parser.add_argument("--b_hs", type=int, default=2)
    parser.add_argument("--b_hh", type=int, default=1)
    parser.add_argument("--b_sh", type=int, default=2)

    return parser.parse_args()

def main(args):
    ############################################################################
    # NEW #2: Initialize seed if provided
    ############################################################################
    if args.seed > 0:
        # Fix Python's built-in random, NumPy, and PyTorch seeds
        import random
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # If you want deterministic cuDNN:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Info] Using fixed seed={args.seed}")
    else:
        print("[Info] Using a random seed (non-reproducible).")

    aggregator_device = torch.device('cuda:'+str(args.gpu)
                                     if args.gpu>=0 and torch.cuda.is_available()
                                     else 'cpu')
    outputs_path = os.path.join(os.getcwd(), "outputs")
    if not os.path.isdir(outputs_path):
        os.makedirs(outputs_path)
    filename = os.path.join(outputs_path, args.exp)

    # Load data
    trainObject = utils.load_data(args.dataset, args.batch_size, args.lr, args.fraction)
    data, labels = trainObject.train_data, trainObject.train_labels
    test_data = trainObject.test_data
    inp_dim = trainObject.num_inputs
    out_dim = trainObject.num_outputs
    lr = args.lr
    batch_size = trainObject.batch_size
    net_name = trainObject.net_name

    # CHANGED #2: Distribute data among spokes with guaranteed >= batch_size
    distributedData = utils.distribute_data_noniid(
        (data, labels),
        args.num_spokes,
        args.bias,
        inp_dim,
        out_dim,
        net_name,
        torch.device("cpu"),  # keep data on CPU
        min_samples=batch_size  # pass the min_samples requirement
    )
    distributed_data = distributedData.distributed_input
    distributed_label = distributedData.distributed_output
    node_weights = distributedData.wts

    # CHANGED #3: Print min, max, mean, variance of the dataset sizes
    num_samples_per_spoke = [len(distributed_data[i]) for i in range(args.num_spokes)]
    min_size = min(num_samples_per_spoke)
    max_size = max(num_samples_per_spoke)
    mean_size = sum(num_samples_per_spoke)/len(num_samples_per_spoke)
    var_size = sum((s - mean_size)**2 for s in num_samples_per_spoke)/len(num_samples_per_spoke)
    print(f"[Data Dist] Min={min_size}, Max={max_size}, Mean={mean_size:.1f}, Var={var_size:.1f}")

    ############################################################################
    # NEW #3: Use the same seed for model initialization
    # If we want the model weights also to be reproducible,
    # we have already set torch.manual_seed(args.seed) if seed>0.
    ############################################################################
    global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)

    # Round-robin indices
    rr_indices = {}
    for node_id in range(args.num_spokes):
        ds_size = distributed_data[node_id].shape[0]
        if ds_size > 0:
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

    ############################################################################
    # Example of sequential training (no concurrency) for simplicity here.
    # If your code previously used multiprocessing, remove or adapt that as needed.
    ############################################################################
    for rnd in range(args.num_rounds):
        global_wts = utils.model_to_vec(global_model).cpu()
        node_return = {}

        # Train each spoke sequentially
        for node_id in range(args.num_spokes):
            updated_wts = train_node.local_train_worker_inline(
                node_id,
                global_wts,
                distributed_data[node_id],
                distributed_label[node_id],
                inp_dim, out_dim, net_name,
                args.num_local_iters,
                batch_size, lr,
                aggregator_device,
                args.sample_type,
                rr_indices
            )
            node_return[node_id] = updated_wts

        # Aggregation
        if args.aggregation == 'fedsgd':
            local_models = [node_return[i] for i in range(args.num_spokes)]
            aggregated_wts = aggregation.federated_aggregation(local_models, node_weights)
            global_model = utils.vec_to_model(aggregated_wts.to(aggregator_device),
                                              net_name, inp_dim, out_dim, aggregator_device)

            if (rnd+1) % args.eval_time == 0:
                traced_model = torch.jit.trace(
                    deepcopy(global_model),
                    torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device)
                )
                loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                metrics['round'].append(rnd+1)
                metrics['global_loss'].append(loss)
                metrics['global_acc'].append(acc)
                print(f"[Round {rnd+1}] FedSGD => Global Acc: {acc:.4f}, Loss: {loss:.4f}")

        elif args.aggregation == 'p2p':
            W = utils.create_k_random_regular_graph(args.num_spokes, args.k, aggregator_device)
            spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)]).to(aggregator_device)
            spoke_wts = aggregation.p2p_aggregation(spoke_wts, W)
            global_model = utils.vec_to_model(spoke_wts[0], net_name, inp_dim, out_dim, aggregator_device)

            if (rnd+1) % args.eval_time == 0:
                local_accs, local_losses = [], []
                for node_id in range(args.num_spokes):
                    node_model = utils.vec_to_model(spoke_wts[node_id], net_name, inp_dim, out_dim, aggregator_device)
                    traced_model = torch.jit.trace(
                        deepcopy(node_model),
                        torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device)
                    )
                    loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                    local_accs.append(acc)
                    local_losses.append(loss)
                metrics['round'].append(rnd+1)
                metrics['local_acc'].append(local_accs)
                metrics['local_loss'].append(local_losses)
                print(f"[Round {rnd+1}] P2P => Local Acc range: [{min(local_accs):.4f}, {max(local_accs):.4f}]")

        elif args.aggregation == 'p2p_local':
            spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)]).to(aggregator_device)
            updated_spoke_wts = aggregation.p2p_local_aggregation(spoke_wts, args.k)
            global_model = utils.vec_to_model(updated_spoke_wts[0], net_name, inp_dim, out_dim, aggregator_device)

            if (rnd+1) % args.eval_time == 0:
                local_accs, local_losses = [], []
                for node_id in range(args.num_spokes):
                    node_model = utils.vec_to_model(updated_spoke_wts[node_id], net_name, inp_dim, out_dim, aggregator_device)
                    traced_model = torch.jit.trace(
                        deepcopy(node_model),
                        torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device)
                    )
                    loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                    local_accs.append(acc)
                    local_losses.append(loss)
                metrics['round'].append(rnd+1)
                metrics['local_acc'].append(local_accs)
                metrics['local_loss'].append(local_losses)
                print(f"[Round {rnd+1}] P2P Local => Local Acc range: [{min(local_accs):.4f}, {max(local_accs):.4f}]")

        elif args.aggregation == 'hsl':
            spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)]).to(aggregator_device)
            # Step 1: Hubs gather from b_hs spokes
            hub_wts = []
            for hub_id in range(args.num_hubs):
                if args.b_hs <= args.num_spokes:
                    chosen_spoke_ids = torch.randperm(args.num_spokes, device=aggregator_device)[:args.b_hs]
                else:
                    chosen_spoke_ids = torch.randint(low=0, high=args.num_spokes, size=(args.b_hs,),
                                                     device=aggregator_device)
                hub_agg = spoke_wts[chosen_spoke_ids].mean(dim=0)
                hub_wts.append(hub_agg)
            hub_wts = torch.stack(hub_wts)

            # Step 2: "EL Local" among hubs
            hub_wts = aggregation.p2p_local_aggregation(hub_wts, args.b_hh)

            # Step 3: Spokes gather from b_sh hubs
            final_spoke_wts = []
            for spoke_id in range(args.num_spokes):
                if args.b_sh <= args.num_hubs:
                    chosen_hub_ids = torch.randperm(args.num_hubs, device=aggregator_device)[:args.b_sh]
                else:
                    chosen_hub_ids = torch.randint(low=0, high=args.num_hubs, size=(args.b_sh,),
                                                   device=aggregator_device)
                spoke_agg = hub_wts[chosen_hub_ids].mean(dim=0)
                final_spoke_wts.append(spoke_agg)
            final_spoke_wts = torch.stack(final_spoke_wts)
            global_model = utils.vec_to_model(final_spoke_wts.mean(dim=0),
                                              net_name, inp_dim, out_dim, aggregator_device)

            if (rnd+1) % args.eval_time == 0:
                spoke_accs=[]
                for s_id in range(args.num_spokes):
                    s_model = utils.vec_to_model(final_spoke_wts[s_id], net_name, inp_dim, out_dim, aggregator_device)
                    traced_model = torch.jit.trace(
                        deepcopy(s_model),
                        torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device)
                    )
                    loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                    spoke_accs.append(acc)
                metrics['round'].append(rnd+1)
                metrics['spoke_acc'].append(spoke_accs)
                print(f"[Round {rnd+1}] HSL => Spoke Acc range: [{min(spoke_accs):.4f}, {max(spoke_accs):.4f}]")

    # save metrics
    with open(filename + '_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Training complete.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
