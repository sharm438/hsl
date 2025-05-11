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
import math
import pdb
import tqdm
from tqdm import tqdm

from simple_base_graph import SimpleBaseGraph
from base_graph import BaseGraph

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default='experiment',
                        help='Experiment name for output files.')
    parser.add_argument("--dataset", type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'femnist', 'agnews', 'tiny_imagenet', 'cifar100'],
                        help='Dataset to use.')
    parser.add_argument("--fraction", type=float, default=1.0,
                        help='Fraction of the dataset to be used for training.')
    parser.add_argument("--bias", type=float, default=0.0,
                        help='Parameter to control non-IID data distribution.')
    parser.add_argument("--aggregation", type=str, default='fedsgd',
                        choices=['fedsgd', 'p2p', 'p2p_local', 'hsl'],
                        help='Aggregation protocol to use.')
    parser.add_argument("--topo", type=str, default=None,
                        choices=['ring', 'torus', 'erdos-renyi', 'base-graph', 'simple-base-graph', 'exponential'],
                        help='Topology to use for p2p graphs.')
    parser.add_argument("--budget", type=int, default=None,
                        help='Number of edges in an undirected p2p graph '
                             '(actual directed edges = 2*budget).')
    parser.add_argument("--num_spokes", type=int, default=10,
                        help='Number of spokes (worker nodes).')
    parser.add_argument("--num_hubs", type=int, default=1,
                        help='Number of hubs (for HSL).')
    parser.add_argument("--num_rounds", type=int, default=100,
                        help='Number of training or simulation rounds.')
    parser.add_argument("--num_local_iters", type=int, default=1,
                        help='Local training epochs/iterations per round.')
    parser.add_argument("--batch_size", type=int, default=None,
                        help='Mini-batch size for local training.')
    parser.add_argument("--eval_time", type=int, default=1,
                        help='Evaluate the model every "eval_time" rounds.')
    parser.add_argument("--gpu", type=int, default=0,
                        help='GPU ID; use -1 for CPU.')
    parser.add_argument("--k", type=int, default=2,
                        help='Number of neighbors in a k-regular graph.')
    parser.add_argument("--spoke_budget", type=int, default=1,
                        help='(Unused in this script) Spoke connection budget.')
    parser.add_argument("--lr", type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument("--sample_type", type=str, default="round_robin",
                        choices=["round_robin","random"],
                        help='How data is sampled for local training.')
    parser.add_argument("--seed", type=int, default=0,
                        help='Random seed. If 0 or negative, uses a random seed.')

    # HSL parameters
    parser.add_argument("--b_hs", type=int, default=2,
                        help='Max spoke connections per hub in stage 1.')
    parser.add_argument("--b_hh", type=int, default=1,
                        help='Max neighbor connections per hub in stage 2.')
    parser.add_argument("--b_sh", type=int, default=2,
                        help='Max hub connections per spoke in stage 3.')

    # Monitoring flags
    parser.add_argument("--monitor_model_drift", action='store_true',
                        help="Compute and log average model drift among spokes.")
    parser.add_argument("--monitor_degree", action='store_true',
                        help="If set, compute and log node in/out degrees.")
    parser.add_argument("--graph_simulation_only", action='store_true',
                        help="If set, no actual training occurs; only simulate graphs.")

    return parser.parse_args()


def main(args):
    """
    Main function:
    1. Sets random seeds if specified.
    2. Prepares the device, output paths, and data (unless only simulating).
    3. Executes training or graph simulation based on "args.aggregation" and "args.graph_simulation_only".
    4. Logs metrics (accuracy, loss, etc.) for analysis.
    """
    # Set random seeds if desired
    if args.seed > 0:
        import random
        import numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"[Info] Using fixed seed={args.seed}")
    else:
        print("[Info] Using a random seed (non-reproducible).")

    # Set device for training
    aggregator_device = torch.device(
        f'cuda:{args.gpu}' if (args.gpu >= 0 and torch.cuda.is_available()) else 'cpu'
    )

    # Prepare output directory
    outputs_path = os.path.join(os.getcwd(), "outputs")
    if not os.path.isdir(outputs_path):
        os.makedirs(outputs_path)
    filename = os.path.join(outputs_path, args.exp)

    # Load data if we are doing real training
    if not args.graph_simulation_only:
        if args.dataset == 'femnist':
            trainObject, distributed_data, distributed_label = utils.load_data(args.dataset, 32, args.lr, args.fraction)
            test_data = trainObject.test_data
            inp_dim = trainObject.num_inputs
            out_dim = trainObject.num_outputs
            lr = args.lr
            batch_size = trainObject.batch_size
            net_name = trainObject.net_name
            num_clients = len(distributed_label)
            counts = torch.tensor([distributed_label[i].shape[0] for i in range(num_clients)],dtype=torch.float32)
            node_weights = counts / counts.sum()
            node_weights = node_weights.to(aggregator_device)
            
            
        else:
            trainObject = utils.load_data(args.dataset, args.batch_size, args.lr, args.fraction)
            data, labels = trainObject.train_data, trainObject.train_labels
            test_data = trainObject.test_data
            inp_dim = trainObject.num_inputs
            out_dim = trainObject.num_outputs
            lr = args.lr
            batch_size = trainObject.batch_size
            net_name = trainObject.net_name
            
            # Distribute data to spokes in a potentially non-IID way
            distributedData = utils.distribute_data_noniid(
                (data, labels),
                args.num_spokes,
                args.bias,
                inp_dim,
                out_dim,
                net_name,
                torch.device("cpu"),
                min_samples=(batch_size if batch_size else 1)
            )
            #distributedData = utils.distribute_balanced_per_class(data, labels, args.num_spokes)
            distributed_data = distributedData.distributed_input
            distributed_label = distributedData.distributed_output
            
            node_weights = distributedData.wts.to(aggregator_device)

        # Report basic statistics about data distribution
        num_samples_per_spoke = [len(distributed_data[i]) for i in range(args.num_spokes)]
        min_size = min(num_samples_per_spoke)
        max_size = max(num_samples_per_spoke)
        mean_size = sum(num_samples_per_spoke) / len(num_samples_per_spoke)
        var_size = sum((s - mean_size)**2 for s in num_samples_per_spoke) / len(num_samples_per_spoke)
        sd_size = math.sqrt(var_size)
        
        print(f"[Data Dist] Min={min_size}, Max={max_size}, Mean={mean_size:.1f}, SD={sd_size:.1f}")
        # Initialize the global model
        global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)
        
        # For round-robin or random sampling within each spoke
        rr_indices = {}
        for node_id in range(args.num_spokes):
            ds_size = distributed_data[node_id].shape[0]
            if ds_size > 0:
                perm = torch.randperm(ds_size)
                distributed_data[node_id] = distributed_data[node_id][perm]
                distributed_label[node_id] = distributed_label[node_id][perm]
            rr_indices[node_id] = 0
    else:
        # If only simulating graph connectivity, create dummy data placeholders
        test_data = None
        inp_dim, out_dim = 1, 1
        net_name = 'lenet'
        batch_size = 32
        node_weights = torch.ones(args.num_spokes) / args.num_spokes
        global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)

    # Dictionary to log metrics over rounds
    metrics = {
        'round': [],
        'global_acc': [],
        'global_loss': [],
        'local_acc': [],
        'local_loss': [],
        'spoke_acc': []
    }
    if args.monitor_model_drift:
        metrics['pre_drift'] = []
        metrics['post_drift'] = []

    # For node degrees (only if monitor_degree is enabled)
    degree_dict = {'round': []}
    if args.monitor_degree:
        degree_dict['p2p_indegree'] = []
        degree_dict['p2p_outdegree'] = []
        degree_dict['hsl_stage1_spoke_out'] = []
        degree_dict['hsl_stage2_hub_in'] = []
        degree_dict['hsl_stage3_hub_out'] = []

    def compute_model_drift(spoke_wts_list):
        """
        Compute average distance of each spoke’s parameter vector from the mean,
        useful for monitoring model divergence among spokes.
        """
        stack = torch.stack(spoke_wts_list, dim=0)
        mean_model = torch.mean(stack, dim=0)
        diffs = stack - mean_model
        dists = torch.norm(diffs, dim=1)
        return torch.mean(dists).item()

    # ----------------------------------------------------------------------------------
    # Main processing: either real training or graph simulation
    # ----------------------------------------------------------------------------------
    if not args.graph_simulation_only:
        # ------------------------------------------------------------------------
        # Real Training
        # ------------------------------------------------------------------------
        if args.topo == 'base-graph':
            graph = BaseGraph(args.num_spokes, args.k)
            W_base = graph.w_list
        elif args.topo == 'simple-base-graph':
            graph = SimpleBaseGraph(args.num_spokes, args.k)
            W_simple_base = graph.w_list
        elif args.topo == 'exponential':
            W = utils.create_exponential_graph(args.num_spokes).to(device)

        for rnd in range(args.num_rounds):
            # Step 1: Gather the current global model weights
            global_wts = utils.model_to_vec(global_model)
            node_return = {}

            # Step 2: Each spoke trains locally and returns updated weights
            for node_id in range(args.num_spokes):
                updated_wts = train_node.local_train_worker_inline(
                    node_id,
                    global_wts,
                    distributed_data[node_id],
                    distributed_label[node_id],
                    inp_dim, out_dim, net_name,
                    args.num_local_iters,
                    batch_size, args.lr,
                    aggregator_device,
                    args.sample_type,
                    rr_indices
                )
                node_return[node_id] = updated_wts

            # (Optional) Log pre-aggregation drift
            if args.monitor_model_drift and (rnd + 1) % args.eval_time == 0:
                spoke_wts_list = [node_return[i] for i in range(args.num_spokes)]
                drift_val = compute_model_drift(spoke_wts_list)
                metrics['pre_drift'].append(drift_val)

            # Step 3: Aggregation
            if args.aggregation == 'fedsgd':
                local_models = [node_return[i] for i in range(args.num_spokes)]
                aggregated_wts = aggregation.federated_aggregation(local_models, node_weights)
                # Update global model
                global_model = utils.vec_to_model(
                    aggregated_wts.to(aggregator_device),
                    net_name, inp_dim, out_dim, aggregator_device
                )

                # Evaluate periodically
                if (rnd + 1) % args.eval_time == 0:
                    if args.dataset == 'agnews':
                        traced_model = deepcopy(global_model)
                    else:
                        traced_model = torch.jit.trace(
                            deepcopy(global_model),
                            torch.randn([batch_size] + utils.get_input_shape(args.dataset)).to(aggregator_device)
                        )
                    loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                    metrics['round'].append(rnd + 1)
                    metrics['global_loss'].append(loss)
                    metrics['global_acc'].append(acc)
                    print(f"[Round {rnd+1}] FedSGD => Global Acc: {acc:.4f}, Loss: {loss:.4f}")

                # (Optional) Log post-aggregation drift
                if args.monitor_model_drift and (rnd + 1) % args.eval_time == 0:
                    post_wts_list = [aggregated_wts for _ in range(args.num_spokes)]
                    drift_val = compute_model_drift(post_wts_list)
                    metrics['post_drift'].append(drift_val)

            elif args.aggregation == 'p2p':
                # Stack spoke updates
                spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)])

                # Create adjacency matrix based on the chosen topology
                if args.topo == 'ring':
                    W = utils.create_ring_graph(args.num_spokes, aggregator_device)
                elif args.topo == 'torus':
                    W = utils.create_torus_graph(args.num_spokes, aggregator_device)
                elif args.topo == 'erdos-renyi':
                    W = utils.create_erdos_renyi_graph(args.num_spokes, args.budget, aggregator_device)
                elif args.topo == 'base-graph':
                    W = W_base[rnd % len(W_base)].cuda()
                elif args.topo == 'simple-base-graph': 
                    W = W_simple_base[rnd % len(W_simple_base)].cuda()
                else:
                    W = utils.create_k_random_regular_graph(args.num_spokes, args.k, aggregator_device)

                # Perform p2p averaging
                new_spoke_wts = aggregation.p2p_aggregation(spoke_wts, W)

                # (Optional) Log node degrees if requested
                if args.monitor_degree:
                    indegs = (W.clone().cpu().sum(dim=0)).tolist()
                    outdegs = (W.clone().cpu().sum(dim=1)).tolist()
                    degree_dict['p2p_indegree'].append(indegs)
                    degree_dict['p2p_outdegree'].append(outdegs)

                # Use the first node’s model as "global" for convenience
                global_model = utils.vec_to_model(
                    new_spoke_wts[0], net_name, inp_dim, out_dim, aggregator_device
                )

                # Evaluate local metrics periodically
                if (rnd + 1) % args.eval_time == 0:
                    local_accs, local_losses = [], []
                    for node_id in range(args.num_spokes):
                        node_model = utils.vec_to_model(
                            new_spoke_wts[node_id], net_name, inp_dim, out_dim, aggregator_device
                        )
                        if args.dataset == 'agnews':
                            traced_model = deepcopy(node_model)
                        else:
                            traced_model = torch.jit.trace(
                                deepcopy(node_model),
                                torch.randn([batch_size] + utils.get_input_shape(args.dataset)).to(aggregator_device)
                            )
                        loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                        local_accs.append(acc)
                        local_losses.append(loss)

                    metrics['round'].append(rnd + 1)
                    metrics['local_acc'].append(local_accs)
                    metrics['local_loss'].append(local_losses)
                    print(f"[Round {rnd+1}] P2P => Local Acc range: [{min(local_accs):.4f}, {max(local_accs):.4f}]")

                    # (Optional) Log post-aggregation drift
                    if args.monitor_model_drift:
                        spoke_wts_list = [new_spoke_wts[i] for i in range(args.num_spokes)]
                        drift_val = compute_model_drift(spoke_wts_list)
                        metrics['post_drift'].append(drift_val)

            elif args.aggregation == 'p2p_local':
                # Each spoke merges with neighbors chosen locally in each round
                spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)])
                updated_spoke_wts, W_local = aggregation.p2p_local_aggregation(
                    spoke_wts, args.k, return_W=True
                )

                # (Optional) Log node degrees if requested
                if args.monitor_degree:
                    indegs = (W_local.clone().cpu().sum(dim=0)).tolist()
                    outdegs = (W_local.clone().cpu().sum(dim=1)).tolist()
                    degree_dict['p2p_indegree'].append(indegs)
                    degree_dict['p2p_outdegree'].append(outdegs)

                # Use the first node’s model as "global"
                global_model = utils.vec_to_model(
                    updated_spoke_wts[0], net_name, inp_dim, out_dim, aggregator_device
                )

                # Evaluate local metrics periodically
                if (rnd + 1) % args.eval_time == 0:
                    local_accs, local_losses = [], []
                    for node_id in range(args.num_spokes):
                        node_model = utils.vec_to_model(
                            updated_spoke_wts[node_id], net_name, inp_dim, out_dim, aggregator_device
                        )
                        if args.dataset == 'agnews':
                            traced_model = deepcopy(node_model)
                        else:
                            traced_model = torch.jit.trace(
                                deepcopy(node_model),
                                torch.randn([batch_size] + utils.get_input_shape(args.dataset)).to(aggregator_device)
                            )
                        loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                        local_accs.append(acc)
                        local_losses.append(loss)

                    metrics['round'].append(rnd + 1)
                    metrics['local_acc'].append(local_accs)
                    metrics['local_loss'].append(local_losses)
                    print(f"[Round {rnd+1}] P2P Local => Local Acc range: [{min(local_accs):.4f}, {max(local_accs):.4f}]")

                    # (Optional) Log post-aggregation drift
                    if args.monitor_model_drift:
                        spoke_wts_list = [updated_spoke_wts[i] for i in range(args.num_spokes)]
                        drift_val = compute_model_drift(spoke_wts_list)
                        metrics['post_drift'].append(drift_val)

            elif args.aggregation == 'hsl':
                """
                Hubs and Spokes Learning (HSL):
                  Stage 1: Spokes -> Hubs
                  Stage 2: Hubs perform decentralized mixing (p2p_local)
                  Stage 3: Hubs -> Spokes
                """
                spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)])

                # Stage 1: Spokes -> Hubs
                stage1_matrix = torch.zeros((args.num_hubs, args.num_spokes), device=aggregator_device)
                for hub_id in range(args.num_hubs):
                    if args.b_hs <= args.num_spokes:
                        chosen_spoke_ids = torch.randperm(args.num_spokes, device=aggregator_device)[:args.b_hs]
                    else:
                        chosen_spoke_ids = torch.randint(0, args.num_spokes, (args.b_hs,), device=aggregator_device)
                    for s_id in chosen_spoke_ids:
                        stage1_matrix[hub_id, s_id] = 1.0
                # Row-normalize
                for row_i in range(args.num_hubs):
                    row_sum = torch.sum(stage1_matrix[row_i])
                    if row_sum > 0:
                        stage1_matrix[row_i] /= row_sum

                # Compute hub-level aggregation
                hub_wts = []
                for hub_id in range(args.num_hubs):
                    indices = (stage1_matrix[hub_id] > 0).nonzero(as_tuple=True)[0]
                    if len(indices) > 0:
                        hub_agg = spoke_wts[indices].mean(dim=0)
                    else:
                        hub_agg = torch.zeros_like(spoke_wts[0])
                    hub_wts.append(hub_agg)
                hub_wts = torch.stack(hub_wts)

                # Stage 2: Hubs perform local p2p mixing among themselves
                updated_hub_wts, stage2_matrix = aggregation.p2p_local_aggregation(
                    hub_wts, args.b_hh, return_W=True
                )

                # Stage 3: Hubs -> Spokes
                stage3_matrix = torch.zeros((args.num_spokes, args.num_hubs), device=aggregator_device)
                for spoke_id in range(args.num_spokes):
                    if args.b_sh <= args.num_hubs:
                        chosen_hub_ids = torch.randperm(args.num_hubs, device=aggregator_device)[:args.b_sh]
                    else:
                        chosen_hub_ids = torch.randint(0, args.num_hubs, (args.b_sh,), device=aggregator_device)
                    for h_id in chosen_hub_ids:
                        stage3_matrix[spoke_id, h_id] = 1.0
                # Row-normalize
                for row_i in range(args.num_spokes):
                    row_sum = torch.sum(stage3_matrix[row_i])
                    if row_sum > 0:
                        stage3_matrix[row_i] /= row_sum

                # Final spoke weights after receiving from hubs
                final_spoke_wts = []
                for spoke_id in range(args.num_spokes):
                    indices = (stage3_matrix[spoke_id] > 0).nonzero(as_tuple=True)[0]
                    if len(indices) > 0:
                        spoke_agg = updated_hub_wts[indices].mean(dim=0)
                    else:
                        spoke_agg = torch.zeros_like(updated_hub_wts[0])
                    final_spoke_wts.append(spoke_agg)
                final_spoke_wts = torch.stack(final_spoke_wts)

                # (Optional) Track degrees
                if args.monitor_degree:
                    stage1_T = stage1_matrix.t()
                    spoke_outdeg_stage1 = (stage1_T > 0).sum(dim=1).tolist()
                    hub_indeg_stage2 = (stage2_matrix > 0).sum(dim=0).tolist()
                    hub_outdeg_stage3 = (stage3_matrix > 0).sum(dim=0).tolist()
                    degree_dict['hsl_stage1_spoke_out'].append(spoke_outdeg_stage1)
                    degree_dict['hsl_stage2_hub_in'].append(hub_indeg_stage2)
                    degree_dict['hsl_stage3_hub_out'].append(hub_outdeg_stage3)

                # Update global model (simple average of final spoke weights)
                global_model = utils.vec_to_model(
                    final_spoke_wts.mean(dim=0), net_name, inp_dim, out_dim, aggregator_device
                )

                # Evaluate local spokes periodically
                if (rnd + 1) % args.eval_time == 0:
                    spoke_accs = []
                    for s_id in range(args.num_spokes):
                        s_model = utils.vec_to_model(
                            final_spoke_wts[s_id], net_name, inp_dim, out_dim, aggregator_device
                        )
                        if args.dataset == 'agnews':
                            traced_model = deepcopy(s_model)
                        else:
                            traced_model = torch.jit.trace(
                                deepcopy(s_model),
                                torch.randn([batch_size] + utils.get_input_shape(args.dataset)).to(aggregator_device)
                            )
                        loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                        spoke_accs.append(acc)
                    metrics['round'].append(rnd + 1)
                    metrics['spoke_acc'].append(spoke_accs)
                    print(f"[Round {rnd+1}] HSL => Spoke Acc range: [{min(spoke_accs):.4f}, {max(spoke_accs):.4f}]")

                    # (Optional) Log post-aggregation drift
                    if args.monitor_model_drift:
                        spoke_wts_list = [final_spoke_wts[i] for i in range(args.num_spokes)]
                        drift_val = compute_model_drift(spoke_wts_list)
                        metrics['post_drift'].append(drift_val)

    else:
        # ------------------------------------------------------------------------
        # Graph Simulation Only (no actual training)
        # ------------------------------------------------------------------------
        # These accumulators track spectral gap and edge usage across rounds
        sum_spectral_gap = 0.0
        sum_num_edges = 0
        count_graph_rounds = 0

        for _ in tqdm(range(args.num_rounds), desc="Simulating Graphs"):
            # Dummy dimension for node weights
            d = 10
            spoke_wts = torch.randn(args.num_spokes, d, device=aggregator_device)

            if args.aggregation == 'p2p':
                if args.topo == 'erdos-renyi':
                    W = utils.create_erdos_renyi_graph(args.num_spokes, args.budget, aggregator_device)
                elif args.topo == 'ring':
                    W = utils.create_ring_graph(args.num_spokes, aggregator_device)
                elif args.topo == 'torus':
                    W = utils.create_torus_graph(args.num_spokes, aggregator_device)
                else:
                    W = utils.create_k_random_regular_graph(args.num_spokes, args.k, aggregator_device)

                _ = aggregation.p2p_aggregation(spoke_wts, W)

                # Edge count (exclude self edges for p2p)
                num_edges = (W > 0).sum().item() - len(W)

                # Compute spectral gap
                e = torch.linalg.eigvals(W)
                e_abs = torch.abs(e)
                e_sorted, _ = torch.sort(e_abs, descending=True)
                gap_rnd = (1.0 - e_sorted[1]).item()

                sum_spectral_gap += gap_rnd
                sum_num_edges += num_edges
                count_graph_rounds += 1

            elif args.aggregation == 'p2p_local':
                updated_spoke_wts, W_local = aggregation.p2p_local_aggregation(
                    spoke_wts, args.k, return_W=True
                )
                # Remove diagonal self-edges when counting
                W_copy = W_local.clone()
                for i in range(args.num_spokes):
                    W_copy[i, i] = 0.0
                num_edges = (W_copy > 0).sum().item()

                # Compute spectral gap
                e = torch.linalg.eigvals(W_local)
                e_abs = torch.abs(e)
                e_sorted, _ = torch.sort(e_abs, descending=True)
                gap_rnd = (1.0 - e_sorted[1]).item()

                sum_spectral_gap += gap_rnd
                sum_num_edges += num_edges
                count_graph_rounds += 1

            elif args.aggregation == 'hsl':
                # Stage 1: Spokes->Hubs
                stage1_matrix = torch.zeros((args.num_hubs, args.num_spokes), device=aggregator_device)
                for hub_id in range(args.num_hubs):
                    if args.b_hs <= args.num_spokes:
                        chosen_spoke_ids = torch.randperm(args.num_spokes, device=aggregator_device)[:args.b_hs]
                    else:
                        chosen_spoke_ids = torch.randint(0, args.num_spokes, (args.b_hs,), device=aggregator_device)
                    for s_id in chosen_spoke_ids:
                        stage1_matrix[hub_id, s_id] = 1.0
                for row_i in range(args.num_hubs):
                    row_sum = torch.sum(stage1_matrix[row_i])
                    if row_sum > 0:
                        stage1_matrix[row_i] /= row_sum

                # Stage 2: Hubs->Hubs (p2p_local among hubs)
                hub_wts = torch.randn(args.num_hubs, d, device=aggregator_device)
                _, stage2_matrix = aggregation.p2p_local_aggregation(
                    hub_wts, args.b_hh, return_W=True
                )

                # Stage 3: Hubs->Spokes
                stage3_matrix = torch.zeros((args.num_spokes, args.num_hubs), device=aggregator_device)
                for spoke_id in range(args.num_spokes):
                    if args.b_sh <= args.num_hubs:
                        chosen_hub_ids = torch.randperm(args.num_hubs, device=aggregator_device)[:args.b_sh]
                    else:
                        chosen_hub_ids = torch.randint(0, args.num_hubs, (args.b_sh,), device=aggregator_device)
                    for h_id in chosen_hub_ids:
                        stage3_matrix[spoke_id, h_id] = 1.0
                for row_i in range(args.num_spokes):
                    row_sum = torch.sum(stage3_matrix[row_i])
                    if row_sum > 0:
                        stage3_matrix[row_i] /= row_sum

                # Total directed edges across the three stages
                edges_stage1 = (stage1_matrix > 0).sum().item()
                W_copy = stage2_matrix.clone()
                for i in range(args.num_hubs):
                    W_copy[i, i] = 0.0
                edges_stage2 = (W_copy > 0).sum().item()
                edges_stage3 = (stage3_matrix > 0).sum().item()
                total_edges = edges_stage1 + edges_stage2 + edges_stage3

                # Effective mixing matrix for this round (spokes x spokes)
                W_eff_round = torch.matmul(stage3_matrix, torch.matmul(stage2_matrix, stage1_matrix))
                e = torch.linalg.eigvals(W_eff_round)
                e_abs = torch.abs(e)
                e_sorted, _ = torch.sort(e_abs, descending=True)
                gap_rnd = (1.0 - e_sorted[1]).item()

                sum_spectral_gap += gap_rnd
                sum_num_edges += total_edges
                count_graph_rounds += 1

        # Once simulation is complete, report average spectral gap and edges
        if count_graph_rounds > 0:
            avg_gap = sum_spectral_gap / float(count_graph_rounds)
            avg_edges = sum_num_edges / float(count_graph_rounds)
            metrics['avg_spectral_gap'] = avg_gap
            metrics['avg_num_edges'] = avg_edges
            print(f"[Info] Average spectral gap across rounds: {avg_gap:.6f}")
            print(f"[Info] Average directed edges per round: {avg_edges:.2f}")

    # ----------------------------------------------------------------------------------
    # Save final metrics to a JSON file
    # ----------------------------------------------------------------------------------
    with open(filename + '_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Training or simulation complete.")

    # Save degree dictionary if node-degree monitoring was enabled
    if args.monitor_degree:
        with open(filename + '_degree.json', 'w') as f:
            json.dump(degree_dict, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)
