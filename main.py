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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default='experiment')
    parser.add_argument("--dataset", type=str, default='mnist', choices=['mnist', 'cifar10', 'agnews'])
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

    # NEW #1: A single seed argument
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for data splitting + model init. 0 or negative => random")

    # HSL parameters
    parser.add_argument("--b_hs", type=int, default=2)
    parser.add_argument("--b_hh", type=int, default=1)
    parser.add_argument("--b_sh", type=int, default=2)

    # (1) Monitor model drift
    parser.add_argument("--monitor_model_drift", action='store_true',
                        help="If set, compute average model drift among spokes before/after aggregation.")

    # (5) Monitor degrees
    parser.add_argument("--monitor_degree", action='store_true',
                        help="If set, compute and log node in/out degrees each round.")

    # (6) Graph simulation only
    parser.add_argument("--graph_simulation_only", action='store_true',
                        help="If set, skip actual training. Only simulate mixing matrices per round.")

    return parser.parse_args()

def main(args):
    if args.seed > 0:
        import random
        random.seed(args.seed)
        import numpy as np
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
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

    # Load data if not graph_simulation_only
    if not args.graph_simulation_only:
        trainObject = utils.load_data(args.dataset, args.batch_size, args.lr, args.fraction)
        data, labels = trainObject.train_data, trainObject.train_labels
        test_data = trainObject.test_data
        inp_dim = trainObject.num_inputs
        out_dim = trainObject.num_outputs
        lr = args.lr
        batch_size = trainObject.batch_size
        net_name = trainObject.net_name

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
        distributed_data = distributedData.distributed_input
        distributed_label = distributedData.distributed_output
        node_weights = distributedData.wts

        num_samples_per_spoke = [len(distributed_data[i]) for i in range(args.num_spokes)]
        min_size = min(num_samples_per_spoke)
        max_size = max(num_samples_per_spoke)
        mean_size = sum(num_samples_per_spoke)/len(num_samples_per_spoke)
        var_size = sum((s - mean_size)**2 for s in num_samples_per_spoke)/len(num_samples_per_spoke)
        sd_size = math.sqrt(var_size)
        print(f"[Data Dist] Min={min_size}, Max={max_size}, Mean={mean_size:.1f}, SD={sd_size:.1f}")

        global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)

        rr_indices = {}
        for node_id in range(args.num_spokes):
            ds_size = distributed_data[node_id].shape[0]
            if ds_size > 0:
                perm = torch.randperm(ds_size)
                distributed_data[node_id] = distributed_data[node_id][perm]
                distributed_label[node_id] = distributed_label[node_id][perm]
            rr_indices[node_id] = 0
    else:
        test_data = None
        inp_dim, out_dim = 1, 1
        net_name = 'lenet'
        batch_size = 32
        node_weights = torch.ones(args.num_spokes)/args.num_spokes
        global_model = models.load_net(net_name, inp_dim, out_dim, aggregator_device)

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

    # Keep old accumulators (for backward compatibility, no functionality change):
    W_accumulator = None
    count_W = 0
    W_hs_acc = None
    W_hh_acc = None
    W_sh_acc = None
    count_hsl = 0

    # For node degrees
    degree_dict = {'round': []}
    if args.monitor_degree:
        degree_dict['p2p_indegree'] = []
        degree_dict['p2p_outdegree'] = []
        degree_dict['hsl_stage1_spoke_out'] = []
        degree_dict['hsl_stage2_hub_in'] = []
        degree_dict['hsl_stage3_hub_out'] = []

    def compute_model_drift(spoke_wts_list):
        stack = torch.stack(spoke_wts_list, dim=0)
        mean_model = torch.mean(stack, dim=0)
        diffs = stack - mean_model
        dists = torch.norm(diffs, dim=1)
        return torch.mean(dists).item()

    ##################################################################
    # NEW Variables for round-based spectral gap & edge counting
    ##################################################################
    sum_spectral_gap = 0.0
    sum_num_edges = 0
    count_graph_rounds = 0

    # Actual training or graph simulation
    if not args.graph_simulation_only:
        # Real training
        for rnd in range(args.num_rounds):
            global_wts = utils.model_to_vec(global_model).cpu()
            node_return = {}

            # Local training
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

            # Pre-drift
            if args.monitor_model_drift and (rnd+1) % args.eval_time == 0:
                spoke_wts_list = [node_return[i].cpu() for i in range(args.num_spokes)]
                drift_val = compute_model_drift(spoke_wts_list)
                metrics['pre_drift'].append(drift_val)

            # Aggregation step
            if args.aggregation == 'fedsgd':
                local_models = [node_return[i] for i in range(args.num_spokes)]
                aggregated_wts = aggregation.federated_aggregation(local_models, node_weights)
                global_model = utils.vec_to_model(aggregated_wts.to(aggregator_device),
                                                  net_name, inp_dim, out_dim, aggregator_device)

                if (rnd+1) % args.eval_time == 0:
                    if args.dataset == 'agnews':
                        traced_model = deepcopy(global_model)
                    else:
                        traced_model = torch.jit.trace(
                            deepcopy(global_model),
                            torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device)
                        )
                    loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                    metrics['round'].append(rnd+1)
                    metrics['global_loss'].append(loss)
                    metrics['global_acc'].append(acc)
                    print(f"[Round {rnd+1}] FedSGD => Global Acc: {acc:.4f}, Loss: {loss:.4f}")

                if args.monitor_model_drift and (rnd+1) % args.eval_time == 0:
                    post_wts_list = [aggregated_wts.cpu() for _ in range(args.num_spokes)]
                    drift_val = compute_model_drift(post_wts_list)
                    metrics['post_drift'].append(drift_val)

            elif args.aggregation == 'p2p':
                spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)]).to(aggregator_device)
                W = utils.create_k_random_regular_graph(args.num_spokes, args.k, aggregator_device)
                new_spoke_wts = aggregation.p2p_aggregation(spoke_wts, W)

                # Old average matrix accumulation
                if W_accumulator is None:
                    W_accumulator = W.clone().cpu()
                else:
                    W_accumulator += W.clone().cpu()
                count_W += 1

                # Node degrees
                if args.monitor_degree:
                    indegs = (W.clone().cpu().sum(dim=0)).tolist()
                    outdegs = (W.clone().cpu().sum(dim=1)).tolist()
                    degree_dict['p2p_indegree'].append(indegs)
                    degree_dict['p2p_outdegree'].append(outdegs)

                # Global model
                global_model = utils.vec_to_model(new_spoke_wts[0], net_name, inp_dim, out_dim, aggregator_device)

                # Count edges & spectral gap for this round's W
                # 1) Edges
                num_edges = (W > 0).sum().item()
                # 2) Spectral gap
                e = torch.linalg.eigvals(W)
                e_abs = torch.abs(e)
                e_sorted, _ = torch.sort(e_abs, descending=True)
                gap_rnd = (1.0 - e_sorted[1]).item()
                sum_spectral_gap += gap_rnd
                sum_num_edges += num_edges
                count_graph_rounds += 1

                if (rnd+1) % args.eval_time == 0:
                    local_accs, local_losses = [], []
                    for node_id in range(args.num_spokes):
                        node_model = utils.vec_to_model(new_spoke_wts[node_id], net_name, inp_dim, out_dim, aggregator_device)
                        if args.dataset == 'agnews':
                            traced_model = deepcopy(node_model)
                        else:
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

                    if args.monitor_model_drift:
                        spoke_wts_list = [new_spoke_wts[i].cpu() for i in range(args.num_spokes)]
                        drift_val = compute_model_drift(spoke_wts_list)
                        metrics['post_drift'].append(drift_val)

            elif args.aggregation == 'p2p_local':
                spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)]).to(aggregator_device)
                updated_spoke_wts, W_local = aggregation.p2p_local_aggregation(spoke_wts, args.k, return_W=True)

                # Old accumulators
                if W_accumulator is None:
                    W_accumulator = W_local.clone().cpu()
                else:
                    W_accumulator += W_local.clone().cpu()
                count_W += 1

                # Degrees
                if args.monitor_degree:
                    indegs = (W_local.clone().cpu().sum(dim=0)).tolist()
                    outdegs = (W_local.clone().cpu().sum(dim=1)).tolist()
                    degree_dict['p2p_indegree'].append(indegs)
                    degree_dict['p2p_outdegree'].append(outdegs)

                # Global model from the first spoke
                global_model = utils.vec_to_model(updated_spoke_wts[0], net_name, inp_dim, out_dim, aggregator_device)

                # (A) Count edges for p2p_local but exclude self-edges
                W_copy = W_local.clone()
                # zero out the diagonal to exclude self-edges from counting
                for i in range(args.num_spokes):
                    W_copy[i, i] = 0.0
                num_edges = (W_copy > 0).sum().item()

                # (B) Compute spectral gap for W_local (full matrix):
                e = torch.linalg.eigvals(W_local)
                e_abs = torch.abs(e)
                e_sorted, _ = torch.sort(e_abs, descending=True)
                gap_rnd = (1.0 - e_sorted[1]).item()

                sum_spectral_gap += gap_rnd
                sum_num_edges += num_edges
                count_graph_rounds += 1

                if (rnd+1) % args.eval_time == 0:
                    local_accs, local_losses = [], []
                    for node_id in range(args.num_spokes):
                        node_model = utils.vec_to_model(updated_spoke_wts[node_id], net_name, inp_dim, out_dim, aggregator_device)
                        if args.dataset == 'agnews':
                            traced_model = deepcopy(node_model)
                        else:
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

                    if args.monitor_model_drift:
                        spoke_wts_list = [updated_spoke_wts[i].cpu() for i in range(args.num_spokes)]
                        drift_val = compute_model_drift(spoke_wts_list)
                        metrics['post_drift'].append(drift_val)

            elif args.aggregation == 'hsl':
                spoke_wts = torch.stack([node_return[i] for i in range(args.num_spokes)]).to(aggregator_device)
                # Step 1
                stage1_matrix = torch.zeros((args.num_hubs, args.num_spokes), device=aggregator_device)
                for hub_id in range(args.num_hubs):
                    if args.b_hs <= args.num_spokes:
                        chosen_spoke_ids = torch.randperm(args.num_spokes, device=aggregator_device)[:args.b_hs]
                    else:
                        chosen_spoke_ids = torch.randint(low=0, high=args.num_spokes, size=(args.b_hs,),
                                                         device=aggregator_device)
                    for s_id in chosen_spoke_ids:
                        stage1_matrix[hub_id, s_id] = 1.0
                for row_i in range(args.num_hubs):
                    row_sum = torch.sum(stage1_matrix[row_i])
                    if row_sum > 0:
                        stage1_matrix[row_i] /= row_sum
                hub_wts = []
                for hub_id in range(args.num_hubs):
                    indices = (stage1_matrix[hub_id] > 0).nonzero(as_tuple=True)[0]
                    if len(indices) > 0:
                        hub_agg = spoke_wts[indices].mean(dim=0)
                    else:
                        hub_agg = torch.zeros_like(spoke_wts[0])
                    hub_wts.append(hub_agg)
                hub_wts = torch.stack(hub_wts)

                # Step 2
                updated_hub_wts, stage2_matrix = aggregation.p2p_local_aggregation(hub_wts, args.b_hh, return_W=True)

                # Step 3
                stage3_matrix = torch.zeros((args.num_spokes, args.num_hubs), device=aggregator_device)
                final_spoke_wts = []
                for spoke_id in range(args.num_spokes):
                    if args.b_sh <= args.num_hubs:
                        chosen_hub_ids = torch.randperm(args.num_hubs, device=aggregator_device)[:args.b_sh]
                    else:
                        chosen_hub_ids = torch.randint(low=0, high=args.num_hubs, size=(args.b_sh,),
                                                       device=aggregator_device)
                    for h_id in chosen_hub_ids:
                        stage3_matrix[spoke_id, h_id] = 1.0
                for row_i in range(args.num_spokes):
                    row_sum = torch.sum(stage3_matrix[row_i])
                    if row_sum > 0:
                        stage3_matrix[row_i] /= row_sum
                for spoke_id in range(args.num_spokes):
                    indices = (stage3_matrix[spoke_id] > 0).nonzero(as_tuple=True)[0]
                    if len(indices) > 0:
                        spoke_agg = updated_hub_wts[indices].mean(dim=0)
                    else:
                        spoke_agg = torch.zeros_like(updated_hub_wts[0])
                    final_spoke_wts.append(spoke_agg)
                final_spoke_wts = torch.stack(final_spoke_wts)

                if W_hs_acc is None:
                    W_hs_acc = stage1_matrix.clone().cpu()
                    W_hh_acc = stage2_matrix.clone().cpu()
                    W_sh_acc = stage3_matrix.clone().cpu()
                else:
                    W_hs_acc += stage1_matrix.clone().cpu()
                    W_hh_acc += stage2_matrix.clone().cpu()
                    W_sh_acc += stage3_matrix.clone().cpu()
                count_hsl += 1

                if args.monitor_degree:
                    stage1_T = stage1_matrix.t()
                    spoke_outdeg_stage1 = (stage1_T > 0).sum(dim=1).tolist()
                    hub_indeg_stage2 = (stage2_matrix > 0).sum(dim=0).tolist()
                    hub_outdeg_stage3 = (stage3_matrix > 0).sum(dim=0).tolist()
                    degree_dict['hsl_stage1_spoke_out'].append(spoke_outdeg_stage1)
                    degree_dict['hsl_stage2_hub_in'].append(hub_indeg_stage2)
                    degree_dict['hsl_stage3_hub_out'].append(hub_outdeg_stage3)

                global_model = utils.vec_to_model(final_spoke_wts.mean(dim=0),
                                                  net_name, inp_dim, out_dim, aggregator_device)

                # (A) Count total directed edges in HSL for this round
                edges_stage1 = (stage1_matrix > 0).sum().item()
                edges_stage2 = (stage2_matrix > 0).sum().item()
                edges_stage3 = (stage3_matrix > 0).sum().item()
                total_edges = edges_stage1 + edges_stage2 + edges_stage3

                # (B) Effective matrix W_eff for that round
                #   shape (spokes x spokes) = stage3_matrix (spokes x hubs) *
                #                             stage2_matrix (hubs x hubs) *
                #                             stage1_matrix (hubs x spokes)
                W_eff_round = torch.matmul(stage3_matrix,
                               torch.matmul(stage2_matrix, stage1_matrix))
                e = torch.linalg.eigvals(W_eff_round)
                e_abs = torch.abs(e)
                e_sorted, _ = torch.sort(e_abs, descending=True)
                gap_rnd = (1.0 - e_sorted[1]).item()

                sum_spectral_gap += gap_rnd
                sum_num_edges += total_edges
                count_graph_rounds += 1

                if (rnd+1) % args.eval_time == 0:
                    spoke_accs = []
                    for s_id in range(args.num_spokes):
                        s_model = utils.vec_to_model(final_spoke_wts[s_id], net_name, inp_dim, out_dim, aggregator_device)
                        if args.dataset == 'agnews':
                            traced_model = deepcopy(s_model)
                        else:
                            traced_model = torch.jit.trace(
                                deepcopy(s_model),
                                torch.randn([batch_size]+utils.get_input_shape(args.dataset)).to(aggregator_device)
                            )
                        loss, acc = utils.evaluate_global_metrics(traced_model, test_data, aggregator_device)
                        spoke_accs.append(acc)
                    metrics['round'].append(rnd+1)
                    metrics['spoke_acc'].append(spoke_accs)
                    print(f"[Round {rnd+1}] HSL => Spoke Acc range: [{min(spoke_accs):.4f}, {max(spoke_accs):.4f}]")

                    if args.monitor_model_drift:
                        spoke_wts_list = [final_spoke_wts[i].cpu() for i in range(args.num_spokes)]
                        drift_val = compute_model_drift(spoke_wts_list)
                        metrics['post_drift'].append(drift_val)

    else:
        # Graph simulation only
        for rnd in tqdm(range(args.num_rounds)):
            d = 10
            spoke_wts = torch.randn(args.num_spokes, d, device=aggregator_device)

            if args.aggregation == 'p2p':
                W = utils.create_k_random_regular_graph(args.num_spokes, args.k, aggregator_device)
                _ = aggregation.p2p_aggregation(spoke_wts, W)

                if W_accumulator is None:
                    W_accumulator = W.clone().cpu()
                else:
                    W_accumulator += W.clone().cpu()
                count_W += 1

                # Edge + spectral gap this round
                num_edges = (W > 0).sum().item()
                e = torch.linalg.eigvals(W)
                e_abs = torch.abs(e)
                e_sorted, _ = torch.sort(e_abs, descending=True)
                gap_rnd = (1.0 - e_sorted[1]).item()

                sum_spectral_gap += gap_rnd
                sum_num_edges += num_edges
                count_graph_rounds += 1

            elif args.aggregation == 'p2p_local':
                updated_spoke_wts, W_local = aggregation.p2p_local_aggregation(spoke_wts, args.k, return_W=True)
                if W_accumulator is None:
                    W_accumulator = W_local.clone().cpu()
                else:
                    W_accumulator += W_local.clone().cpu()
                count_W += 1

                # Exclude diagonal self-edges
                W_copy = W_local.clone()
                for i in range(args.num_spokes):
                    W_copy[i, i] = 0.0
                num_edges = (W_copy > 0).sum().item()
                

                e = torch.linalg.eigvals(W_local)
                e_abs = torch.abs(e)
                e_sorted, _ = torch.sort(e_abs, descending=True)
                gap_rnd = (1.0 - e_sorted[1]).item()

                sum_spectral_gap += gap_rnd
                sum_num_edges += num_edges
                count_graph_rounds += 1

            elif args.aggregation == 'hsl':
                stage1_matrix = torch.zeros((args.num_hubs, args.num_spokes), device=aggregator_device)
                for hub_id in range(args.num_hubs):
                    if args.b_hs <= args.num_spokes:
                        chosen_spoke_ids = torch.randperm(args.num_spokes, device=aggregator_device)[:args.b_hs]
                    else:
                        chosen_spoke_ids = torch.randint(low=0, high=args.num_spokes, size=(args.b_hs,),
                                                         device=aggregator_device)
                    for s_id in chosen_spoke_ids:
                        stage1_matrix[hub_id, s_id] = 1.0
                for row_i in range(args.num_hubs):
                    row_sum = torch.sum(stage1_matrix[row_i])
                    if row_sum > 0:
                        stage1_matrix[row_i] /= row_sum

                hub_wts = torch.randn(args.num_hubs, d, device=aggregator_device)
                _, stage2_matrix = aggregation.p2p_local_aggregation(hub_wts, args.b_hh, return_W=True)

                stage3_matrix = torch.zeros((args.num_spokes, args.num_hubs), device=aggregator_device)
                for spoke_id in range(args.num_spokes):
                    if args.b_sh <= args.num_hubs:
                        chosen_hub_ids = torch.randperm(args.num_hubs, device=aggregator_device)[:args.b_sh]
                    else:
                        chosen_hub_ids = torch.randint(low=0, high=args.num_hubs, size=(args.b_sh,),
                                                       device=aggregator_device)
                    for h_id in chosen_hub_ids:
                        stage3_matrix[spoke_id, h_id] = 1.0
                for row_i in range(args.num_spokes):
                    row_sum = torch.sum(stage3_matrix[row_i])
                    if row_sum > 0:
                        stage3_matrix[row_i] /= row_sum

                if W_hs_acc is None:
                    W_hs_acc = stage1_matrix.clone().cpu()
                    W_hh_acc = stage2_matrix.clone().cpu()
                    W_sh_acc = stage3_matrix.clone().cpu()
                else:
                    W_hs_acc += stage1_matrix.clone().cpu()
                    W_hh_acc += stage2_matrix.clone().cpu()
                    W_sh_acc += stage3_matrix.clone().cpu()
                count_hsl += 1

                # Edge count
                edges_stage1 = (stage1_matrix > 0).sum().item()

                W_copy = stage2_matrix.clone()
                for i in range(args.num_hubs):
                    W_copy[i, i] = 0.0
                edges_stage2 = (W_copy > 0).sum().item()

                #edges_stage2 = (stage2_matrix > 0).sum().item()
                edges_stage3 = (stage3_matrix > 0).sum().item()
                total_edges = edges_stage1 + edges_stage2 + edges_stage3

                # Effective matrix
                W_eff_round = torch.matmul(stage3_matrix,
                                torch.matmul(stage2_matrix, stage1_matrix))
                e = torch.linalg.eigvals(W_eff_round)
                e_abs = torch.abs(e)
                e_sorted, _ = torch.sort(e_abs, descending=True)
                gap_rnd = (1.0 - e_sorted[1]).item()

                sum_spectral_gap += gap_rnd
                sum_num_edges += total_edges
                count_graph_rounds += 1

    ##################################################################
    # FINAL: Report average spectral gap & average #edges
    ##################################################################
    if count_graph_rounds > 0:
        avg_gap = sum_spectral_gap / float(count_graph_rounds)
        avg_edges = sum_num_edges / float(count_graph_rounds)
        metrics['avg_spectral_gap'] = avg_gap
        metrics['avg_num_edges'] = avg_edges
        print(f"[Info] Round-based average spectral gap: {avg_gap:.6f}")
        print(f"[Info] Round-based average directed edges: {avg_edges:.2f}")

    # The old code that computed average mixing matrix remains, but we no longer use it to get spectral_gap.
    # We'll keep it for compatibility (no functionality removed):
    if args.aggregation in ['p2p','p2p_local']:
        if count_W > 0 and W_accumulator is not None:
            W_avg = W_accumulator / float(count_W)
            # We'll still store it in metrics, but not compute spectral gap from it.
            metrics['avg_mixing_matrix'] = W_avg.tolist()
            print(f"[Info] {args.aggregation}: average mixing matrix stored (no spectral gap).")

    elif args.aggregation == 'hsl':
        if count_hsl > 0 and W_hs_acc is not None:
            W_hs_avg = W_hs_acc / float(count_hsl)
            W_hh_avg = W_hh_acc / float(count_hsl)
            W_sh_avg = W_sh_acc / float(count_hsl)
            metrics['W_hs_avg'] = W_hs_avg.tolist()
            metrics['W_hh_avg'] = W_hh_avg.tolist()
            metrics['W_sh_avg'] = W_sh_avg.tolist()
            print(f"[Info] HSL: average stage matrices stored (no spectral gap).")

    # Save metrics
    with open(filename + '_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Training (or simulation) complete.")

    if args.monitor_degree:
        with open(filename + '_degree.json', 'w') as f:
            json.dump(degree_dict, f, indent=2)

if __name__ == "__main__":
    args = parse_args()
    main(args)
